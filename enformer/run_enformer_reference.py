from optparse import OptionParser
import tensorflow as tf
# Make sure the GPU is enabled
# assert tf.config.list_physical_devices('GPU')

import tensorflow_hub as hub
import joblib
import gzip
import kipoiseq
from kipoiseq import Interval
import pyfaidx
import pandas as pd
import numpy as np
import shutil
import gzip
import multiprocessing as mp
from itertools import repeat, product
from functools import partial
import os, io

transform_path = 'gs://dm-enformer/models/enformer.finetuned.SAD.robustscaler-PCA500-robustscaler.transform.pkl'
model_path = 'https://tfhub.dev/deepmind/enformer/1'
clinvar_vcf = '/clusterfs/nilah/connie/enformer/clinvar.vcf.gz'
GENE_FILE = '/clusterfs/nilah/connie/enformer/data/eur_eqtl_genes_converted.csv'
GENE_EXP_FILE = '/clusterfs/nilah/connie/enformer/GD462.GeneQuantRPKM.50FN.samplename.resk10.txt.gz'

targets_txt = 'https://raw.githubusercontent.com/calico/basenji/master/manuscripts/cross2020/targets_human.txt'
df_targets = pd.read_csv(targets_txt, sep='\t')
# DF targets row 5110 is lymphoblastoid cell line

SEQUENCE_LENGTH = 393216
INTERVAL = 114688


class Enformer:

    def __init__(self, tfhub_url):
        self._model = hub.load(tfhub_url).model

    def predict_on_batch(self, inputs):
        predictions = self._model.predict_on_batch(inputs)
        return {k: v.numpy() for k, v in predictions.items()}

    @tf.function
    def contribution_input_grad(self, input_sequence,
                                target_mask, output_head='human'):
        input_sequence = input_sequence[tf.newaxis]

        target_mask_mass = tf.reduce_sum(target_mask)
        with tf.GradientTape() as tape:
            tape.watch(input_sequence)
            prediction = tf.reduce_sum(
                target_mask[tf.newaxis] *
                self._model.predict_on_batch(input_sequence)[output_head]) / target_mask_mass

        input_grad = tape.gradient(prediction, input_sequence) * input_sequence
        input_grad = tf.squeeze(input_grad, axis=0)
        return tf.reduce_sum(input_grad, axis=-1)


class FastaStringExtractor:

    def __init__(self, fasta_file):
        self.fasta = pyfaidx.Fasta(fasta_file)
        self._chromosome_sizes = {k: len(v) for k, v in self.fasta.items()}

    def extract(self, interval: Interval, **kwargs) -> str:
        # Truncate interval if it extends beyond the chromosome lengths.
        chromosome_length = self._chromosome_sizes[interval.chrom]
        trimmed_interval = Interval(interval.chrom,
                                    max(interval.start, 0),
                                    min(interval.end, chromosome_length),
                                    )
        # pyfaidx wants a 1-based interval
        sequence = str(self.fasta.get_seq(trimmed_interval.chrom,
                                          trimmed_interval.start + 1,
                                          trimmed_interval.stop).seq).upper()
        # Fill truncated values with N's.
        pad_upstream = 'N' * max(-interval.start, 0)
        pad_downstream = 'N' * max(interval.end - chromosome_length, 0)
        return pad_upstream + sequence + pad_downstream

    def close(self):
        return self.fasta.close()

def one_hot_encode(sequence):
    return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)

if __name__=="__main__":
    """
    Predict expression for all genes using their reference genome.

    Arguments:
    - consensus_dir: directory containing consensus and reference sequences for each gene
    - genes_csv: file containing Ensembl gene IDs, chromosome, TSS position, gene symbol, and strand
    """
    usage = "usage: %prog [options] <consensus_dir> <genes_csv>"
    parser = OptionParser(usage)
    parser.add_option("-o", dest="out_dir",
                      default='preds',
                      type=str,
                      help="Output directory for predictions [Default: %default]")
    (options, args) = parser.parse_args()

    num_expected_args = 2
    if len(args) != num_expected_args:
        parser.error(
            "Incorrect number of arguments, expected {} arguments but got {}".format(num_expected_args, len(args)))

    # Setup
    consensus_dir = args[0]
    genes_file = args[1]

    model = Enformer(model_path)

    gene_df = pd.read_csv(genes_file, names=["geneId", "chr", "tss", "name", "strand"])
    print("## Starting predictions ##")

    outFile = open(options.out_dir, "w")
    outFile.write("geneId,mean,tss3,tss10,max\n")

    for i, row in gene_df.iterrows():
        chr = int(row["chr"])
        start = int(row["tss"]) - SEQUENCE_LENGTH // 2
        end = int(row["tss"]) + SEQUENCE_LENGTH // 2 - 1
        file = f"{consensus_dir}/chr{chr}.fa"
        fasta_extractor = FastaStringExtractor(file)
        target_interval = kipoiseq.Interval(f'chr{chr}', start, end)
        sequence_one_hot = one_hot_encode(fasta_extractor.extract(target_interval.resize(SEQUENCE_LENGTH)))
        predictions = model.predict_on_batch(sequence_one_hot[np.newaxis])['human'][0][:, 5110]
        outFile.write(f"{row['geneId']},{np.mean(predictions)},{np.mean(predictions[INTERVAL//128//2 - 1: INTERVAL//128//2 + 2])},{np.mean(predictions[INTERVAL//128//2 - 5: INTERVAL//128//2 + 5])},{np.max(predictions)}\n")
    outFile.close()
