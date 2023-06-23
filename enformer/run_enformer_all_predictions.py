import tensorflow as tf
# Make sure the GPU is enabled
assert tf.config.list_physical_devices('GPU')
from optparse import OptionParser

import tensorflow_hub as hub
import joblib
import gzip
import kipoiseq
from kipoiseq import Interval
import pyfaidx
import h5py
import pandas as pd
import numpy as np
import shutil
import gzip
from tqdm import tqdm
import multiprocessing as mp
from itertools import repeat, product
from functools import partial
import os, io, sys

transform_path = 'gs://dm-enformer/models/enformer.finetuned.SAD.robustscaler-PCA500-robustscaler.transform.pkl'
model_path = 'https://tfhub.dev/deepmind/enformer/1'

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


def get_fasta(consensus_dir, geneId, sample, copy):
    return f"{consensus_dir}/{gene['name']}/samples/{s}.{copy}pIu.fa"

def get_gene_df(genes_file):
    gene_df = pd.read_csv(genes_file, names=["geneId", "chr", "tss", "name", "strand"])
    gene_df["name"] = gene_df.apply(lambda x: x["name"] if not any(x.isnull()) else x["geneId"], axis=1).str.lower()
    gene_df = gene_df.sort_values(by="name")
    return gene_df

def get_record_ids(chr_interval, strand, samples):
    r_ids = []
    for s in samples:
        r_ids.extend([f"{chr_interval}|{s}|{strand}|1pIu", f"{chr_interval}|{s}|{strand}|2pIu"])
    return pd.Series(data=r_ids, dtype="S40")

with open("/clusterfs/nilah/connie/enformer/data/same_length_inds.txt", "r") as f:
    samples = f.read().split("\n")[:-1]

if __name__=="__main__":
    """
    Predict expression for all genes and each individual gene using their personalized input sequences.

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
    gene_df = get_gene_df(genes_file) 
    os.makedirs(options.out_dir)

    for i, gene in tqdm(gene_df.iterrows(), total=gene_df.shape[0]):
        chr = int(gene["chr"])
        start = int(gene["tss"]) - SEQUENCE_LENGTH // 2
        end = int(gene["tss"]) + SEQUENCE_LENGTH // 2 - 1
        chr_interval = f'chr{chr}:{start}-{end}'
        preds_file_name = f"{options.out_dir}/{gene['name']}/{gene['name']}.h5"
        print(preds_file_name)
        if os.path.exists(preds_file_name):
            try:
                f = h5py.File(preds_file_name, "r")
                if len(list(f.keys())) == 2:
                    f.close()
                    print("done")
                    continue
            except OSError:
                print("continue")
        if not os.path.exists(f"{options.out_dir}/{gene['name']}"):
            os.makedirs(f"{options.out_dir}/{gene['name']}")
        h5_file = h5py.File(preds_file_name, "w")
        record_ids = h5_file.create_dataset("record_ids", data=get_record_ids(chr_interval, gene["strand"], samples))
        preds = np.zeros((846, 896), dtype="float32")
        index = 0
        print(f"start {gene['name']}")
        for s in samples:
            for copy in [1, 2]:
                file = get_fasta(consensus_dir, gene["geneId"], s, copy)
                fasta_extractor = FastaStringExtractor(file)
                target_interval = kipoiseq.Interval(chr_interval, 0, SEQUENCE_LENGTH)
                sequence_one_hot = one_hot_encode(fasta_extractor.extract(target_interval.resize(SEQUENCE_LENGTH)).upper())
                predictions = model.predict_on_batch(sequence_one_hot[np.newaxis])['human'][0]
                preds[index] = predictions[:, 5110]
                index += 1

        dset = h5_file.create_dataset("preds", data=preds)
        print(f"success {gene['name']}")
        h5_file.close()

    
