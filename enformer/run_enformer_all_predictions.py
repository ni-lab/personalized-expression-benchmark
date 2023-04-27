import tensorflow as tf
# Make sure the GPU is enabled
assert tf.config.list_physical_devices('GPU')

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

PRED_PATH = "preds"
transform_path = 'gs://dm-enformer/models/enformer.finetuned.SAD.robustscaler-PCA500-robustscaler.transform.pkl'
model_path = 'https://tfhub.dev/deepmind/enformer/1'
GENE_FILE = '/clusterfs/nilah/connie/enformer/data/eur_eqtl_genes_converted.csv'

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


class EnformerScoreVariantsRaw:

    def __init__(self, tfhub_url, organism='human'):
        self._model = Enformer(tfhub_url)
        self._organism = organism

    def predict_on_batch(self, inputs):
        ref_prediction = self._model.predict_on_batch(inputs['ref'])[self._organism]
        alt_prediction = self._model.predict_on_batch(inputs['alt'])[self._organism]

        return alt_prediction.mean(axis=1) - ref_prediction.mean(axis=1)


class EnformerScoreVariantsNormalized:

    def __init__(self, tfhub_url, transform_pkl_path,
                 organism='human'):
        assert organism == 'human', 'Transforms only compatible with organism=human'
        self._model = EnformerScoreVariantsRaw(tfhub_url, organism)
        with tf.io.gfile.GFile(transform_pkl_path, 'rb') as f:
            transform_pipeline = joblib.load(f)
        self._transform = transform_pipeline.steps[0][1]  # StandardScaler.

    def predict_on_batch(self, inputs):
        scores = self._model.predict_on_batch(inputs)
        return self._transform.transform(scores)


class EnformerScoreVariantsPCANormalized:

    def __init__(self, tfhub_url, transform_pkl_path,
                 organism='human', num_top_features=500):
        self._model = EnformerScoreVariantsRaw(tfhub_url, organism)
        with tf.io.gfile.GFile(transform_pkl_path, 'rb') as f:
            self._transform = joblib.load(f)
        self._num_top_features = num_top_features

    def predict_on_batch(self, inputs):
        scores = self._model.predict_on_batch(inputs)
        return self._transform.transform(scores)[:, :self._num_top_features]


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


def variant_generator(vcf_file, gzipped=False):
    """Yields a kipoiseq.dataclasses.Variant for each row in VCF file."""

    def _open(file):
        return gzip.open(vcf_file, 'rt') if gzipped else open(vcf_file)

    with _open(vcf_file) as f:
        for line in f:
            if line.startswith('#'):
                continue
            chrom, pos, id, ref, alt_list = line.split('\t')[:5]
            # Split ALT alleles and return individual variants as output.
            for alt in alt_list.split(','):
                yield kipoiseq.dataclasses.Variant(chrom=chrom, pos=pos,
                                                   ref=ref, alt=alt, id=id)


def one_hot_encode(sequence):
    return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)


def variant_centered_sequences(vcf_file, sequence_length, gzipped=False,
                               chr_prefix=''):
    seq_extractor = kipoiseq.extractors.VariantSeqExtractor(
        reference_sequence=FastaStringExtractor(fasta_file))

    for variant in variant_generator(vcf_file, gzipped=gzipped):
        interval = Interval(chr_prefix + variant.chrom,
                            variant.pos, variant.pos)
        interval = interval.resize(sequence_length)
        center = interval.center() - interval.start

        reference = seq_extractor.extract(interval, [], anchor=center)
        alternate = seq_extractor.extract(interval, [variant], anchor=center)

        yield {'inputs': {'ref': one_hot_encode(reference),
                          'alt': one_hot_encode(alternate)},
               'metadata': {'chrom': chr_prefix + variant.chrom,
                            'pos': variant.pos,
                            'id': variant.id,
                            'ref': variant.ref,
                            'alt': variant.alt}}

def get_fasta(geneId, sample, copy):
    FASTA_DIR = '/clusterfs/nilah/personalized_expression/consensus_seqs/enformer'
    return f"{FASTA_DIR}/{gene['name']}/samples/{s}.{copy}pIu.fa"

def get_gene_df():
    gene_df = pd.read_csv(GENE_FILE, names=["geneId", "chr", "tss", "name", "strand"])
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
    model = Enformer(model_path)
    gene_df = get_gene_df() 
    os.makedirs("preds")

    for i, gene in tqdm(gene_df.iterrows(), total=gene_df.shape[0]):
        chr = int(gene["chr"])
        start = int(gene["tss"]) - SEQUENCE_LENGTH // 2
        end = int(gene["tss"]) + SEQUENCE_LENGTH // 2 - 1
        chr_interval = f'chr{chr}:{start}-{end}'
        preds_file_name = f"{PRED_PATH}/{gene['name']}/{gene['name']}.h5"
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
        if not os.path.exists(f"{PRED_PATH}/{gene['name']}"):
            os.makedirs(f"{PRED_PATH}/{gene['name']}")
        h5_file = h5py.File(preds_file_name, "w")
        record_ids = h5_file.create_dataset("record_ids", data=get_record_ids(chr_interval, gene["strand"], samples))
        preds = np.zeros((846, 896), dtype="float32")
        index = 0
        print(f"start {gene['name']}")
        for s in samples:
            for copy in [1, 2]:
                file = get_fasta(gene["geneId"], s, copy)
                fasta_extractor = FastaStringExtractor(file)
                target_interval = kipoiseq.Interval(chr_interval, 0, SEQUENCE_LENGTH)
                sequence_one_hot = one_hot_encode(fasta_extractor.extract(target_interval.resize(SEQUENCE_LENGTH)).upper())
                predictions = model.predict_on_batch(sequence_one_hot[np.newaxis])['human'][0]
                preds[index] = predictions[:, 5110]
                index += 1

        dset = h5_file.create_dataset("preds", data=preds)
        print(f"success {gene['name']}")
        h5_file.close()

    
