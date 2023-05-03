#!/usr/bin/env python

from __future__ import print_function

import glob
import gzip
import json
import os
from optparse import OptionParser
from pathlib import Path
from typing import Generator, List

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from basenji import seqnn, stream
from basenji2_utils import *
from basenji.dna_io import dna_1hot
from Bio import SeqIO
from natsort import natsorted
from tqdm import tqdm


def main():
    usage = 'usage: %prog [options] <params_file> <model_file> <consensus_dir> <genes_csv>'
    parser = OptionParser(usage)
    parser.add_option('-o', dest='out_dir',
                      default='test_out',
                      help='Output directory for predictions [Default: %default]')
    parser.add_option('--rc', dest='rc',
                      default=False, action='store_true',
                      help='Average the fwd and rc predictions [Default: %default]')
    parser.add_option('--shifts', dest='shifts',
                      default='0',
                      help='Ensemble prediction shifts [Default: %default]')
    parser.add_option('-n', dest='n_center',
                      default=10, type='int',
                      help='Number of center bins to average predictions around TSS')
    parser.add_option('--n_uniform', dest='n_uniform', action='store_true', default=False, help='Replace Ns with uniform random [Default: %default]')
    parser.add_option('--all_bins', dest='all_bins', default=False, action='store_true', help='Save all bins to h5 file')
    parser.add_option('--num_chunks', dest='num_chunks', default=None, type=int, help="Number of chunks to split computation into")
    parser.add_option('--chunk_i', dest='chunk_i', default=None, type=int, help="chunk index (0-indexed)")
    (options, args) = parser.parse_args()

    num_expected_args = 4
    if len(args) != num_expected_args:
        parser.error(
            'Incorrect number of arguments, expected {} arguments but got {}'.format(num_expected_args, len(args)))
    params_file = args[0]
    model_file = args[1]
    consensus_dir = args[2]
    genes_file = args[3]

    os.makedirs(options.out_dir, exist_ok=True)

    # parse shifts to integers
    options.shifts = [int(shift) for shift in options.shifts.split(',')]

    assert options.n_center % 2 == 0, "Number of center bins to average preds around TSS should be even"

    #######################################################
    # read model parameters
    with open(params_file) as params_open:
        params = json.load(params_open)
    params_model = params['model']
    params_train = params['train']

    # initialize model
    seqnn_model = seqnn.SeqNN(params_model)
    seqnn_model.restore(model_file)
    seqnn_model.build_ensemble(options.rc, options.shifts)

    #######################################################
    # evaluate
    genes = natsorted([os.path.basename(file) for file in glob.glob(f'{consensus_dir}/*')])

    # read genes_file
    genes_df = pd.read_csv(genes_file, names=['ens_id', 'chrom', 'bp', 'gene_symbol', 'strand'], index_col=False)
    genes_df['gene_symbol'] = genes_df['gene_symbol'].fillna(genes_df['ens_id'])
    genes_df = genes_df.set_index('gene_symbol')
    genes_df.index = genes_df.index.str.lower()

    assert np.all(np.in1d(np.array(genes), genes_df.index)), "Genes in consensus dir not in genes file"

    # Split into chunks if options are set
    if options.num_chunks is not None:
        gene_splits = np.array_split(genes, options.num_chunks)
        genes = gene_splits[options.chunk_i]
        assert len(genes) > 0, "Gene split resulted in empty list"

    # Predict for all samples
    print("Predicting all samples for all genes...")
    for gene in tqdm(genes):
        fasta_files = glob.glob(f'{consensus_dir}/{gene}/samples/*.fa')
        preds_stream = stream.PredStreamGen(seqnn_model,
                                            gen_sample_seqs_for_gene(fasta_files, gene, params_model['seq_length'],
                                                                     strand=genes_df.loc[gene, 'strand'], n_uniform=options.n_uniform),
                                            params_train['batch_size'])

        if os.path.exists(f'{options.out_dir}/{gene}/{gene}.h5'):
            print(f"Skipping {gene} since it already exists")
            continue

        preds_dir = f'{options.out_dir}/{gene}'
        sample_preds_dir = f'{preds_dir}/all_bins_per_sample'
        os.makedirs(sample_preds_dir, exist_ok=True)

        fasta_record_ids = []
        center_sample_preds = []
        for si, fasta_file in enumerate(fasta_files):
            # Get predictions on consensus seqs
            record = list(SeqIO.parse(fasta_file, 'fasta'))[0]
            fasta_record_ids.append(f"{record.id}|{Path(fasta_file).stem}")

            preds = preds_stream[si]
            center_sample_preds.append(avg_center_bins(preds, n_center=options.n_center))

            if options.all_bins:
                # Save to h5
                with h5py.File(f'{sample_preds_dir}/{Path(fasta_file).stem}.h5', 'w') as preds_h5:
                    preds_h5.create_dataset('all_preds', data=preds)

        # Save to h5
        with h5py.File(f'{preds_dir}/{gene}.h5', 'w') as preds_h5:
            preds_h5.create_dataset('preds', data=center_sample_preds)
            preds_h5.create_dataset('record_ids', data=np.array(fasta_record_ids, 'S'))


def gen_sample_seqs_for_gene(fasta_files: List[str], gene: str, seq_length: int, strand: str, n_uniform: bool = False) -> Generator:
    """
    Create generator for 1-hot encoded sequences for input into Basenji2 for all samples for a given gene.
    Assumes TSS is at center of fasta sequence, with less sequence at the end if seq is even length -> at index len(seq) // 2

    fasta_files: list of paths to fasta files
    gene: Ensembl ID of gene
    seq_length: length of sequence (model input)
    n_uniform: whether to replace Ns with 0.25 for uniform background
    """
    for fasta_file in fasta_files:
        for record in SeqIO.parse(fasta_file, 'fasta'):
            seq = str(record.seq).upper()

            # first, deal with truncated sequences due to being at beginning or end of chromosome
            interval = record.id.split(":")[1]
            if interval.startswith("-"):
                # if sequence has negative sign in interval, it means the sequence is definitely missing the beginning

                # sanity check
                bp_start = -int(interval.split("-")[-2])  # we need to parse like this because - sign is in front of first number
                bp_end = int(interval.split("-")[-1])
                assert bp_end - bp_start + 1 == ENFORMER_SEQ_LENGTH

                # pad with Ns to beginning of sequence
                seq = "N" * (ENFORMER_SEQ_LENGTH - len(seq)) + seq

            else:
                # sanity check
                bp_start, bp_end = map(int, interval.split("-"))
                assert bp_end - bp_start + 1 == ENFORMER_SEQ_LENGTH

                # check if sequence is missing end of sequence
                if len(seq) < ENFORMER_SEQ_LENGTH:
                    # pad with Ns to end of sequence
                    seq = seq + "N" * (ENFORMER_SEQ_LENGTH - len(seq))

            assert len(seq) == ENFORMER_SEQ_LENGTH, f"Sequence length is {len(seq)} for {record.id}"  # one last check

            # trim sequence to seq_length while keeping TSS at center
            # for Basenji, the TSS will be centered with less sequence upstream than downstream
            assert len(seq) % 2 == 0, f"Expected even length sequence, got {len(seq)} for record {record.id} in gene {gene}"
            assert seq_length % 2 == 0, f"Expected even length sequence input for model"
            if strand == '+':
                start = len(seq) // 2 - seq_length // 2 + 1
            elif strand == '-':
                start = len(seq) // 2 - seq_length // 2
            else:
                assert False, f"Invalid strand {strand} for gene {gene}"
            end = start + seq_length
            seq = seq[start:end]
            assert len(seq) == seq_length, f"Expected {seq_length} length sequence, got {len(seq)} for record {id} " \
                                           f"in gene {gene}"
            yield dna_1hot(seq, n_uniform=True)


if __name__ == '__main__':
    main()
