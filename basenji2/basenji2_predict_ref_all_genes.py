#!/usr/bin/env python

from __future__ import print_function

import glob
import json
import os
from optparse import OptionParser
from typing import Generator, List

import numpy as np
import pandas as pd
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
    parser.add_option('--ti', dest='ti',
                      default=5110, type='int',
                      help='Prediction track index to save preds for (5110 corresponds to lymphoblastoid cell line)')
    parser.add_option('--n_uniform', dest='n_uniform',
                      action='store_true', default=False,
                      help='Replace Ns with uniform random [Default: %default]')
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

    # make predictions
    print("Predicting on reference sequence for all genes...")
    preds_stream = stream.PredStreamGen(seqnn_model,
                                        gen_ref_seqs_for_all_genes(consensus_dir, genes, params_model['seq_length'], genes_df=genes_df, n_uniform=options.n_uniform),
                                        params_train['batch_size'])

    ref_gene_preds = []
    for gi, gene in enumerate(tqdm(genes)):
        preds = preds_stream[gi]
        pred = avg_center_bins(preds, n_center=options.n_center)[options.ti]
        ref_gene_preds.append(pred)
    ref_gene_preds = np.array(ref_gene_preds)

    # save to csv
    df = pd.DataFrame({"genes": genes, "ref_preds": ref_gene_preds})
    df.to_csv(f'{options.out_dir}/ref_preds.csv', header=True, index=False)


def get_1_seq_from_fasta(fasta_file: str) -> str:
    """
    Get sequence from fasta file. Fasta file should only have 1 record. Automatically upper cases all nts.
    Also deals with possible seq truncations from being at beginning or end of chromosome by padding appropriately with Ns.
    """
    records = list(SeqIO.parse(fasta_file, "fasta"))
    assert len(records) == 1, f"Expected 1 record in fasta file {fasta_file}, but got {len(records)} records"

    record = records[0]
    seq = str(record.seq).upper()

    # deal with possible seq truncations from being at beginning or end of chromosome
    interval = record.id.split(":")[1]
    if interval.startswith("-"):
        # if sequence has negative sign in interval, it means the sequence is definitely missing the beginning

        # sanity check
        bp_start = -int(interval.split("-")[-2])  # we need to parse like this because negative sign is in front of first number
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

    assert len(seq) == ENFORMER_SEQ_LENGTH, f"Sequence length is {len(seq)} for {record.id}, expected {ENFORMER_SEQ_LENGTH}"  # one last check

    return seq


def gen_ref_seqs_for_all_genes(consensus_dir: str, genes: List[str], seq_length: int, genes_df: pd.DataFrame, n_uniform: bool = False) -> Generator:
    """
    Create generator for 1-hot encoded sequences for the reference sequence surrounding each gene.
    - consensus_dir
    - genes: list of strings; determines order of genes to return
    - seq_length: length of sequence input
    """
    for gene in genes:
        ref_fasta = f'{consensus_dir}/{gene}/ref.fa'
        strand = genes_df.loc[gene, 'strand']
        seq = get_1_seq_from_fasta(ref_fasta)

        # trim sequence to seq_length while keeping TSS at center
        assert len(seq) % 2 == 0, f"Expected even length sequence, got {len(seq)} for gene {gene}"
        assert seq_length % 2 == 0, f"Expected even length sequence input for model"
        if strand == '+':
            start = len(seq) // 2 - seq_length // 2 + 1
        elif strand == '-':
            start = len(seq) // 2 - seq_length // 2
        else:
            assert False, f"Invalid strand {strand} for gene {gene}"
        end = start + seq_length
        seq = seq[start:end]
        assert len(seq) == seq_length, f"Expected sequence length to be {seq_length}, but got {len(seq)}"
        yield dna_1hot(seq, n_uniform=n_uniform)


if __name__ == '__main__':
    main()
