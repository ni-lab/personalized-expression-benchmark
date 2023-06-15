# -*- coding: utf-8 -*-
import argparse
import os

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from Beluga import Beluga
from Bio import SeqIO
from expecto_utils import *
from tqdm import tqdm

ENFORMER_SEQ_LENGTH = 393216


def main():
    """
    Use ExPecto to predict expression for all genes using the reference genome.

    Arguments:
    - expecto_model: XGBoost model file
    - consensus_dir: directory containing consensus and reference sequences for each gene
    - genes_file: file containing Ensembl gene IDs, chromosome, TSS position, gene symbol, and strand
    """
    parser = argparse.ArgumentParser(description="Predict expression for consensus sequences using ExPecto")
    parser.add_argument("expecto_model")
    parser.add_argument("consensus_dir")
    parser.add_argument("genes_file")
    parser.add_argument("--beluga_model",
                        type=str,
                        default="./expecto/resources/deepsea.beluga.pth",
                        help="Path to trained Beluga model.")
    parser.add_argument("--batch_size",
                        type=int,
                        default=1024,
                        help="Batch size for neural network predictions.")
    parser.add_argument("-o", dest="out_dir",
                        type=str,
                        required=True,
                        help="Output directory")
    args = parser.parse_args()

    # Setup
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Beluga model and ExPecto XGBoost model
    model = Beluga()
    model.load_state_dict(torch.load(args.beluga_model))
    model.eval().to(device)

    bst = xgb.Booster()
    bst.load_model(args.expecto_model.strip())

    # Load genes file
    genes_df = read_genes_file(args.genes_file)

    # Predict on reference
    shifts = np.array(list(range(-20000, 20000, 200)))
    expecto_ref_preds = []
    for i, gene in enumerate(tqdm(genes_df.index)):
        strand = genes_df.loc[gene, "strand"]

        # Load reference sequence
        ref_fasta = f"{args.consensus_dir}/{gene}/ref.fa"
        ref_seq = get_seq_from_fasta(ref_fasta)

        # Predict chromatin marks
        seq_shifts = encodeSeqs(get_seq_shifts_for_sample_seq(ref_seq, strand, shifts)).astype(np.float32)
        ref_preds = np.zeros((seq_shifts.shape[0], 2002))
        for i in range(0, seq_shifts.shape[0], args.batch_size):
            batch = torch.from_numpy(seq_shifts[i * args.batch_size:(i + 1) * args.batch_size]).to(device)
            batch = batch.unsqueeze(2)
            ref_preds[i * args.batch_size:(i + 1) * args.batch_size] = model.forward(batch).cpu().detach().numpy()

        ref_preds = (ref_preds[:ref_preds.shape[0] // 2] + ref_preds[ref_preds.shape[0] // 2:]) / 2  # avg the reverse complement
        beluga_ref_preds = np.array(ref_preds)[None]

        # Get ExPecto predictions
        pos_weight_shifts = shifts
        pos_weights = np.vstack([
            np.exp(-0.01 * np.abs(pos_weight_shifts) / 200) * (pos_weight_shifts <= 0),
            np.exp(-0.02 * np.abs(pos_weight_shifts) / 200) * (pos_weight_shifts <= 0),
            np.exp(-0.05 * np.abs(pos_weight_shifts) / 200) * (pos_weight_shifts <= 0),
            np.exp(-0.1 * np.abs(pos_weight_shifts) / 200) * (pos_weight_shifts <= 0),
            np.exp(-0.2 * np.abs(pos_weight_shifts) / 200) * (pos_weight_shifts <= 0),
            np.exp(-0.01 * np.abs(pos_weight_shifts) / 200) * (pos_weight_shifts >= 0),
            np.exp(-0.02 * np.abs(pos_weight_shifts) / 200) * (pos_weight_shifts >= 0),
            np.exp(-0.05 * np.abs(pos_weight_shifts) / 200) * (pos_weight_shifts >= 0),
            np.exp(-0.1 * np.abs(pos_weight_shifts) / 200) * (pos_weight_shifts >= 0),
            np.exp(-0.2 * np.abs(pos_weight_shifts) / 200) * (pos_weight_shifts >= 0)])

        # "backwards compatibility" for ExPecto model
        features = np.sum(pos_weights[None, :, :, None] * beluga_ref_preds[:, None, :, :], axis=2).reshape(-1, 10 * 2002)
        features = np.concatenate([np.zeros((1, 10, 1)), features.reshape((-1, 10, 2002))], axis=2).reshape((-1, 20030))  # add 0 shift
        expecto_ref_features = xgb.DMatrix(features)

        expecto_ref_preds.append(bst.predict(expecto_ref_features))

    expecto_ref_preds = np.array(expecto_ref_preds).squeeze()

    # Save predictions
    df = pd.DataFrame({"genes": np.array(genes_df.index.values), "ref_preds": expecto_ref_preds})
    df.to_csv(f"{args.out_dir}/ref_preds.csv", header=True, index=False)


def get_seq_from_fasta(fasta_file: str) -> str:
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
    return seq


def get_seq_shifts_for_sample_seq(sample_seq, strand, shifts, windowsize=2000):
    """
    Get shifts for sequence, centered at TSS.
    windowsize denotes input size for neural network, which is 2000 for default Beluga model.
    """
    # assumes TSS is at center of sequence, at index len(sample_seq) // 2, which is true of our enformer consensus sequences
    tss_i = len(sample_seq) // 2
    if strand == "+":
        strand = 1
    elif strand == "-":
        strand = -1
    else:
        assert False, f"strand {strand} not recognized"

    seq_shifts = []
    for shift in shifts:
        seq = list(sample_seq[tss_i + (shift * strand) - int(windowsize / 2 - 1):
                         tss_i + (shift * strand) + int(windowsize / 2) + 1])

        assert len(seq) == windowsize, f"Expected seq of length f{windowsize} but got {len(seq)}"
        seq_shifts.append(seq)

    return np.vstack(seq_shifts)


if __name__ == "__main__":
    main()
