# -*- coding: utf-8 -*-
import argparse
import glob
import os
from pathlib import Path
from typing import Generator, List, Tuple

import h5py
import numpy as np
import torch
import xgboost as xgb
from Beluga import Beluga
from Bio import SeqIO
from expecto_utils import *
from natsort import natsorted
from tqdm import tqdm

ENFORMER_SEQ_LENGTH = 393216


def main():
    """
    Predict expression for all genes and each individual gene using their personalized input sequences.

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
    parser.add_argument("--overwrite",
                        default=False,
                        action="store_true",
                        dest="overwrite",
                        help="If true, overwrite existing predictions. Otherwise, skip if h5 file is present.")
    parser.add_argument("--exp_only",
                        default=False,
                        action="store_true",
                        dest="exp_only",
                        help="If true, load in chromatin preds and make predictions.")
    parser.add_argument("--num_chunks", dest="num_chunks",
                        default=None,
                        type=int,
                        help="Total number of chunks to split predictions")
    parser.add_argument("--chunk_i", dest="chunk_i",
                        default=None,
                        type=int,
                        help="Chunk index for current run, starting from 0")
    parser.add_argument("-o", dest="out_dir",
                        type=str,
                        required=True,
                        help="Output directory")
    args = parser.parse_args()

    consensus_dir = args.consensus_dir
    genes_file = args.genes_file

    # Setup
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Beluga model and ExPecto XGBoost model
    model = Beluga()
    model.load_state_dict(torch.load(args.beluga_model))
    model.eval().to(device)

    bst = xgb.Booster()
    bst.load_model(args.expecto_model.strip())

    # Evaluate on all genes
    shifts = np.array(list(range(-20000, 20000, 200)))
    genes = natsorted([os.path.basename(file) for file in glob.glob(f"{consensus_dir}/*")])

    # Load genes file
    genes_df = read_genes_file(genes_file)

    # Parallelization: split into chunks if options are set
    if args.num_chunks is not None:
        gene_splits = np.array_split(genes, args.num_chunks)
        genes = gene_splits[args.chunk_i]
        assert len(genes) > 0, "Gene split resulted in empty list"

    # Make predictions
    print("Predicting chromatin for all samples for all genes...")
    for gene in tqdm(genes):
        fasta_files = glob.glob(f"{consensus_dir}/{gene}/samples/*.fa")
        strand = genes_df.loc[gene, "strand"]

        preds_dir = f"{args.out_dir}/{gene}"
        os.makedirs(preds_dir, exist_ok=True)

        if not args.overwrite and os.path.exists(f"{preds_dir}/{gene}.h5"):
            # skip if output h5 file already exists
            print(f"Skipping gene {gene} since h5 is already present.")
            continue

        if args.exp_only:
            # if we've already made chromatin predictions, just load them
            with h5py.File(f"{preds_dir}/{gene}_chromatin.h5", "r") as f:
                preds = np.array(f["chromatin_preds"])
                fasta_record_ids = [x.decode("utf-8") for x in f["record_ids"]]
        else:
            # otherwise, make chromatin predictions
            fasta_record_ids = []
            sample_seqs_gen = gen_sample_seqs_and_id_for_gene(fasta_files)
            preds = []
            for sample_seq, record_id in sample_seqs_gen:
                seq_shifts = encodeSeqs(get_seq_shifts_for_sample_seq(sample_seq, strand, shifts)).astype(np.float32)

                sample_preds = np.zeros((seq_shifts.shape[0], 2002))
                for i in range(0, seq_shifts.shape[0], args.batch_size):
                    batch = torch.from_numpy(seq_shifts[i * args.batch_size:(i+1) * args.batch_size]).to(device)
                    batch = batch.unsqueeze(2)
                    sample_preds[i * args.batch_size:(i+1) * args.batch_size] = model.forward(batch).cpu().detach().numpy()

                # avg the reverse complement
                sample_preds = (sample_preds[:sample_preds.shape[0] // 2] + sample_preds[sample_preds.shape[0] // 2:]) / 2
                fasta_record_ids.append(record_id)
                preds.append(sample_preds)

            preds = np.stack(preds, axis=0)

        # make expression predictions
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

        # "backwards compatibility"
        features = np.sum(pos_weights[None, :, :, None] * preds[:, None, :, :], axis=2)
        features = np.concatenate([np.zeros((features.shape[0], 10, 1)), features], axis=2).reshape((-1, 20030))  # add 0 shift
        expecto_features = xgb.DMatrix(features)

        expecto_preds = bst.predict(expecto_features)

        # Save predictions for chromatin features
        with h5py.File(f"{preds_dir}/{gene}_chromatin.h5", "w") as preds_h5:
            preds_h5.create_dataset("chromatin_preds", data=preds)  # n_samples x n_bins x n_features
            preds_h5.create_dataset("record_ids", data=np.array(fasta_record_ids, "S"))

        # Save expression predictions from ExPecto
        with h5py.File(f"{preds_dir}/{gene}.h5", "w") as preds_h5:
            preds_h5.create_dataset("expecto_preds", data=expecto_preds)
            preds_h5.create_dataset("record_ids", data=np.array(fasta_record_ids, "S"))


def gen_sample_seqs_and_id_for_gene(fasta_files: List[str]) -> Generator[Tuple[str, str], None, None]:
    """
    Create generator for 1-hot encoded sequences for input into Basenji2 for all samples for a given gene.
    fasta_gz: consensus seqs in the form of gzipped fasta e.g. {gene}/{gene}.fa.gz
    """
    for fasta_file in fasta_files:
        for record in SeqIO.parse(fasta_file, "fasta"):
            seq = str(record.seq).upper()

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
            yield seq, f"{record.id}|{Path(fasta_file).stem}"


def get_seq_shifts_for_sample_seq(sample_seq, strand, shifts, windowsize=2000):
    """
    Get shifts for sequence, centered at TSS.
    windowsize denotes input size for neural network, which is 2000 for default Beluga model.
    """
    # For enformer consensus seqs, the TSS is always at len(sample_seq) // 2
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
