from __future__ import print_function
from constants import CONSENSUS_OUTDIR, GM_MODEL, GM_MODEL_ABBREV, HM_MODEL, HM_MODEL_ABBREV, PREDICT_OUTDIR, XPRESSO_GEUVADIS

import subprocess
import sys
import argparse
import multiprocessing as mp
import os

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def predict(input):
    ens_id, _, _, gene_symbol, _ = input.split(',')
    name = gene_symbol if gene_symbol else ens_id
    gene_name = name.lower()

    consensus_fasta = f"{CONSENSUS_OUTDIR}/{gene_name}/{gene_name}.fa.gz"

    eprint(f"---------------------------------------- Starting {gene_name} ----------------------------------------")

    if os.path.exists(f"{PREDICT_OUTDIR}/{gene_name}.GM12878.preds.txt") and os.path.exists(f"{PREDICT_OUTDIR}/{gene_name}.humanMedian.preds.txt"):
        eprint(f"Skipping {gene_name}")
        return


    for model in [GM_MODEL, HM_MODEL]:
        abbrev = GM_MODEL_ABBREV if model == GM_MODEL else HM_MODEL_ABBREV
        cmd = PREDICT_CMD.format(XPRESSO_GEUVADIS, model, consensus_fasta, PREDICT_OUTDIR, gene_name, abbrev)
        eprint(cmd)
        subprocess.call(cmd, shell=True)

if __name__ == "__main__":
    PREDICT_CMD = "python {0}/xpresso_predict.py {0}/pretrained_models/{1} {2} {3}/{4}.{5}.preds.txt"

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str)
    args = parser.parse_args()

    input_file = args.input

    CREATE_DIR_CMD = f"mkdir -p {PREDICT_OUTDIR}"
    subprocess.call(CREATE_DIR_CMD, shell=True)

    with open(input_file) as f:
        input = [line.strip() for line in f.readlines()]

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(predict, input)
    
