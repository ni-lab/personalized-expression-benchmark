from __future__ import print_function
from constants import ALL_MUTS_VCF_DIR, CONSENSUS_OUTDIR, SNPS_ONLY_VCF_DIR

import subprocess
import sys
import argparse
import multiprocessing as mp
import os

## example command for regional consensus:
# python consensus_job_submitter.py
## example command for chromosomal consensus:
# python consensus_job_submitter.py

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def submit_job(input):
    ens_id, chr_num, tss, gene_symbol, strand = input.split(',')
    name = gene_symbol if gene_symbol else ens_id
    gene_name = name.lower()
    chr_num = int(chr_num)
    tss = int(tss)

    if strand == '+':
        start, end = tss - upstream, tss + downstream
    else:
        start, end = tss - downstream, tss + upstream

    vcf = f"{vcf_dir}/GEUVADIS.chr{chr_num}.cleaned.vcf.gz"
    consensus_fasta = f"{CONSENSUS_OUTDIR}/{gene_name}/{gene_name}.fa.gz"

    eprint(f"---------------------------------------- Starting {gene_name} ----------------------------------------")

    if os.path.exists(consensus_fasta) and cache:
        eprint(f"Skipping {gene_name}")
        return

    cmd = CMD.format(gene_name, start, end, chr_num, strand, vcf, region)
    eprint(cmd)
    subprocess.call(cmd, shell=True)

if __name__ == "__main__":
    CMD = "sbatch -J {0}_consensus --export=ALL,name={0},start={1},end={2},chrom={3},strand={4},vcf={5},region={6} process_consensus.sh"

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str)
    parser.add_argument("-u", "--upper", type=int)
    parser.add_argument("-l", "--lower", type=int)
    parser.add_argument("-U", "--upstream", type=int)
    parser.add_argument("-D", "--downstream", type=int)
    parser.add_argument("-r", "--region", action="store_true", default=False)
    parser.add_argument("-m", "--all_muts", action="store_true", default=False)
    parser.add_argument("--cache", action="store_true", default=False)
    args = parser.parse_args()

    input_file = args.input
    region = int(args.region)
    vcf_dir = ALL_MUTS_VCF_DIR if args.all_muts else SNPS_ONLY_VCF_DIR
    upstream = args.upstream - 1
    downstream = args.downstream
    upper = args.upper
    lower = args.lower
    cache = args.cache

    if region:
        with open(input_file) as f:
            inp = [line.strip() for line in f.readlines()]
    else:
        inp = list(range(lower, upper+1))
        raise NotImplementedError("Have not configured consensus files to process entire chromosomes at once")

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(submit_job, inp)
    
