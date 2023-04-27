import argparse
import subprocess
import re
import os
import sys
import multiprocessing as mp
import pyfaidx
from itertools import product

REF_DIR = "ref"
INDS = "samples"
GENE_FILE = "data/gene_list.csv"
SAMPLE_FILE = "samples.txt"
SEQUENCE_LENGTH = 393216
INTERVAL = 114688


def get_vcf(chr):
    return f"/clusterfs/nilah/parth/snps_geuv_vcf/GEUVADIS.chr{chr}.cleaned.vcf.gz"


def get_ref_fasta(chr):
    return f"/clusterfs/nilah/fasta/hg19_by_chr/chr{chr}.fa"


def get_items(file):
    with open(file, "r") as f:
        return f.read().splitlines()


def get_sample_files(sample, gene_id):
    return f"{INDS}/{sample}/{gene_id}.1pIu.fa", f"{INDS}/{sample}/{gene_id}.2pIu.fa"

def get_index_files(sample, gene_id):
    return f"{INDS}/{sample}/{gene_id}.1pIu.fai", f"{INDS}/{sample}/{gene_id}.2pIu.fai"

def generate_ref(gene):
    # gene format: 'ENSG00000263280,16,2917619,AC003965.1,-'
    gene_id, chr, tss, _, strand = gene.split(",")
    print(f"#### Starting reference fasta for {gene_id} ####")
    out_file = f"{REF_DIR}/{gene_id}.fa"
    with open(out_file, "w") as f:
        start, end = int(tss) - SEQUENCE_LENGTH // 2, int(tss) + SEQUENCE_LENGTH // 2 - 1
#        start, end = int(tss) - INTERVAL // 2, int(tss) + INTERVAL // 2
        ref_command = f"samtools faidx {get_ref_fasta(chr)} chr{chr}:{start}-{end} -o {out_file}"
        subprocess.run(ref_command, shell=True)


def generate_consensus(pair):
    gene, sample = pair
    gene_id, chr, tss, _, strand = gene.split(",")
    out1, out2 = get_sample_files(sample, gene_id)
    ind1, ind2 = get_index_files(sample, gene_id)

    print(f"#### Starting consensus fasta for {gene_id}, Sample {sample} ####")
    hap1 = f"bcftools consensus -s {sample} -f {REF_DIR}/{gene_id}.fa -I -H 1pIu {get_vcf(chr)} > {out1}"
    hap2 = f"bcftools consensus -s {sample} -f {REF_DIR}/{gene_id}.fa -I -H 2pIu {get_vcf(chr)} > {out2}"
    subprocess.run(hap1, shell=True)
    subprocess.run(hap2, shell=True)
    pyfaidx.Faidx(out1)
    pyfaidx.Faidx(out2)

def make_dirs(samples):
    for sample in samples:
        if not os.path.exists(f"{INDS}/{sample}"):
            os.makedirs(f"{INDS}/{sample}")

if __name__ == "__main__":
    genes = get_items(GENE_FILE)
    samples = get_items(SAMPLE_FILE)
    #make sample directories
    make_dirs(samples)
    pool = mp.Pool(processes=mp.cpu_count())
    with pool:
        pairs = product(genes, samples)
        pool.map(generate_consensus, pairs)



