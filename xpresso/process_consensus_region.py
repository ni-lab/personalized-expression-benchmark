from __future__ import print_function
from os import name
from constants import USERNAME, NILAH_HOME, CONSENSUS_TEMPDIR, CONSENSUS_OUTDIR

import argparse
import subprocess
import re
import sys
import multiprocessing as mp

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def build_consensus(sample):
    ## Create consensus sequence for a specific individual where reference is the chromosome fasta
    if re.match(r"(HG|NA)\d{5}", sample):
        eprint(f"#### Starting consenus file for {sample} ####")

        for i in [1, 2]:
            VAR_CMD = f"samtools faidx {ref_chr_path} chr{chr_num}:{start}-{end} | bcftools consensus -s {sample} -H {i}pIu -o {tempdir}/{sample}.{i}pIu.fa {vcf}"
            eprint(VAR_CMD)
            subprocess.run(VAR_CMD, shell=True)

            RENAME_FASTA_HEADER_CMD = f"sed -i 's/>.*/&|{sample}|{strand}|{i}pIu/' {tempdir}/{sample}.{i}pIu.fa"
            eprint(RENAME_FASTA_HEADER_CMD)
            subprocess.run(RENAME_FASTA_HEADER_CMD, shell=True)

            if strand == '-':
                REV_CMD = f"seqtk seq -r {tempdir}/{sample}.{i}pIu.fa > {rev_tempdir}/{sample}.{i}pIu.fa"
                eprint(REV_CMD)
                subprocess.run(REV_CMD, shell=True)

                MV_CMD = f"cat {rev_tempdir}/{sample}.{i}pIu.fa > {tempdir}/{sample}.{i}pIu.fa"
                eprint(MV_CMD)
                subprocess.run(MV_CMD, shell=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str)
    parser.add_argument("-l", "--start", type=int)
    parser.add_argument("-u", "--end", type=int)
    parser.add_argument("-c", "--chr", type=int)
    parser.add_argument("-v", "--vcf", type=str)
    parser.add_argument("-s", "--strand", type=str)
    parser.add_argument("--cache", action="store_true", default=False)
    args = parser.parse_args()

    chr_num = args.chr
    vcf = args.vcf
    strand = args.strand
    start = args.start
    end = args.end
    gene_name = args.name
    cache = args.cache

    tempdir = f"{CONSENSUS_TEMPDIR}/{gene_name}"
    rev_tempdir = f"{CONSENSUS_TEMPDIR}/int_{gene_name}"
    outdir = f"{CONSENSUS_OUTDIR}/{gene_name}"
    final_vcf_name = f"{NILAH_HOME}/parth/snps_geuv_vcf/GEUVADIS.chr{chr_num}.cleaned.vcf.gz"
    ref_chr_path = f"{NILAH_HOME}/fasta/hg19_by_chr/chr{chr_num}.fa"
    scratch = f"/global/scratch/users/{USERNAME}"

    ref_region_path = f"{CONSENSUS_OUTDIR}/{gene_name}/ref_roi.fa" 
    if strand == "+":
        REGION_CMD = f"samtools faidx {ref_chr_path} chr{chr_num}:{start}-{end} > {ref_region_path}"
    elif strand == "-":
        REGION_CMD = f"samtools faidx -i {ref_chr_path} chr{chr_num}:{start}-{end} > {ref_region_path}"
    subprocess.run(REGION_CMD, shell=True)

    RENAME_FASTA_HEADER_CMD = f"sed -i 's/>.*/&|ref|{gene_name}|{strand}/' {ref_region_path}"
    subprocess.run(RENAME_FASTA_HEADER_CMD, shell=True)

    eprint(f"-----------------------Starting {gene_name}------------------")
    if not cache:
        ## Get all the sample names
        process = subprocess.run(f'bcftools query -l {vcf}', shell=True, check=True, stdout=subprocess.PIPE, universal_newlines=True)
        output = process.stdout
        output_split = output.split('\n')
        eprint(output_split)
        
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(build_consensus, output_split)