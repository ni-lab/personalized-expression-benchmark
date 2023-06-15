from __future__ import print_function
from constants import GEUVADIS, HOMEDIR, VCF_TEMPDIR

import argparse
import subprocess
import os
import sys
import multiprocessing as mp

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def collect_result(result):
    results.append(results)

def process_vcf(vcf, final_vcf, chr_num):
    if not os.path.exists(f"{output_path}/{final_vcf}"):
        ## Copy the file to scratch
        CP_CMD = f"cp -c {GEUVADIS}/{vcf} {VCF_TEMPDIR}/{vcf}"
        print("CP command: ", CP_CMD)
        subprocess.run(CP_CMD, shell=True)

        ## Decompress the file
        DGZ_CMD = f"gzip -d {VCF_TEMPDIR}/{vcf}"
        print("DGZ command: ", DGZ_CMD)
        subprocess.run(DGZ_CMD, shell=True)

        ## Rename the column
        RENAME_CMD = """awk '{if($0 !~ /^#/) print "chr"$0; else print $0}' """ + f"{VCF_TEMPDIR}/{vcf[:-3]} > {VCF_TEMPDIR}/renamed.{vcf[:-3]}"
        print("RENAME command: ", RENAME_CMD)
        subprocess.call(RENAME_CMD, shell=True)

        ## Compress the file with bgzip
        GZ_CMD = f"bgzip -c {VCF_TEMPDIR}/renamed.{vcf[:-3]} > {VCF_TEMPDIR}/{vcf}"
        print("GZ command: ", GZ_CMD)
        subprocess.run(GZ_CMD, shell=True)

        APPEND_CONTIG_CMD = f"echo '##contig=<ID=chr{chr_num}>' >> {hdr_file}"
        print("APPEND_CONTIG command: ", APPEND_CONTIG_CMD)
        subprocess.run(APPEND_CONTIG_CMD, shell=True)        

        ## Reheader the column
        REHEADER_CMD = f"bcftools annotate -h {hdr_file} -Oz -o {VCF_TEMPDIR}/hdr_{vcf} {VCF_TEMPDIR}/{vcf}"
        print("REHEADER command: ", REHEADER_CMD)
        subprocess.run(REHEADER_CMD, shell=True)

        ## Filter to keep only samples
        SAMPLES_CMD = f"bcftools view --samples-file {sample_file} -Oz -o {VCF_TEMPDIR}/sample_{vcf}  {VCF_TEMPDIR}/hdr_{vcf}"
        print("SAMPLES command: ", SAMPLES_CMD)
        subprocess.run(SAMPLES_CMD, shell=True)

        if all_muts:
            ALL_MUTS_CMD = f"mv {VCF_TEMPDIR}/sample_{vcf} {output_path}/{final_vcf}"
            print("ALL_MUTS command: ", ALL_MUTS_CMD)
            subprocess.run(ALL_MUTS_CMD, shell=True)
        else:
            ## Filter to keep only SNPs
            SNPS_ONLY_CMD = f"bcftools view --exclude-types indels,mnps,ref,bnd,other -Oz -o {output_path}/{final_vcf} {VCF_TEMPDIR}/sample_{vcf}"
            print("SNPS_ONLY command: ", SNPS_ONLY_CMD)
            subprocess.run(SNPS_ONLY_CMD, shell=True)

    ## Create an index for the vcf file
    if not os.path.exists(f"{output_path}/{final_vcf}.csi"):
        IDX_CMD = f"bcftools index {output_path}/{final_vcf}"
        print("IDX command: ", IDX_CMD)
        subprocess.run(IDX_CMD, shell=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--lower", type=int)
    parser.add_argument("-u", "--upper", type=int)
    parser.add_argument("-m", "--all_muts", action="store_true", default=False)
    args = parser.parse_args()

    upper = args.upper
    lower = args.lower
    all_muts = args.all_muts

    vcf = "GEUVADIS.chr{0}.PH1PH2_465.IMPFRQFILT_BIALLELIC_PH.annotv2.genotypes.vcf.gz"
    final_vcf_name = "GEUVADIS.chr{0}.cleaned.vcf.gz"
    output_path = f"{HOMEDIR}/snps_geuv_vcf" if not all_muts else f"{HOMEDIR}/allmuts_geuv_vcf"
    hdr_file = f"{HOMEDIR}/basefiles/hdr.txt"
    sample_file = f"{HOMEDIR}/basefiles/same_length_inds.txt"


    eprint(f"------------------------------- Starting VCF processing -------------------------------")
    
    pool = mp.Pool(processes=mp.cpu_count())
    results = []
    for i in range(lower, upper + 1):
        pool.apply_async(process_vcf, (vcf.format(i), final_vcf_name.format(i), i), callback=collect_result)
        
    pool.close()
    pool.join()

