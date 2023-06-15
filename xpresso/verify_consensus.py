from Bio import SeqIO
from pyfaidx import Fasta
from constants import CONSENSUS_OUTDIR

import gzip
import argparse

def count_variants(ref_seq, seq):
    if len(ref_seq) != len(seq):
        return -1
    
    num_variants = 0
    for i in range(len(seq)):
        num_variants += int(seq[i] != ref_seq[i])

    return num_variants

def parse_fasta(ref_record, record):
    if (not bgzipped and record.id == ref_record.id) or (record.name == ref_record.name):
        return ["ref", "", str(len(record.seq)), ""]
    # ref, nothing, length, nothing
    else:
        record_info = record.id.split('|') if not bgzipped else record.name.split('|')
        sample, phase = record_info[1], record_info[-1]
        match_ref_len = str(len(record.seq) == len(ref_record.seq))
        num_variants = count_variants(ref_record.seq, record.seq)

        return [sample, phase, match_ref_len, str(num_variants)]
        # sample, phase, matches ref length, number of variants

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str)
    parser.add_argument("-b", "--bgzip", action="store_true", default=False)
    args = parser.parse_args()

    name = args.name
    bgzipped = args.bgzip

    named_outdir = f"{CONSENSUS_OUTDIR}/{name}"
    ref_record = list(SeqIO.parse(f"{named_outdir}/ref_roi.fa", "fasta"))[0]
    info = '\t'.join(parse_fasta(ref_record, ref_record)) + "\n"

    if not bgzipped:
        fasta_file = f"{named_outdir}/{name}.fa.gz"
        with gzip.open(fasta_file, "rt") as handle:
            with open(f"{named_outdir}/{name}.log.txt", "a") as log_file:
                log_file.write(info)
                records = list(SeqIO.parse(handle, "fasta"))
                for record in records:
                    info = '\t'.join(parse_fasta(ref_record, record)) + "\n"
                    log_file.write(info)

    else:
        fasta_file = f"{named_outdir}/bg_{name}.fa.gz"
        records = Fasta(fasta_file)
        for record in records:
            info = '\t'.join(parse_fasta(ref_record, record)) + "\n"
            with open(f"{named_outdir}/{name}.log.txt", "a") as log_file:
                log_file.write(info)
