#!/usr/bin/env python

from __future__ import print_function
from optparse import OptionParser
import os
import pandas as pd
import tensorflow as tf
import glob
import subprocess
from tqdm import tqdm

if tf.__version__[0] == '1':
    tf.compat.v1.enable_eager_execution()

"""
compute_dosages.py

Compute dosages of each individual for each SNP in input vcf.
"""


################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <vcf_file>'
    parser = OptionParser(usage)
    parser.add_option('-o', dest='out_dir',
                      default='test_out',
                      help='Output directory for predictions [Default: %default]')
    (options, args) = parser.parse_args()

    num_expected_args = 1
    if len(args) != num_expected_args:
        parser.error(
            'Incorrect number of arguments, expected {} arguments but got {}'.format(num_expected_args, len(args)))
    vcf_file = args[0]

    os.makedirs(options.out_dir, exist_ok=True)

    #######################################################
    # Convenience skip ## lines
    skiprows = 0
    with open(vcf_file, 'r') as vcf:
        for line in vcf:
            if line.startswith("#CHROM"):
                break
            else:
                skiprows += 1

    vcf_df = pd.read_csv(vcf_file, sep='\t', header=0, skiprows=skiprows)
    dosages_df = pd.concat([vcf_df[['#CHROM', 'POS', 'ID', 'REF', 'ALT']],
                        vcf_df.iloc[:, 9:].applymap(get_dosages)], axis=1)

    dosages_df.to_csv(f'{options.out_dir}/dosages.tsv', sep='\t', index=False)


def get_dosages(x):
    return sum(map(int, x[:3].split('|')))

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
