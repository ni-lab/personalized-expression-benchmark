# personalized-expression-benchmark
Official repository for the manuscript
"Personal transcriptome variation is poorly explained by current genomic deep learning models"

TODO: insert link later

## Installation
We evaluated gene expression prediction performance using 4 models: Enformer, Basenji2, ExPecto, and Xpresso. We installed each model in a separate python environment. To find instructions for installing dependencies for each model, see the corresponding README.md file in each model's directory.

## Data
To make predictions on the reference genome and personalized consensus sequences, you will need to generate the reference and consensus sequences for each gene and place these sequences in the `consensus/seqs` directory. 

First download the dependencies (you may want to be in a virtual environment with python version 3.7):
```
pip install pyfaidx
```

To download reference fasta data, run the following commands:
```
mkdir data/ref_fasta && cd data/ref_fasta
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz
gunzip hg19.fa.gz 
faidx -x hg19.fa 
rm *random* *hap* *Un* # cleanup unnecessary files
for file in *.fa; do faidx -x $file; done
rm *random* *hap* *Un* # cleanup unnecessary files
```

To download individual variant data:
1. Navigate to https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-GEUV-1
2. On the right side of the page there should be a blue box titled `Data Files`. Enter `.vcf` in the search bar to find only the vcf files. There should be 23 files total.
3. Select all files and click `Download 23 files`. Then click `Download` under `Download as a ZIP file`.

The following script will generate consensus sequences for each individual and gene:
```
python3 consensus/make_consensus_enformer.py data/ref_fasta data/variants data/gene_list.csv consensus/samples.txt -o consensus/seq
```

## Evaluating gene expression prediction performance
Within each directory, we provide instructions for evaluating gene expression prediction performance using the corresponding model. See the README.md file in each directory for more details.
