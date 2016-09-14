#!/bin/bash

# As per the README, the Treebank should already be downloaded and unarchived.
# The Treebank directory is expected to be `treebank_3`, with the path to
# PTB parses at ../../datasets/treebank_3/parsed/mrg/wsj

# This script converts the constituency trees to dependency trees.

# Makes a copy of the original WSJ directory. This is used for constructing
# base NPs, whereas the patched version is used for constructing the dependency
# trees (as recommended by the conversion script).

mkdir "../../datasets/treebank_3_original"
cp -r ../../datasets/treebank_3/parsed/mrg/wsj ../../datasets/treebank_3_original/

# Patch PTB with David Vadas' NP bracketing script
cd ../../datasets/treebank_3

patch -p1 < ../../external_tools/PTB_NP_Bracketing_Data_1.0/ptb_wsj_np_bracketing_00_24.diff

cd ../../data/preprocessing

PENN_CONVERTER_DIR="../../external_tools/Penn2Dependency"
DATA_DIR="../../datasets"
mkdir ${DATA_DIR}/dependency
mkdir ${DATA_DIR}/dependency/logs

python create_ptb_dependency_trees.py \
    --ptb_patched_dir ../../datasets/treebank_3/parsed/mrg/wsj/ \
    --penn_converter_dir ${PENN_CONVERTER_DIR} \
    --data_output_dir ${DATA_DIR}/dependency \
    --data_output_logs_dir ${DATA_DIR}/dependency/logs
