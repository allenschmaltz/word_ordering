#!/bin/bash

# This script creates the word ordering datasets from the PTB constiuency trees
# and the converted dependency trees.


TOP_DATA_DIR=$(dirname $(pwd))
REPO_DIR=$(dirname $TOP_DATA_DIR)

# absolute path to PTB is needed for NLTK
PTB_DIR=${REPO_DIR}/"datasets/treebank_3_original/wsj"
DATA_DIR="../../datasets"

MALT_PARSER_DIR="../../external_tools/maltparser-1.8.1"

FILTERED_DATA_DIR=${DATA_DIR}"/dependency_filtered_ordering_atomic"
mkdir ${FILTERED_DATA_DIR}

# Construct the BNP delineated token files from the PTB constiuency trees,
# and collapse/filter the dependency tress such that they align with the
#  BNP delineated tokens:
python ptb_to_word_ordering_dataset.py \
    --ptb_dir ${PTB_DIR} \
    --data_dir ${DATA_DIR}

# Projectivize the dependency trees:
for SPLIT_NAME in "train" "valid" "test";
do

    java -jar ${MALT_PARSER_DIR}/maltparser-1.8.1.jar -c pproj -m proj -i ${FILTERED_DATA_DIR}/${SPLIT_NAME}_filtered_dep.txt -o ${FILTERED_DATA_DIR}/${SPLIT_NAME}_filtered_dep_projected.txt -pp head

    java -jar ${MALT_PARSER_DIR}/maltparser-1.8.1.jar -c pproj -m proj -i ${FILTERED_DATA_DIR}/${SPLIT_NAME}_filtered_dep_nonpsyms.txt -o ${FILTERED_DATA_DIR}/${SPLIT_NAME}_filtered_dep_nonpsyms_projected.txt -pp head

done

mkdir ${DATA_DIR}/zgen_data
mkdir ${DATA_DIR}/zgen_data_nonpsyms

# Convert the dependency trees to the ZGen and Yara parser formats:
python dependency_trees_to_zgen_format.py \
    --ptb_dir ${PTB_DIR} \
    --filtered_dep_dir ${FILTERED_DATA_DIR} \
    --zgen_output_dir ${DATA_DIR}/zgen_data \
    --zgen_output_dir_nonpsyms ${DATA_DIR}/zgen_data_nonpsyms


# Convert the ZGen formatted dataset to the format used by the LSTM and NGram decoders:

ZGEN_FORMATTED_PTB_DIR=${DATA_DIR}/zgen_data
TRAIN_FILE=${ZGEN_FORMATTED_PTB_DIR}/train_ref.txt
VALID_FILE=${ZGEN_FORMATTED_PTB_DIR}/valid_ref.txt
TEST_FILE=${ZGEN_FORMATTED_PTB_DIR}/test_ref.txt


OUTPUT_GOLD_DATA_DIR=${DATA_DIR}/zgen_data_gold
mkdir ${OUTPUT_GOLD_DATA_DIR}

PROCESSED_DATA_DIR=${DATA_DIR}/zgen_data_npsyms_freq3_unkUNK

mkdir ${PROCESSED_DATA_DIR}
mkdir ${PROCESSED_DATA_DIR}/temp_files
mkdir ${PROCESSED_DATA_DIR}/no_npsyms
mkdir ${PROCESSED_DATA_DIR}/npsyms


python zgen_format_to_lstm_format.py    -l ${TRAIN_FILE} \
                                        -v ${VALID_FILE} \
                                        -t ${TEST_FILE} \
                                        -o ${PROCESSED_DATA_DIR}/temp_files \
                                        --retain_unk_case \
                                        --lowercase_unk_sym "unk" \
                                        --uppercase_unk_sym "UNK" \
                                        -f 3


# save the data without <unk> and N replacements for BLEU comparisons

python zgen_output_to_tokens.py -i ${TRAIN_FILE} -o ${OUTPUT_GOLD_DATA_DIR}/train_words_ref.txt

python zgen_output_to_tokens.py -i ${VALID_FILE} -o ${OUTPUT_GOLD_DATA_DIR}/valid_words_ref.txt

python zgen_output_to_tokens.py -i ${TEST_FILE} -o ${OUTPUT_GOLD_DATA_DIR}/test_words_ref.txt

# with base NP symbols:

python zgen_output_to_tokens_npsyms.py -i ${TRAIN_FILE} -o ${OUTPUT_GOLD_DATA_DIR}/train_words_ref_npsyms.txt

python zgen_output_to_tokens_npsyms.py -i ${VALID_FILE} -o ${OUTPUT_GOLD_DATA_DIR}/valid_words_ref_npsyms.txt

python zgen_output_to_tokens_npsyms.py -i ${TEST_FILE} -o ${OUTPUT_GOLD_DATA_DIR}/test_words_ref_npsyms.txt

# remove EOS symbols for training the n-gram and LSTM models (EOS is not
# used in the current input data format convention):

for splitname in "train" "valid" "test";
do
    python remove_eos.py < ${PROCESSED_DATA_DIR}/temp_files/${splitname}_words_with_np_symbols.txt >${PROCESSED_DATA_DIR}/npsyms/${splitname}_words_with_np_symbols_no_eos.txt
    python remove_eos.py < ${PROCESSED_DATA_DIR}/temp_files/${splitname}_words_with_np_symbols_shuffled.txt >${PROCESSED_DATA_DIR}/npsyms/${splitname}_words_with_np_symbols_shuffled_no_eos.txt
done

# remove base NP symbols
for splitname in "train" "valid" "test";
do
    python remove_base_npsyms.py < ${PROCESSED_DATA_DIR}/temp_files/${splitname}_words_with_np_symbols.txt >${PROCESSED_DATA_DIR}/temp_files/${splitname}_words.txt

    # the file without base NPs should be shuffled without respect to base NPs:
    python remove_base_npsyms_shuffle.py < ${PROCESSED_DATA_DIR}/temp_files/${splitname}_words_with_np_symbols.txt >${PROCESSED_DATA_DIR}/temp_files/${splitname}_words_fullyshuffled.txt
done


for splitname in "train" "valid" "test";
do
    python remove_eos.py < ${PROCESSED_DATA_DIR}/temp_files/${splitname}_words.txt >${PROCESSED_DATA_DIR}/no_npsyms/${splitname}_words_no_eos.txt
    python remove_eos.py < ${PROCESSED_DATA_DIR}/temp_files/${splitname}_words_fullyshuffled.txt >${PROCESSED_DATA_DIR}/no_npsyms/${splitname}_words_fullyshuffled_no_eos.txt
done

## temporary files can be safely deleted at this point:
rm -r ${PROCESSED_DATA_DIR}/temp_files
