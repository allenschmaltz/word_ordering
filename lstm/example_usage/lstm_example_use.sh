#############################################################################
####### EXAMPLE use of the LSTM model. This is meant for reference.
####### Change directory variables as necessary.
#############################################################################


#############################################################################
####### Training examples (These were the training parameters used for the results in our paper)
#############################################################################

DATA_DIR= # the path to the datasets folder (i.e, REPO_DIRECTORY/datasets)
MODEL_DIR= # a directory for saving model files
OUTPUT_LOG_DIR= # a directory for logging training progress
#######

mkdir ${MODEL_DIR}/checkpoint_m2seq35b20_with_npsyms_freq3_unkUNK
mkdir ${MODEL_DIR}/checkpoint_m2seq35b20_no_npsyms_freq3_unkUNK

SES=4 # gpu id

for INDEX_HOLDER in 1;
do

echo "Training with BNP"

th lstm_trainer.lua \
-data_dir ${DATA_DIR}/zgen_data_npsyms_freq3_unkUNK/npsyms \
-input_train_filename train_words_with_np_symbols_no_eos.txt \
-input_valid_filename valid_words_with_np_symbols_no_eos.txt \
-input_test_filename "" \
-perform_text_preprocessing_int 1 \
-run_test_on_valid 1 \
-run_test_on_tokens 0 \
-checkpoint_dir ${MODEL_DIR}/checkpoint_m2seq35b20_with_npsyms_freq3_unkUNK \
-savefile lstm \
-checkpoint_after_these_epochs "25" \
-seq_length 35 \
-model_group 2 \
-batch_size 20 \
-gpuid ${SES} >${OUTPUT_LOG_DIR}/train_m2seq35b20_with_npsyms_freq3_unkUNK_out.txt


echo "Training no BNP"

th lstm_trainer.lua \
-data_dir ${DATA_DIR}/zgen_data_npsyms_freq3_unkUNK/no_npsyms \
-input_train_filename train_words_no_eos.txt \
-input_valid_filename valid_words_no_eos.txt \
-input_test_filename "" \
-perform_text_preprocessing_int 1 \
-run_test_on_valid 1 \
-run_test_on_tokens 0 \
-checkpoint_dir ${MODEL_DIR}/checkpoint_m2seq35b20_no_npsyms_freq3_unkUNK \
-savefile lstm \
-checkpoint_after_these_epochs "25" \
-seq_length 35 \
-model_group 2 \
-batch_size 20 \
-gpuid ${SES} >${OUTPUT_LOG_DIR}/train_m2seq35b20_no_npsyms_freq3_unkUNK_out.txt

done


#############################################################################
####### Decoding examples
####### If you use the training models above, the following will replicate
####### the non-Gigaword beam 1 and 10 LSTM results in Table 2.
#############################################################################

OUTPUT_DIR= # a directory for saving reordered output
MODEL_DIR= # a directory for saving model files
REPO_DIR= # path to the repo
DATA_DIR=${REPO_DIR}/datasets # change this if you moved the datasets directory generated by the provided preprocessing scripts
TEMP_DIR= # a temp directory (for BLEU script, etc.)
OUTPUT_SCORES_LOG_FILE= # a file in which to save logging information, including BLEU scores (this should already exist, as lines will be appended below)
OUTPUT_DECODING_LOG_DIR=${OUTPUT_DIR}/logs # directory for additional logs
UNIGRAM_ARPA_FILE_WITH_BNP= # the unigram lm from BNP train file (e.g., LM1_noeos_withnpsyms_freq3_unkUNK.arpa created in ngram_example_use.sh)
UNIGRAM_ARPA_FILE_NO_BNP= # the unigram lm from the non-BNP train file (e.g., LM1_noeos_nonpsyms_freq3_unkUNK.arpa created in ngram_example_use.sh)

SPLIT_NAME=valid


SES=4 # gpu id

for BEAM_SIZE in 1 10;
do

LABEL=LSTM_FUTURE_BNP_${SPLIT_NAME}_BEAM${BEAM_SIZE}
echo $LABEL

OUTPUT_FILE=lstm_reordered_output_${SPLIT_NAME}_with_npsyms_futurecosts_beam${BEAM_SIZE}
FUTURE_MODEL=${UNIGRAM_ARPA_FILE_WITH_BNP}

th lstm_decoder.lua \
-data_dir ${DATA_DIR}/zgen_data_npsyms_freq3_unkUNK/npsyms \
-input_test_filename ${SPLIT_NAME}_words_with_np_symbols_shuffled_no_eos.txt \
-perform_text_preprocessing_int 1 \
-checkpoint_dir ${MODEL_DIR}/checkpoint_m2seq35b20_with_npsyms_freq3_unkUNK \
-checkpoint_filename lm_lstm_final.t7 \
-print_every 100 \
-base_nps 1 \
-base_np_symbols "<sonp>,<eonp>" \
-beam_size ${BEAM_SIZE} \
-unigram_lm_path_with_filename ${FUTURE_MODEL} \
-output_dir ${OUTPUT_DIR} \
-output_parse_filename ${OUTPUT_FILE} \
-output_scores_filename ${OUTPUT_FILE}_scores.txt \
-gpuid ${SES} > ${OUTPUT_DECODING_LOG_DIR}/${OUTPUT_FILE}_log_out.txt


python ${REPO_DIR}/data/postprocessing/randomly_replace_unkUNK.py \
--generated_reordering_with_unk ${OUTPUT_DIR}/${OUTPUT_FILE}.txt \
--gold_unprocessed ${DATA_DIR}/zgen_data_gold/${SPLIT_NAME}_words_ref_npsyms.txt \
--gold_processed ${DATA_DIR}/zgen_data_npsyms_freq3_unkUNK/npsyms/${SPLIT_NAME}_words_with_np_symbols_no_eos.txt \
--out_file ${OUTPUT_DIR}/${OUTPUT_FILE}_removed_unk.txt \
--remove_npsyms

echo "####" >> ${OUTPUT_SCORES_LOG_FILE}
echo ${LABEL} >> ${OUTPUT_SCORES_LOG_FILE}

${REPO_DIR}/analysis/eval/zgen_bleu/ScoreBLEU.sh -t ${OUTPUT_DIR}/${OUTPUT_FILE}_removed_unk.txt -r ${DATA_DIR}/zgen_data_gold/${SPLIT_NAME}_words_ref.txt -odir ${TEMP_DIR} >> ${OUTPUT_SCORES_LOG_FILE}

#######

LABEL=LSTM_NOFUTURE_BNP_${SPLIT_NAME}_BEAM${BEAM_SIZE}
echo $LABEL

OUTPUT_FILE=lstm_reordered_output_${SPLIT_NAME}_with_npsyms_nofuturecosts_beam${BEAM_SIZE}

th lstm_decoder.lua \
-data_dir ${DATA_DIR}/zgen_data_npsyms_freq3_unkUNK/npsyms \
-input_test_filename ${SPLIT_NAME}_words_with_np_symbols_shuffled_no_eos.txt \
-perform_text_preprocessing_int 1 \
-checkpoint_dir ${MODEL_DIR}/checkpoint_m2seq35b20_with_npsyms_freq3_unkUNK \
-checkpoint_filename lm_lstm_final.t7 \
-print_every 100 \
-base_nps 1 \
-base_np_symbols "<sonp>,<eonp>" \
-beam_size ${BEAM_SIZE} \
-output_dir ${OUTPUT_DIR} \
-output_parse_filename ${OUTPUT_FILE} \
-output_scores_filename ${OUTPUT_FILE}_scores.txt \
-gpuid ${SES} > ${OUTPUT_DECODING_LOG_DIR}/${OUTPUT_FILE}_log_out.txt


python ${REPO_DIR}/data/postprocessing/randomly_replace_unkUNK.py \
--generated_reordering_with_unk ${OUTPUT_DIR}/${OUTPUT_FILE}.txt \
--gold_unprocessed ${DATA_DIR}/zgen_data_gold/${SPLIT_NAME}_words_ref_npsyms.txt \
--gold_processed ${DATA_DIR}/zgen_data_npsyms_freq3_unkUNK/npsyms/${SPLIT_NAME}_words_with_np_symbols_no_eos.txt \
--out_file ${OUTPUT_DIR}/${OUTPUT_FILE}_removed_unk.txt \
--remove_npsyms

echo "####" >> ${OUTPUT_SCORES_LOG_FILE}
echo ${LABEL} >> ${OUTPUT_SCORES_LOG_FILE}

${REPO_DIR}/analysis/eval/zgen_bleu/ScoreBLEU.sh -t ${OUTPUT_DIR}/${OUTPUT_FILE}_removed_unk.txt -r ${DATA_DIR}/zgen_data_gold/${SPLIT_NAME}_words_ref.txt -odir ${TEMP_DIR} >> ${OUTPUT_SCORES_LOG_FILE}


#######

LABEL=LSTM_FUTURE_NOBNP_${SPLIT_NAME}_BEAM${BEAM_SIZE}
echo $LABEL

OUTPUT_FILE=lstm_reordered_output_${SPLIT_NAME}_no_npsyms_futurecosts_beam${BEAM_SIZE}
FUTURE_MODEL=${UNIGRAM_ARPA_FILE_NO_BNP}

th lstm_decoder.lua \
-data_dir ${DATA_DIR}/zgen_data_npsyms_freq3_unkUNK/no_npsyms \
-input_test_filename ${SPLIT_NAME}_words_fullyshuffled_no_eos.txt \
-perform_text_preprocessing_int 1 \
-checkpoint_dir ${MODEL_DIR}/checkpoint_m2seq35b20_no_npsyms_freq3_unkUNK \
-checkpoint_filename lm_lstm_final.t7 \
-print_every 100 \
-base_nps 0 \
-beam_size ${BEAM_SIZE} \
-unigram_lm_path_with_filename ${FUTURE_MODEL} \
-output_dir ${OUTPUT_DIR} \
-output_parse_filename ${OUTPUT_FILE} \
-output_scores_filename ${OUTPUT_FILE}_scores.txt \
-gpuid ${SES} > ${OUTPUT_DECODING_LOG_DIR}/${OUTPUT_FILE}_log_out.txt


python ${REPO_DIR}/data/postprocessing/randomly_replace_unkUNK.py \
--generated_reordering_with_unk ${OUTPUT_DIR}/${OUTPUT_FILE}.txt \
--gold_unprocessed ${DATA_DIR}/zgen_data_gold/${SPLIT_NAME}_words_ref.txt \
--gold_processed ${DATA_DIR}/zgen_data_npsyms_freq3_unkUNK/no_npsyms/${SPLIT_NAME}_words_no_eos.txt \
--out_file ${OUTPUT_DIR}/${OUTPUT_FILE}_removed_unk.txt \
--remove_npsyms

echo "####" >> ${OUTPUT_SCORES_LOG_FILE}
echo ${LABEL} >> ${OUTPUT_SCORES_LOG_FILE}

${REPO_DIR}/analysis/eval/zgen_bleu/ScoreBLEU.sh -t ${OUTPUT_DIR}/${OUTPUT_FILE}_removed_unk.txt -r ${DATA_DIR}/zgen_data_gold/${SPLIT_NAME}_words_ref.txt -odir ${TEMP_DIR} >> ${OUTPUT_SCORES_LOG_FILE}

#######

LABEL=LSTM_NOFUTURE_NOBNP_${SPLIT_NAME}_BEAM${BEAM_SIZE}
echo $LABEL

OUTPUT_FILE=lstm_reordered_output_${SPLIT_NAME}_no_npsyms_nofuturecosts_beam${BEAM_SIZE}

th lstm_decoder.lua \
-data_dir ${DATA_DIR}/zgen_data_npsyms_freq3_unkUNK/no_npsyms \
-input_test_filename ${SPLIT_NAME}_words_fullyshuffled_no_eos.txt \
-perform_text_preprocessing_int 1 \
-checkpoint_dir ${MODEL_DIR}/checkpoint_m2seq35b20_no_npsyms_freq3_unkUNK \
-checkpoint_filename lm_lstm_final.t7 \
-print_every 100 \
-base_nps 0 \
-beam_size ${BEAM_SIZE} \
-output_dir ${OUTPUT_DIR} \
-output_parse_filename ${OUTPUT_FILE} \
-output_scores_filename ${OUTPUT_FILE}_scores.txt \
-gpuid ${SES} > ${OUTPUT_DECODING_LOG_DIR}/${OUTPUT_FILE}_log_out.txt


python ${REPO_DIR}/data/postprocessing/randomly_replace_unkUNK.py \
--generated_reordering_with_unk ${OUTPUT_DIR}/${OUTPUT_FILE}.txt \
--gold_unprocessed ${DATA_DIR}/zgen_data_gold/${SPLIT_NAME}_words_ref.txt \
--gold_processed ${DATA_DIR}/zgen_data_npsyms_freq3_unkUNK/no_npsyms/${SPLIT_NAME}_words_no_eos.txt \
--out_file ${OUTPUT_DIR}/${OUTPUT_FILE}_removed_unk.txt \
--remove_npsyms

echo "####" >> ${OUTPUT_SCORES_LOG_FILE}
echo ${LABEL} >> ${OUTPUT_SCORES_LOG_FILE}

${REPO_DIR}/analysis/eval/zgen_bleu/ScoreBLEU.sh -t ${OUTPUT_DIR}/${OUTPUT_FILE}_removed_unk.txt -r ${DATA_DIR}/zgen_data_gold/${SPLIT_NAME}_words_ref.txt -odir ${TEMP_DIR} >> ${OUTPUT_SCORES_LOG_FILE}

done