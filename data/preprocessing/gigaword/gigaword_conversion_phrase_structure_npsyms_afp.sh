DATA_DIR=/n/rush_lab/users/schmaltz/projects/lm/gigaword_proc/stanford_phrase_structure

OUTPUT_DIR=/n/rush_lab/users/schmaltz/projects/lm/gigaword_proc/stanford_phrase_structure_npsyms
LOGS_DIR=/n/rush_lab/users/schmaltz/projects/lm/gigaword_proc/stanford_phrase_structure_npsyms/logs


counter=0
for filename in $DATA_DIR/afp_eng_*.xml.gz_phrase_structure.txt;
do
    if [ ! -f $LOGS_DIR/$(basename $filename)_npsyms_complete.txt ]; then
        echo "About to process file: "${filename}
        python add_npsyms_to_gigaword.py -i ${filename} -o $OUTPUT_DIR/$(basename $filename)_processed 2>$LOGS_DIR/$(basename $filename)_out.txt || echo "error processing " >$LOGS_DIR/$(basename $filename)_err.txt
        echo "Processed file: "${filename}
        let counter++
        echo "File number: "$counter

        if [ ! -f $LOGS_DIR/$(basename $filename)_err.txt ]; then
            echo "completed " >$LOGS_DIR/$(basename $filename)_npsyms_complete.txt
        fi

    fi
done
