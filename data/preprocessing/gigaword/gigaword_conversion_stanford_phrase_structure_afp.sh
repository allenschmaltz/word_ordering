DATA_DIR=/n/rush_lab/data/ldc/media/srush/wdc/ldc/gigaword/v5/annotated/data/xml

OUTPUT_DIR=/n/rush_lab/users/schmaltz/projects/lm/gigaword_proc/stanford_phrase_structure
LOGS_DIR=/n/rush_lab/users/schmaltz/projects/lm/gigaword_proc/stanford_phrase_structure/logs


counter=0
for filename in $DATA_DIR/afp_eng_*.xml.gz;
do
    if [ ! -f $LOGS_DIR/$(basename $filename)_phrase_structure_complete.txt ]; then
        echo "About to process file: "${filename}
        java -cp build/agiga-1.2.jar:lib/* edu.jhu.agiga.AgigaPrinter stanford-phrase-structure ${filename} >$OUTPUT_DIR/$(basename $filename)_phrase_structure.txt 2>$LOGS_DIR/$(basename $filename)_out.txt || echo "error processing " >$LOGS_DIR/$(basename $filename)_err.txt
        echo "Processed file: "${filename}
        let counter++
        echo "File number: "$counter

        if [ ! -f $LOGS_DIR/$(basename $filename)_err.txt ]; then
            echo "completed " >$LOGS_DIR/$(basename $filename)_phrase_structure_complete.txt
        fi

    fi
done
