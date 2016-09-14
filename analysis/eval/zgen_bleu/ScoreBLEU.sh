#!/bin/bash

usage(){
    echo "########################################################"
    echo "Usage: ScoreBLEU.sh -t hyp -r ref1 [-r ref2... -r refN]"
    echo "  -t       : translation hypothesis to be evaluated";
    echo "  -r       : one (or more) reference translation/s"; 
    echo "optional:";
    echo "  -odir        : output directory (default: scoring/ )";
    echo "  -case        : preserve case (by default, case insensitive)";
    echo "  -d           : detailed output";
    echo "############################################"; exit
}

BLEU=1
HYP="" 
ODIR="scoring"
CASE=0
VERBOSE=0

while [ "$1" != "" ]; do
    case $1 in
        -t )                    shift
	                        HYP=$1
                                ;;
        -r )                    shift
	                        REFS[${#REFS[*]}]=$1
                                ;;
        -odir )                 shift
	                        ODIR=$1
                                ;;
        -case)          	CASE=1
                                ;;
        -d)                     VERBOSE=1
                                ;;
        --help )                usage;
                                exit
                                ;;
        * )                     usage;
                                exit 1
    esac
    shift
done

if [ ! -e "$HYP" ]; then echo "Error: Unable to read HYP file $HYP !"; usage; exit; fi
if [ "${#REFS[*]}" -eq  0 ]; then usage; exit; fi
NREFS=`expr ${#REFS[*]}`
for a in `seq 0 $[ $NREFS - 1 ]`; do 
    if [ ! -r ${REFS[$a]} ]; then echo "Error: ${REFS[$a]} cannot be read!"; exit; fi
done


### PREPARE SGML FILEs
mkdir -p $ODIR
NAME=`basename $HYP`
SRC=$ODIR/$NAME.src.sgml
TRANS=$ODIR/$NAME.hyp.sgml
REF=$ODIR/$NAME.ref.sgml

N=`cat $HYP | wc -l | tr -d ' '`

echo '<tstset setid="y" srclang="src" trglang="trg">' > $TRANS
yes | head -$N | paste - $HYP | awk 'BEGIN{docid=""; segid=1}{if (docid!=$1) {if (docid!="") print "</doc>"; docid=$1; segid=1; printf("<doc docid=\"%s\" sysid=\"1\">\n",docid)} else {segid++} if (segid==1) tname="hl"; else tname="p"; printf("<%s>\n<seg id=\"%d\"> ",tname,segid); for (i=2; i<NF; i++) printf("%s ",$i); print $NF "</seg>"; print "</" tname ">"}END{print "</doc>\n</tstset>"}' >> $TRANS

echo '<srcset setid="y" srclang="src" trglang="trg">' > $SRC
yes | head -$N | paste - $HYP | awk 'BEGIN{docid=""; segid=1}{if (docid!=$1) {if (docid!="") print "</doc>"; docid=$1; segid=1; printf("<doc docid=\"%s\" sysid=\"1\">\n",docid)} else {segid++} if (segid==1) tname="hl"; else tname="p"; printf("<%s>\n<seg id=\"%d\"> ",tname,segid); for (i=2; i<NF; i++) printf("%s ",$i); print $NF "</seg>"; print "</" tname ">"}END{print "</doc>\n</srcset>"}' >> $SRC


echo '<refset setid="y" srclang="src" trglang="trg">' > $REF
for k in `seq 1 $NREFS`; do
    a=$[ $k - 1 ];
    yes | head -$N | paste - ${REFS[$a]} | awk 'BEGIN{docid=""; segid=1}{if (docid!=$1) {if (docid!="") print "</doc>"; docid=$1; segid=1; printf("<doc docid=\"%s\" sysid=\"r'$k'\">\n",docid)} else {segid++} if (segid==1) tname="hl"; else tname="p"; printf("<%s>\n<seg id=\"%d\"> ",tname,segid); for (i=2; i<NF; i++) printf("%s ",$i); print $NF "</seg>"; print "</" tname ">"}END{print "</doc>"}' >> $REF
done
echo '</refset>' >> $REF


### COMPUTE BLEU SCORE  (text sets)
BLOPT=""
OCASE=""

if [ $CASE == 1 ]; then BLOPT="$BLOPT -c"; OCASE=".uppercase"; fi

SDIR=$(cd `dirname $0` && pwd)
if [ $VERBOSE -gt 0 ]; then 
    $SDIR/scripts/mteval-v13-ADRIA.pl -b -d 2 -r $REF -s $SRC -t $TRANS $BLOPT | tee $ODIR/$NAME.bleu.log$OCASE
else
    $SDIR/scripts/mteval-v13-ADRIA.pl -b -d 2 -r $REF -s $SRC -t $TRANS $BLOPT >& $ODIR/$NAME.bleu.log$OCASE
    grep 'BLEU score = ' $ODIR/$NAME.bleu.log$OCASE
fi

#rm -f $SRC $TRANS $REF


