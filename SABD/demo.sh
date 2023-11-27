#!/bin/bash

# Makes programs, downloads sample data, trains a GloVe model, and then evaluates it.
# One optional argument can specify the language used for eval script: matlab, octave or [default] python

make

CORPUS=/home/TOSEM-DBRD/SABD/word_embedding/hadoop-old/hadoop-old_soft_clean.txt
VOCAB_FILE=./tmp/vocab_hadoop-old_80_20_soft_clean.txt
COOCCURRENCE_FILE=./tmp/cooccurrence_hadoop-old_soft_clean.bin
COOCCURRENCE_SHUF_FILE=./tmp/cooccurrence_hadoop-old_soft_clean.shuf.bin
BUILDDIR=build
SAVE_FILE=/home/TOSEM-DBRD/SABD/word_embedding/hadoop-old/glove_300d_hadoop-old
VERBOSE=2
MEMORY=12.0
VOCAB_MIN_COUNT=5
VECTOR_SIZE=300
MAX_ITER=30
WINDOW_SIZE=15
BINARY=0
NUM_THREADS=8


$BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE
$BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE
$BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE

$BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE