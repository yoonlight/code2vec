#!/usr/bin/env bash
###########################################################
# Change the following values to preprocess a new dataset.
# TRAIN_DIR, VAL_DIR and TEST_DIR should be paths to      
#   directories containing sub-directories with .java files
#   each of {TRAIN_DIR, VAL_DIR and TEST_DIR} should have sub-dirs,
#   and data will be extracted from .java files found in those sub-dirs).
# DATASET_NAME is just a name for the currently extracted 
#   dataset.                                              
# MAX_CONTEXTS is the number of contexts to keep for each 
#   method (by default 200).                              
# WORD_VOCAB_SIZE, PATH_VOCAB_SIZE, TARGET_VOCAB_SIZE -   
#   - the number of words, paths and target words to keep 
#   in the vocabulary (the top occurring words and paths will be kept). 
#   The default values are reasonable for a Tesla K80 GPU 
#   and newer (12 GB of board memory).
# NUM_THREADS - the number of parallel threads to use. It is 
#   recommended to use a multi-core machine for the preprocessing 
#   step and set this value to the number of cores.
# PYTHON - python3 interpreter alias.
DATASET_DIR=data
TRAIN_DIR=${DATASET_DIR}/my_train_dir
VAL_DIR=${DATASET_DIR}/my_val_dir
TEST_DIR=${DATASET_DIR}/my_test_dir
DATASET_NAME=my_dataset
MAX_CONTEXTS=200
WORD_VOCAB_SIZE=1301136
PATH_VOCAB_SIZE=911417
TARGET_VOCAB_SIZE=261245
NUM_THREADS=64
PYTHON=python3
###########################################################

TRAIN_DATA_FILE=${DATASET_NAME}.train.raw.txt
VAL_DATA_FILE=${DATASET_NAME}.val.raw.txt
TEST_DATA_FILE=${DATASET_NAME}.test.raw.txt

mkdir -p ${DATASET_DIR}
mkdir -p ${DATASET_DIR}/${DATASET_NAME}

TARGET_HISTOGRAM_FILE=${DATASET_DIR}/${DATASET_NAME}/${DATASET_NAME}.histo.tgt.c2v
ORIGIN_HISTOGRAM_FILE=${DATASET_DIR}/${DATASET_NAME}/${DATASET_NAME}.histo.ori.c2v
PATH_HISTOGRAM_FILE=${DATASET_DIR}/${DATASET_NAME}/${DATASET_NAME}.histo.path.c2v

echo "Creating histograms from the training data"
cat ${TRAIN_DATA_FILE} | cut -d' ' -f1 | awk '{n[$0]++} END {for (i in n) print i,n[i]}' > ${TARGET_HISTOGRAM_FILE}
cat ${TRAIN_DATA_FILE} | cut -d' ' -f2- | tr ' ' '\n' | cut -d',' -f1,3 | tr ',' '\n' | awk '{n[$0]++} END {for (i in n) print i,n[i]}' > ${ORIGIN_HISTOGRAM_FILE}
cat ${TRAIN_DATA_FILE} | cut -d' ' -f2- | tr ' ' '\n' | cut -d',' -f2 | awk '{n[$0]++} END {for (i in n) print i,n[i]}' > ${PATH_HISTOGRAM_FILE}
