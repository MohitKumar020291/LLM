#!/usr/bin/bash

# THIS FILE IS FOR FULL TRAINING - TOKENIZER + GPT2
echo "Starting full training: Tokenizer + GPT2"

TOKENIZER_PATH=""
# Step 1: Train Tokenizer
for arg in "$@"; do
    case $arg in
        TRAIN_BPE=*) TRAIN_BPE="${arg#*=}" ;;
        CORPUS_PATH=*) CORPUS_PATH="${arg#*=}" ;;
        CORPUS_URLS=*) CORPUS_URLS="${arg#*=}" ;;
        VOCAB_SIZE=*) VOCAB_SIZE="${arg#*=}" ;;
        TRAIN_MODEL=*) TRAIN_MODEL="${arg#*=}" ;;
        TRAINING_CORPUS_URLS=*) TRAINING_CORPUS_URLS="${arg#*=}" ;;
        GENERATE=*) GENERATE="${arg#*=}" ;;
    esac
done


if [ "$TRAIN_BPE" == "true" ]; then
    echo "Training Tokenizer"
    # This returns path of the trained tokenizer
    python_output=$(python3 -m Tokenizer.train +corpus_path=$CORPUS_PATH +corpus_urls=$CORPUS_URLS +vocab_size=$VOCAB_SIZE)
    echo "Tokenizer training output:"
    echo "$python_output"
    # Extract tokenizer path from output
    tokenizer_path=$(echo "$python_output" | grep -oP 'Tokenizer/Cache/Tokenizers/\S+\.pkl')
    echo "Trained tokenizer path: $tokenizer_path"
    TOKENIZER_PATH=$tokenizer_path
fi

if [ "$TRAIN_MODEL" == "true" ]; then
    echo "Training GPT-2 Model with tokenizer at $TOKENIZER_PATH"
    if [ "$GENERATE" == "true" ]; then
        echo "Generating text with GPT-2 Model using tokenizer at $TOKENIZER_PATH"
        python3 -m GPT2.train +train_model=true +tokenizer_path=$TOKENIZER_PATH +generate=true +training_corpus_urls=$TRAINING_CORPUS_URLS +training_corpus_path=$CORPUS_PATH
        exit 0
    else
        echo "Training GPT-2 Model without text generation using tokenizer at $TOKENIZER_PATH"
        python3 -m GPT2.train +train_model=true +tokenizer_path=$TOKENIZER_PATH +generate=false +training_corpus_urls=$TRAINING_CORPUS_URLS +training_corpus_path=$CORPUS_PATH
        exit 0
    fi
fi

if [ "$GENERATE" == "true" ]; then
    echo "Generating text with GPT-2 Model using tokenizer at $TOKENIZER_PATH"
    python3 -m GPT2.train +train_model=false +tokenizer_path=$TOKENIZER_PATH +generate=true +training_corpus_urls=$TRAINING_CORPUS_URLS +training_corpus_path=$CORPUS_PATH
    exit 0
fi