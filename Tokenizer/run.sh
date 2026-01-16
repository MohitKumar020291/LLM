#/bin/sh

base_dir=$(basename `pwd`)

for arg in "$@"; do
  case $arg in
    RUN_TYPE=*) #RUN_TYPE=BLAH_BLAH_STRING, Assigning RUN_TYPE to ignoring everything till *
        RUN_TYPE="${arg#*=}" ;;
    CONFIG_PATH=*) CONFIG_PATH="${arg#*=}" ;;
    CONFIG_NAME=*) CONFIG_NAME="${arg#*=}" ;;
    MODEL_NAME=*) MODEL_NAME="${arg#*=}" ;;
    TRAIN_STACKED_TOKENIZER=*) TRAIN_STACKED_TOKENIZER="${arg#*=}" ;;
    TRAIN_BPE=*) TRAIN_BPE="${arg#*=}" ;;
    CORPUS_PATH=*) CORPUS_PATH="${arg#*=}" ;;
    CORPUS_URLS=*) CORPUS_URLS="${arg#*=}" ;;
    VOCAB_SIZE=*) VOCAB_SIZE="${arg#*=}" ;;
  esac
done

if [ "$RUN_TYPE" == "cli" ]; then
    echo "RUNNING THROUGH CLI"
    if [ "$TRAIN_BPE" == "true" ]; then
      # There should be something called run default
      file_name="Tokenizer/train"
      file_name="${file_name//\//.}"
      echo $file_name
      python3 -m $file_name +corpus_path=$CORPUS_PATH +corpus_urls=$CORPUS_URLS +vocab_size=$VOCAB_SIZE
    elif [ "$TRAIN_STACKED_TOKENIZER" == "true" ]; then
      python3 $base_dir/Tokenizer/Experiment/stacked_tokenizer.py +model_name=$MODEL_NAME
    fi
elif [ "$RUN_TYPE" == "ncli" ]; then
    echo "RUNNING THROUGH CONFIG FILE, filename=$CONFIG_NAME and path=$CONFIG_PATH"
    python -m main --config-path $CONFIG_PATH --config-name $CONFIG_NAME
fi