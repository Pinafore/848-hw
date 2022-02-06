#!/usr/bin/env bash

data_dir="../data"

mkdir -p outputs

# Download the full dev data
if [ ! -f "${data_dir}/qanta.dev.json" ]; then
    echo "Downloading qanta.dev.2021.12.20.json as full dev set."
    wget "https://obj.umiacs.umd.edu/qanta-jmlr-datasets/qanta.dev.2021.12.20.json"
    mv qanta.dev.2021.12.20.json "${data_dir}/qanta.dev.json"
    echo "Download complete.\n"
fi

echo "Running tfidf_guesser.py..."

# Only run this when you change the train data for guesser or add new features
# to `make_guess_dicts_from_question` in feateng/feat_utils.py
python tfidf_guesser.py \
    --guesstrain="${data_dir}/small.guesstrain.json" \
    --guessdev="${data_dir}/small.guessdev.json" \
    --buzztrain="${data_dir}/small.buzztrain.json" \
    --buzzdev="${data_dir}/small.buzzdev.json" \
    --buzztrain_predictions="outputs/buzztrain_predictions_small.jsonl" \
    --buzzdev_predictions="outputs/buzzdev_predictions_small.jsonl" \
    --vocab="models/guess.vocab" \
    --model_path="models/tfidf.pickle" || exit 1

echo "\nTrained TfIDF Guesser and prepared files for Buzzer\n."

python lr_buzzer.py \
    --buzztrain="outputs/buzztrain_predictions_small.jsonl" \
    --buzzdev="outputs/buzzdev_predictions_small.jsonl" \
    --vocab="models/guess.vocab" \
    --model_path="models/lr_buzzer.pickle" || exit 1

echo "\nTrained Buzzer. Running End to end on full dev data..."

# Use the full dev set for evaluation
python run_e2e_eval.py \
    --guesser_model_path="models/tfidf.pickle" \
    --buzzer_model_path="models/lr_buzzer.pickle" \
    --vocab_path="models/guess.vocab" \
    --eval_dataset_path="${data_dir}/qanta.dev.json" || exit 1