#!/usr/bin/env bash

data_dir="../data"

mkdir -p models

# Download the full train data
if [ ! -f "${data_dir}/qanta.train.2018.json" ]; then
    echo "Downloading qanta.train.2018.04.18.json as full train set."
    wget "https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/qanta.train.2018.04.18.json"
    mv qanta.train.2018.04.18.json "${data_dir}/qanta.train.2018.json"
    echo "Download complete.\n"
fi

# Download the full dev data
if [ ! -f "${data_dir}/qanta.dev.2018.json" ]; then
    echo "Downloading qanta.dev.2018.04.18.json as full dev set."
    wget "https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/qanta.dev.2018.04.18.json"
    mv qanta.dev.2018.04.18.json "${data_dir}/qanta.dev.2018.json"
    echo "Download complete.\n"
fi

# Download the Wiki Look up jsons
if [ ! -f "${data_dir}/wiki_lookup.2018.json" ]; then
    echo "Downloading Wiki Lookup jsons 2018.04.18"
    wget "https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/wikipedia/wiki_lookup.json"
    mv wiki_lookup.json "${data_dir}/wiki_lookup.2018.json"
    echo "Download complete.\n"
fi

echo "Running tfidf_guesser.py..."

# Only run this when you change the train data for guesser
python tfidf_guesser.py \
    --guesstrain="${data_dir}/qanta.train.2018.json" \
    --guessdev="${data_dir}/qanta.dev.2018.json" \
    --model_path="models/tfidf.pickle" || exit 1

# Remove the debug flag if you really want to run on eval set
python run_e2e_eval.py --debug_run 

