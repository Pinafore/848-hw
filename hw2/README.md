Homework 2: Just pages are not enough! An extractive approach to QA.
=

Prequisites:
------------
* Understanding of HW1 components
* A high-level understanding of BERT
* PyTorch and HuggingFace transformers library

Problem Statement
------------------
The goal of this assignment is to develop an end to end QA system over QuizBowl dataset that incorporates the following:
1. **A document retriever** that reads an open-domain question text and finds the page from a set of documents that is relevant and contains the answer. The retriever may have two sequential steps:
    1. Predict some top K wikipedia page-id guesses using a TF-IDF (or better) guesser.
    2. Rerank the top guesses using a BERT based **Reranker** and output the best page.
2. **An answer extractor** that takes in a question text, a reference text (page content) and finds a span within the reference text that is the answer to the input question.

The performance of the system will be measured using two metrics:
* mean accuracy over **_Exact Match_** (1 point if the prediction exactly matches the answer), and
* mean **_F1 score_** (over the answer tokens and prediction tokens). 

We will also use two settings to measure this performance:
* **_First Sentence predictions_** (System only tries to answer the question after reading the first sentence.)
* __*Last Sentence predictions*__ (System is given only the last sentence to answer the question, which is also a relatively very easy question for the same answer, given the pyramidal structure of the question.)

NOTE: In this hw, we will not be using the Buzzer based system.


What is provided?
-
We provide you a skeleton codebase that trains the tfidf guesser (like in HW1), loads some pretrained BERT-based models for other components and finally run an end to end evaluation of the QuizBowl system.

### **File Descriptions:**

* `base_models.py`: Contains the abstract base classes that provides an idea of High Level API for each component. This is very barebones. Though you are not required to change this, but you may.

* `models.py`: Gives a working skeleton of the ReRanker, Retriever and AnswerExtractor components that loads some pretrained models from huggingface repository and runs inference on the input questions. YOU WILL BE ADDING THE FINETUNING CODE HERE.

* `tfidf_guesser.py`: TF-IDF based Guesser (just like HW1). However, this time you may change the implementation and reparameterize this component too. You can choose to even entire replace TD-IDF guesser with Dense vector based retriever by using something like Sentence Embeddings, and fine them over QuizBowl accordingly.

* `qb_system.py` Provides a barebones QuizBowl system that instantiates these individual components and provides an API to run the models on input questions. YOU WILL BE EDITING THIS AS PER YOUR NEED.

* `eval_utils.py` Contains the utility functions to compute accuracy and F1 score for each predictions. This file MUST NOT BE CHANGED

* `qbdata.py`: Class files and util methods for QuizBowl questions and dataset. (Just like HW1).

* `run_e2e_eval.py` This file loads the QuizBowlSystem that you implement, loads the evaluation dataset, saves the model predictions and computes the metrics. YOU SHOULD NOT NEED TO CHANGE THIS FILE. However, you may change the input arguments to this file.

* `train_and_eval.sh` This shell script will download the required training and development fold of the QuizBowl dataset, train the tfidf guesser, and run the e2e eval script. Feel Free to change this as per your need.

* `models/` All the saved models must go in this directory.

* `outputs/`All model predictions must go in this directory.

### **Pretrained Models:**

We provide following pretrained huggingface models for ReRanker and Answer Extractor:

* [`amberoad/bert-multilingual-passage-reranking-msmarco`:](https://huggingface.co/amberoad/bert-multilingual-passage-reranking-msmarco) This module takes a search query [1] and a passage [2] and calculates if the passage matches the query. This model is trained using the Microsoft MS Marco Dataset. This training dataset contains approximately 400M tuples of a query, relevant and non-relevant passages. You may change this base pretrained model provided to you, but you need to finetune on the Quizbowl examples. We use this model to solve a Sentence Classification Task: Are given two inputs related or not?

* [`csarron/bert-base-uncased-squad-v1`](https://huggingface.co/csarron/bert-base-uncased-squad-v1) This model was fine-tuned from the HuggingFace BERT base uncased checkpoint on SQuAD1.1. This model is case-insensitive: it does not make a difference between english and English. Similar to above model, you need to finetune this over QuizBowl examples to achieve better performance on Answer Extraction Task.



What do I need to do?
--------------------

Your task is to pick any (or all) of the three below items to better the performance of the QuizBowl System as measured by Exact Match (EM) Accuracy and F1 score for the first and last sentence of the question text.

### Tasks:
* Finetune Reranker module with QuizBowl examples (or more).
* Finetune AnswerExtractor model with Quizbowl examples (or more).
* Replace Tf-Idf guesser with a better (dense vector) based guesser (and maybe disable reranking?)

You may also decide the how you want to prepare the inputs for the BERT models: attention masks, segment encodings, etc. However, you are only required to use the question text from the Qanta questions based on the task setting: first sentence / last sentence.

Data
----
Due to Github limits, we only provide you a small subset of training and validation data in the Homework repo. You can also use following scripts to download the full train and val data along with the wikipedia lookups for pages referred in the dataset:

These are slightly different examples from the ones suggested for HW1. We also provide the links of the Wiki pages referred in the examples. `WikiLookup` class in `qbdata.py` should be easy to use for accessing the page content of any answer page corresponding to a question.

```
# Train Data
wget "https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/qanta.train.2018.04.18.json"

# Validation Data
wget "https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/qanta.dev.2018.04.18.json"

# Wiki Lookup Data
wget "https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/datasets/wikipedia/wiki_lookup.json"
```

You can also use [qanta dataset](https://huggingface.co/datasets/qanta) from the huggingface `datasets` library.

The leaderboard score that determines the grade will be evaluated on a held-out test data that will be provided toward the end of the homework deadline which you all will use to submit the answers predictions in a json format.


System Environment
-------------------
Like previous homework, we will be using Python 3.6 for our autograder. The package version are provided in the `requirements.txt` file:
```
transformers==4.16.2
sentence-transformers==2.2.0
datasets==1.18.3
torch==1.9.0
torchvision==0.10.0
torchaudio==0.9.0
scikit-learn==0.24.2
nltk==3.6.7
spacy==3.2.2
```

Some Tutorial Links:
---------------
- [Extractive Question Answering using Huggingface](https://huggingface.co/docs/transformers/v4.16.2/en/task_summary#extractive-question-answering) and [Finetuning QA](https://huggingface.co/docs/transformers/master/en/custom_datasets#question-answering-with-squad)
- [Using QuestionAnswering Model output](https://huggingface.co/docs/transformers/v4.16.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput)
- [Question Answering Inference Pipeline](https://huggingface.co/docs/transformers/v4.16.2/en/main_classes/pipelines#transformers.QuestionAnsweringPipeline)
- [Using PretrainedTokenizer for corresponding models](https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer)
- [Facebook Faiss Library for efficient similarity search and clustering of dense vectors.](https://github.com/facebookresearch/faiss/wiki/Getting-started)

Leaderboard Score (40+ points)
------------------------------

40 points of your score will be generated from your performance on the
answer prediction competition on the leaderboard. The performance will be
evaluated considering F1 score for both first and last sentence setting 
on a held-out test set. If you are choosing to focus more on improving the retriever using dense representations of the documents, we will weigh more on the Retriever Accuracy.

If you can do much better than your peers, you can earn extra credit (up to 10 points).

Analysis (10 Points) [Keep it very consice]
--------------

The job of the written portion of the homework is to convince the grader that:
* Your approach and intuition works.
* You understand how and to what extent are each component contributing the final score.
* You had a clear methodology for incorporating changes in "model-input design", model selection, data used, etc.

Make sure that you have examples and quantitative evidence that your approach is working well, and include the metrics you chose to measure your system's performance. Be sure to explain how used the data
(e.g., did you have a development set?) and how you inspected the
results.

How to Turn in Your System
---------------------------
```
- qb_system.py
- models.py
- base_models.py
- ... any other dependent modules of qb_system.py
- models/
    |-- tfidf.pickle
    --- ...other saved models...
- outputs/
    |-- first_sent_predictions.json
    |-- last_sent_predictions.json
- requirements.txt (Optional)
- gather_resources.sh (Optional)
- custom_data/ (Optional)
    |-- xyz.json (Optional)
```
* You will be asked to run your system on the held out set toward the end of the homework and keep two predictions (`first_sent_predictions.json` and `last_sent_predictions.json`) within the `outputs/` dir. Nonetheless, we will be spot-verifying some randomly chosen examples on the server to check if online model predictions are same as the the ones in the submitted files. We also provide a sample prediction output file in `outputs/`. (You do not need to worry much about the format of the json file, as the functions in `run_e2e_eval.py` takes care of it)
* Changing `qbdata.py`, `eval_utils.py` and `run_e2e_eval.py` will have NO EFFECTS on the submissions as we will be using our own copy for evaluations on the Gradescope server. You should consider `qb_system.py` as your submission entry point.
* `requirements.txt`: You will be given above mentioned packages in the runtime environment. If you want to use other python packages in your solution, please provide them in a `requirements.txt` file at the root level. However, you mostly won't be needing any additional packages, and installing these packages would be counted towards the submission runtime limit (40 minutes). Python Packages in environment:
  
**If you do not correctly save your trained model, or do not submit one at all, the autograder will fail.**
* **Custom Data**: If you are using additional data at the inference time, put them in a directory `custom_data/` before running the script.
* **IMPORTANT**: If any of your files are >100MB, please submit a shell script named ``gather_resources.sh`` that will retrieve the files programmatically from a public location (e.g. a public S3 bucket) and save them in the required directory. An example `gather_resources.sh` could look like following:
```
wget "https://url/for/tfidf.pickle"
mv tfidf.pickle models/

wget "https://url/for/train/data/qanta.train.json
mkdir custom_data
mv qanta.train.json custom_data/custom.guesstrain.json
```
To host any custom model files, you can any public file hosting service as long as they can allow a GET request to download the same. A Tutorial for using the Amazon AWS S3 for their 5GB Free tier is now provided in this repo [here](../tutorials/aws-s3.md)

Turn in the above files via Gradescope through a zip format such that `feateng`, `models`, `custom_data`, `gather_resources.sh` and `requirements.txt` are all at the root level. We'll be using the leaderboard as before over Accuracy and Expected Win Probability measure.


``analysis.pdf``: Your **PDF** file containing your feature engineering analysis. There will be a separate Grades scope assignment created to submit this.

**Leaderboard name:** Please keep your submission name as `Group {group_number}: {whatever you like}`, and don't forget to add your team members as part of the submission on Gradescope!

HAPPY FINETUNING!!
