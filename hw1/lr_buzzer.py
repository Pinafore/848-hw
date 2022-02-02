import argparse
import itertools
import json
import pickle
from sklearn.linear_model import LogisticRegression
from typing import List
from feateng import feat_utils
import numpy as np


def read_vocab(filename:str):
    with open(filename, 'r') as fp:
        return [x.strip() for x in fp]


def create_train_dataset(vocab:List[str], filename:str):
    """Creates and returns inputs and labels for Logistic Regression from a json."""
    with open(filename) as fp:
        records = [json.loads(line) for line in fp]
    return feat_utils.prepare_train_inputs(vocab, records)


def create_eval_dataset(vocab:List[str], filename:str):
    with open(filename) as fp:
        records = [json.loads(line) for line in fp]
    example_guesses = itertools.groupby(records, lambda x: (x['id'], x['run_length']))
    inputs = []
    labels = []
    for _, guesses in example_guesses:
        input, label = feat_utils.prepare_eval_input(vocab, guesses)
        inputs.append(input)
        labels.append(label)
    return inputs, labels



class LogRegBuzzer:
    """A simple Logistic Regression based binary Buzzer."""
    def __init__(self):
        self.model = LogisticRegression(penalty='none', fit_intercept=False)
    
    def train(self, inputs, labels):
        self.model.fit(inputs, labels)
    
    def predict(self, inputs):
        return self.model.predict(inputs)

    def save(self, filename:str):
        pickle.dump(self.model, open(filename, 'wb'))
    
    def accuracy_score(self, inputs, labels):
        return self.model.score(inputs, labels)

    @classmethod
    def load(self, filename:str):
        return pickle.load(open(filename, 'rb'))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    #''' Switch between the toy and REAL EXAMPLES
    argparser.add_argument("--buzztrain", help="Positive class",
                           type=str, default="../data/small_guess.buzztrain.jsonl")
    argparser.add_argument("--buzzdev", help="Negative class",
                           type=str, default="../data/small_guess.buzzdev.jsonl")
    argparser.add_argument("--vocab", help="Vocabulary that can be features",
                           type=str, default="../data/small_guess.vocab")
    argparser.add_argument('--model_path', help="Path to save the Log Reg model.",
                            type=str, default="models/lr_buzzer.pickle")

    args = argparser.parse_args()

    vocab = read_vocab(args.vocab)
    
    train_inputs, train_labels = create_train_dataset(vocab, args.buzztrain)

    dev_inputs, dev_labels = create_train_dataset(vocab, args.buzzdev)

    buzzer = LogRegBuzzer()
    buzzer.train(train_inputs, train_labels)
    buzzer.save(args.model_path)
    print(f'Buzzer Train Accuracy: {buzzer.model.score(train_inputs, train_labels):.2f}')
    print(f'Buzzer Dev Accuracy  : {buzzer.model.score(dev_inputs, dev_labels):.2f}')