from tqdm import tqdm
import json
from collections import OrderedDict
from typing import Iterable, Mapping, Any, List, Tuple
import numpy as np

import qbdata

kSEED = 1701
kBIAS = "BIAS_CONSTANT"


def prepare_train_inputs(vocab: List[str], examples: Iterable[Mapping[str, Any]]) -> Tuple[List[np.ndarray]]:
    """Fill this method to create input features representations and labels for training Logistic Regression based Buzzer.

    :param vocab: List of possible guesses and categories
    :param examples: An iterable of python dicts representing guesses 
    across all QANTA example in a dataset. It has the following default schema:
        {
            "id": str,
            "label": str,
            "guess:%s": 1,
            "run_length": float,
            "score": float,
            "category%s": 1,
            "year": int
        }

    You must return the fixed sized numpy.ndarray representing the input features.

    Currently, the function only uses the score a feature along with the bias.
    The logistic regression doesn't implicitly model intercept (or bias term), 
    it has to be explicitly provided as one of the input values.
    """
    inputs = np.array([[1.0, e['score']] for e in examples], dtype=np.float32)
    labels = np.array([e['label'] for e in examples], dtype=int)
    return inputs, labels


def prepare_eval_input(vocab: List[str], sub_examples: Iterable[Mapping[str, Any]]) -> List[np.ndarray]:
    """This function is used during end to end evaluation for computing expected win probability. 
    The evaluation is not done just over a logistic regressor, but with the final gold-answer to the question.
    You should assume that the guess with the highest score will be selected as the final prediction, 
    but you may use the properties of other guesses to determine the features to the logistic regression model.

    Note: Any label information will explicitly be removed before calling this function.

    :param vocab: List of possible guesses and categories
    :param sub_examples: An iterable of python dicts representing top-k guesses 
    of a QANTA example at a particular run length. It has the following default schema:
    {
            "guess:%s": 1,
            "run_length": float,
            "score": float,
            "category%s": 1,
            "year": int
    }
    """
    scores = [e['score'] for e in sub_examples]
    idx = np.argmax(scores)
    input = np.array([1.0, scores[idx]], dtype=np.float32)
    return input


def make_guess_dicts_from_question(question: qbdata.Question, guesser, run_length: int = 200, num_guesses: int = 5):
    """Creates an iterable of guess dictionaries from the guesser outputs. 
    Feel Free to add more features to the dictionary. 
    However, DO NOT add any label specific information as those would be removed explicitly
    and will be considered as breaking the Honor Code."""
    runs, _ = question.runs(run_length)
    runs_guesses = guesser.guess(runs, max_n_guesses=num_guesses)

    for question_prefix, guesses in zip(runs, runs_guesses):
        for raw_guess in guesses:
            page_id, score = raw_guess
            guess = {
                "id": question.qanta_id,
                "guess:%s" % page_id: 1,
                "run_length": len(question_prefix)/1000,
                "score": score,
                "label": question.page == page_id,
                "category:%s" % question.category: 1,
                "year:%s" % question.year: 1
            }
            yield guess


def write_guess_json(guesser, filename: str, questions: Iterable[qbdata.Question], run_length: int = 200, censor_features=["id", "label"], num_guesses: int = 5):
    """
    Returns the vocab, which is a list of all features.

    You DON'T NEED TO CHANGE THIS function.
    """
    vocab_set = OrderedDict({kBIAS: 1})

    print("Writing guesses to %s" % filename)

    string_buffer = []

    with open(filename, 'w') as outfile:
        for ques in tqdm(questions):
            guesses = make_guess_dicts_from_question(ques, guesser, run_length, num_guesses)

            for guess in guesses:
                for ii in guess:
                    # Don't let it use features that would allow cheating
                    if ii not in censor_features and ii not in vocab_set:
                        vocab_set[ii] = 1
                string_buffer.append(json.dumps(guess))
        outfile.write('\n'.join(string_buffer))
    print("")
    return [*vocab_set]
