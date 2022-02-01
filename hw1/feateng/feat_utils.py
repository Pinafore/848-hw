import imp
import json
from collections import OrderedDict
from typing import Iterable, Mapping, Any, List, Tuple
import numpy as np

kSEED = 1701
kBIAS = "BIAS_CONSTANT"


def prepare_train_inputs(vocab: List[str], examples: Iterable[Mapping[str, Any]]) -> List[np.ndarray]:
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
    
    :param vocab: List of possible guesses and categories
    :param sub_examples: An iterable of python dicts representing top-k guesses 
    of a QANTA example at a particular run length. It has the following default schema:
    {
            "id": str,
            "label": str,
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


def write_guess_json(guesser, filename, fold, run_length=200, censor_features=["id", "label"], num_guesses=5):
    """
    Returns the vocab, which is a list of all features.
    
    The only part where you would need to add something to this function could be while preparing the `guess` dict.
    You can include more more features if you want
    """
    vocab_set = OrderedDict({kBIAS:1})
    
    print("Writing guesses to %s" % filename)
    num = 0
    with open(filename, 'w') as outfile:
        total = len(fold)
        for qq in fold:
            num += 1
            if num % (total // 80) == 0:
                print('.', end='', flush=True)
            
            runs = qq.runs(run_length)
            guesses = guesser.guess(runs[0], max_n_guesses=5)

            for rr in runs[0]:
                guesses = guesser.guess([rr], max_n_guesses=num_guesses)
                for raw_guess in guesses[0]:
                    gg, ss = raw_guess
                    guess = {"id": qq.qanta_id,
                             "guess:%s" % gg: 1,
                             "run_length": len(rr)/1000,
                             "score": ss,
                             "label": qq.page==gg,
                             "category:%s" % qq.category: 1,
                             "year:%s" % qq.year: 1}

                    for ii in guess:
                        # Don't let it use features that would allow cheating
                        if ii not in censor_features and ii not in vocab_set:
                            vocab_set[ii] = 1

                    outfile.write(json.dumps(guess, sort_keys=True))
                    outfile.write("\n")
    print("")
    return [*vocab_set]