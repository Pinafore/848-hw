import os
import argparse
import itertools
import json
import math
from posixpath import dirname
from tfidf_guesser import TfidfGuesser
from lr_buzzer import LogRegBuzzer, read_vocab
from qbdata import QantaDatabase
from collections import defaultdict
from feateng import feat_utils
from typing import List, Mapping, Any, Iterable


def winning_players_proportion(t: float):
    """Proportion of players that have correctly guessed the answer by length t.add()
    This metric is implemented in https://arxiv.org/pdf/1904.04792.pdf (Page 26)."""
    return min(1.0, 0.9931 + 0.0775 * t - 1.278 * t ** 2 + 0.588 * t ** 3)


def get_guess(guess_example: Mapping[str, Any]):
    """Return the guessed page_id from the guess dict."""
    return [k for k in guess_example.keys() if k.startswith('guess:')][0].split(':')[-1]


def get_the_best_guess(guesses: List[Mapping[str, Any]]):
    """Returns the highest scored guess (based on TfIdf guesser) from the top k guesses."""
    best_score, best_guess = max(
        [(g_dict['score'], get_guess(g_dict)) for g_dict in guesses])
    return best_guess


def compute_metrics(
        all_guesses: Iterable[Mapping[str, Any]], 
        true_labels: Mapping[int, str], 
        guess_vocab: List[str],
        buzzer: LogRegBuzzer, 
        penalize_incorrect_answers=False):
    """Computes four different metrics for the Guesser and Buzzer system:

    Expected Probability of Winning: 
        For each question, we generate top 5 guesses at different positions. The Buzzer looks at the top guesses, 
        considering the highest scored as the final prediction, and decides when to buzz. We stop looking further when it buzzes, 
        and evaluate whether the top guess at that point was indeed the correct guess.
        However, the longer the buzzer waits, system's likelihood of being the first one to buzz drops. 
        We reward expected score based on the winning_players_proportion(t) at that point (determined by human Quizbowl players).
        An average across all examples in the eval set determines the final score.

    Accuracy:
        We only consider the first buzz and whether the top guess was correct and reward a full point if so. 
        An average across all examples gives the accuracy.

    Buzz Percept: (Fun metric)
        What proportions of the total questions, did the system buzz at some point.

    Mean Buzz Position: (Fun metric)
        For all the questions that the system buzzed on, what is the mean relative position of the buzz.
        (This can help you determine if you are buzzing too early or too late)
    """
    n_examples = len(true_labels)

    guesses_collection = defaultdict(lambda: defaultdict(list))

    expected_win_prob = 0.0
    accuracy = 0.0

    n_buzzes = 0
    mean_buzz_position = 0.0

    for guess in all_guesses:
        qid = guess['id']
        t = guess['run_length']
        
        del guess['id']
        del guess['label']
        
        guesses_collection[qid][t].append(guess)

    for qid, qid_guesses in guesses_collection.items():
        buzzer_positions = list(qid_guesses)

        for t_i, t in enumerate(buzzer_positions):
            qid_guesses_at_t = qid_guesses[t]

            buzzer_input = feat_utils.prepare_eval_input(
                guess_vocab, qid_guesses_at_t)
            guess_at_t = get_the_best_guess(qid_guesses_at_t)

            if buzzer.predict([buzzer_input])[0] == 1:
                n_buzzes += 1
                if guess_at_t == true_labels[qid]:
                    accuracy += 1
                    mean_buzz_position += t_i
                    expected_win_prob += winning_players_proportion(t_i)
                elif penalize_incorrect_answers:
                    expected_win_prob -= winning_players_proportion(t_i) * 0.5
                break

    expected_win_prob /= n_examples
    accuracy /= n_examples
    buzz_ratio = n_buzzes / n_examples
    if n_buzzes > 0:
        mean_buzz_position /= n_buzzes
    else:
        mean_buzz_position = 0

    return {
        'accuracy': accuracy * 100,
        'expected_win_prob': expected_win_prob,
        'buzz_percent': buzz_ratio * 100,
        'mean_buzz_position': mean_buzz_position,
        }



if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--guesser_model_path", help="Model path for TfidfGuesser",
                           type=str, default="models/tfidf.pickle")
    argparser.add_argument("--buzzer_model_path", help="Model path for LogRegBuzzer",
                           type=str, default="models/lr_buzzer.pickle")
    argparser.add_argument("--eval_dataset_path", help="Dataset Path for the eval dataset",
                           type=str, default="../data/small.buzzdev.json")
    argparser.add_argument("--vocab_path", help="Vocabulary that can be features",
                           type=str, default="../data/small_guess.vocab")
    args = argparser.parse_args()

    guesser = TfidfGuesser.load(args.guesser_model_path)

    buzzer = LogRegBuzzer.load(args.buzzer_model_path)

    vocab = read_vocab(args.vocab_path)
    buzz_dataset = QantaDatabase(args.eval_dataset_path)
    buzz_eval_questions = buzz_dataset.buzz_dev_questions

    jsonl_filename = 'outputs/guess_buzz_eval.jsonl'
    os.makedirs(os.path.dirname(jsonl_filename), exist_ok=True)

    feat_utils.write_guess_json(guesser, jsonl_filename, buzz_eval_questions)

    with open(jsonl_filename) as fp:
        all_guesses = [json.loads(line) for line in fp]

    true_labels = {
        q.qanta_id: q.page
        for q in buzz_eval_questions
    }

    metrics = compute_metrics(all_guesses, true_labels, vocab, buzzer)
    for key, value in metrics.items():
        print(f'{key:20s}:{value:8.3f}')