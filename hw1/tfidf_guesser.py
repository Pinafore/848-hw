from typing import List, Optional, Tuple
from collections import defaultdict, OrderedDict
import pickle
import argparse
from os import path

from typing import Union, Dict

from sklearn.feature_extraction.text import TfidfVectorizer

from qbdata import QantaDatabase
from feateng import feat_utils

BUZZ_NUM_GUESSES = 10
BUZZ_THRESHOLD = 0.3

class StubDatabase:
    def __init__(self):
        self.guess_train_questions = []

    def add(self, question):
        self.guess_train_questions.append(question)

class StubQuestion:
    def __init__(self, question, answer):
        self.text = question
        self.page = answer


class TfidfGuesser:
    """
    Class that, given a query, finds the most similar question to it.
    """
    def __init__(self):
        """
        Initializes data structures that will be useful later.
        """
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.i_to_ans = None

    def train(self, training_data: Union[StubDatabase, QantaDatabase], limit=-1) -> None:
        """
        Use a tf-idf vectorizer to analyze a training dataset and to process
        future examples.
        
        Keyword arguments:
        training_data -- The dataset to build representation from
        limit -- How many training data to use (default -1 uses all data)
        """
        
        questions = [x.text for x in training_data.guess_train_questions]
        answers = [x.page for x in training_data.guess_train_questions]

        if limit > 0:
            questions = questions[:limit]
            answers = answers[:limit]

        x_array = []
        y_array = []

        for doc, ans in zip(questions, answers):
            x_array.append(doc)
            y_array.append(ans)
        self.i_to_ans = {i: ans for i, ans in enumerate(y_array)}
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 3), max_df=.9
        ).fit(x_array)
        self.tfidf_matrix = self.tfidf_vectorizer.transform(x_array)

    def guess(self, questions: List[str], max_n_guesses: Optional[int]) -> List[List[Tuple[str, float]]]:
        """
        Given the text of questions, generate guesses (a tuple, page id and score) for each one.

        Keyword arguments:
        questions -- Raw text of questions in a list
        max_n_guesses -- How many top guesses to return
        """
        representations = self.tfidf_vectorizer.transform(questions)
        guess_matrix = self.tfidf_matrix.dot(representations.T).T
        guess_indices = (-guess_matrix).toarray().argsort(axis=1)[:, 0:max_n_guesses]
        guesses = []
        for i in range(len(questions)):
            idxs = guess_indices[i]
            guesses.append([(self.i_to_ans[j], guess_matrix[i, j]) for j in idxs])

        return guesses


    def confusion_matrix(self, evaluation_data: QantaDatabase, limit=-1) -> Dict[str, Dict[str, int]]:
        """
        Given a matrix of test examples and labels, compute the confusion
        matrixfor the current classifier.  Should return a dictionary of
        dictionaries where d[ii][jj] is the number of times an example
        with true label ii was labeled as jj.

        :param evaluation_data: Database of questions and answers
        :param limit: How many evaluation questions to use
        """

        questions = [x.text for x in evaluation_data.guess_dev_questions]
        answers = [x.page for x in evaluation_data.guess_dev_questions]

        if limit > 0:
            questions = questions[:limit]
            answers = answers[:limit]

        print("Eval on %i question" % len(questions))
            
        d = defaultdict(dict)
        data_index = 0
        guesses = [x[0][0] for x in self.guess(questions, max_n_guesses=1)]
        for gg, yy in zip(guesses, answers):
            d[yy][gg] = d[yy].get(gg, 0) + 1
            data_index += 1
            if data_index % 100 == 0:
                print("%i/%i for confusion matrix" % (data_index,
                                                      len(guesses)))
        return d
    
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'i_to_ans': self.i_to_ans,
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'tfidf_matrix': self.tfidf_matrix
            }, f)

    @classmethod
    def load(self, filepath):
        """
        Load the guesser from a saved file
        """
        
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
            guesser = TfidfGuesser()
            guesser.tfidf_vectorizer = params['tfidf_vectorizer']
            guesser.tfidf_matrix = params['tfidf_matrix']
            guesser.i_to_ans = params['i_to_ans']
            return guesser
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--guesstrain", default="data/small.guesstrain.json", type=str)
    parser.add_argument("--guessdev", default="data/small.guessdev.json", type=str)
    parser.add_argument("--buzztrain", default="data/small.buzztrain.json", type=str)
    parser.add_argument("--buzzdev", default="data/small.buzzdev.json", type=str)
    parser.add_argument("--limit", default=-1, type=int)
    parser.add_argument("--num_guesses", default=5, type=int)
    parser.add_argument("--vocab", default="", type=str)
    parser.add_argument("--model_path", default="models/tfidf.pickle", type=str)
    parser.add_argument("--buzztrain_predictions", default="", type=str)
    parser.add_argument("--buzzdev_predictions", default="", type=str)

    flags = parser.parse_args()

    print("Loading %s" % flags.guesstrain)
    guesstrain = QantaDatabase(flags.guesstrain)
    guessdev = QantaDatabase(flags.guessdev)
    
    tfidf_guesser = TfidfGuesser()
    tfidf_guesser.train(guesstrain, limit=flags.limit)
    tfidf_guesser.save(flags.model_path)

    confusion = tfidf_guesser.confusion_matrix(guessdev, limit=-1)
    print("Errors:\n=================================================")
    for ii in confusion:
        for jj in confusion[ii]:
            if ii != jj:
                print("%i\t%s\t%s\t" % (confusion[ii][jj], ii, jj))

    
    if flags.buzztrain_predictions:
        print("Loading %s" % flags.buzztrain)
        buzztrain = QantaDatabase(flags.buzztrain)        
        vocab = feat_utils.write_guess_json(tfidf_guesser, flags.buzztrain_predictions, buzztrain.buzz_train_questions, num_guesses=flags.num_guesses)

    if flags.vocab:
        with open(flags.vocab, 'w') as outfile:
            for ii in vocab:
                outfile.write("%s\n" % ii)

    if flags.buzzdev_predictions:
        assert flags.buzztrain_predictions, "Don't have vocab if you don't do buzztrain"
        print("Loading %s" % flags.buzzdev)    
        buzzdev = QantaDatabase(flags.buzzdev)
        feat_utils.write_guess_json(tfidf_guesser, flags.buzzdev_predictions, buzzdev.buzz_dev_questions, num_guesses=flags.num_guesses)
    
