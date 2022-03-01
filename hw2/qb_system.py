from typing import List
from tqdm import tqdm
from qbdata import QantaDatabase
import torch
from pip import main
from tfidf_guesser import TfidfGuesser
from models import AnswerExtractor, Retriever, ReRanker, WikiLookup


class QuizBowlSystem:

    def __init__(self, wiki_lookup_path: str = '../data/wiki_lookup.2018.json') -> None:
        """Fill this method to create attributes, load saved models, etc
        Don't add any other arguments to this constructor. 
        If you really want to have arguments, they should have some default values set.
        """
        guesser = TfidfGuesser()
        print('Loading the Guesser model...')
        guesser.load('models/tfidf.pickle')

        print('Loding the Wiki Lookups...')
        self.wiki_lookup = WikiLookup(wiki_lookup_path)

        reranker = ReRanker()
        print('Loading the Reranker model...')
        reranker.load('amberoad/bert-multilingual-passage-reranking-msmarco')

        self.retriever = Retriever(guesser, reranker, wiki_lookup=self.wiki_lookup)

        answer_extractor_base_model = "csarron/bert-base-uncased-squad-v1"
        self.answer_extractor = AnswerExtractor()
        print('Loading the Answer Extractor model...')
        self.answer_extractor.load(answer_extractor_base_model)

    def retrieve_page(self, question: str, disable_reranking=False) -> str:
        """Retrieves the wikipedia page name for an input question."""
        with torch.no_grad():
            page = self.retriever.retrieve_answer_document(
                question, disable_reranking=disable_reranking)
            return page

    def execute_query(self, question: str, *, get_page=True) -> str:
        """Populate this method to do the following:
        1. Use the Retriever to get the top wikipedia page.
        2. Tokenize the question and the passage text to prepare inputs to the Bert-based Answer Extractor
        3. Predict an answer span for each question and return the list of corresponding answer texts."""
        with torch.no_grad():
            page = self.retrieve_page(question, disable_reranking=True)
            reference_text = self.wiki_lookup[page]['text']
            answer = self.answer_extractor.extract_answer(
                question, reference_text)[0]  # singleton list
            return (answer, page) if get_page else answer


if __name__ == "__main__":
    qa = QuizBowlSystem()
    qanta_db = QantaDatabase('../data/small.guessdev.json')
    small_set_questions = qanta_db.all_questions[:10]

    for question in tqdm(small_set_questions):
        answer = qa.execute_query(question.first_sentence)
