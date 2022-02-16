from abc import ABC
from typing import Optional, List, Tuple

class BaseModel:

    def load(self, model_identifier: str):
        raise NotImplementedError
    
    def save(self, model_path: str):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError


class BaseGuesser(BaseModel):
    def guess(self, questions: List[str], max_n_guesses: Optional[int]) -> List[List[Tuple[str, float]]]:
        raise NotImplementedError


class BaseReRanker(BaseModel):
    def get_best_document(question: str, ref_texts: List[str]):
        raise NotImplementedError


class BaseRetriever(BaseModel):
    guesser: BaseGuesser
    reranker: BaseReRanker

    def retrieve_answer_document(self, question:str, disable_reranking=False):
        raise NotImplementedError


class BaseAnswerExtractor(BaseModel):
    def extract_answer(self, question: str, ref_text: str):
        raise NotImplementedError
