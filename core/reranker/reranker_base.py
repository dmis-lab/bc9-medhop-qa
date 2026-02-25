from abc import ABC, abstractmethod
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class RerankerBase(ABC):
    def __init__(self, model_name_or_path: str, device: str):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self.load_model()

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path).to(self.device)

    @abstractmethod
    def rerank_docs(self,
                    query: str,
                    docs: list[dict],
                    rerank_topk: int = 10) -> list[dict]:
        pass

    @abstractmethod
    def rerank_batch(self,
                    queries: list[str],
                    docs: list[list[dict]],
                    rerank_topk: int = 10) -> list[list[dict]]:
        pass
