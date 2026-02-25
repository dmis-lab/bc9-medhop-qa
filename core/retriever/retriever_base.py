from abc import ABC, abstractmethod

class RetrieverBase(ABC):
    def __init__(self):
        self.load_retriever()

    @abstractmethod
    def load_retriever(self):
        pass

    @abstractmethod
    def search(self, query, top_k):
        pass

    @abstractmethod
    def batch_search(self, queries, top_k):
        pass
