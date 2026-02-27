import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from reranker_base import RerankerBase
import requests

MODEL_NAME = "Qwen/Qwen3-Reranker-8B"
URL = "http://localhost:1234/v1/rerank"

PREFIX = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
QUERY_TEMPLATE = "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
DOCUMENT_TEMPLATE = "<Document>: {doc}{suffix}"
INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)

class Qwen3Reranker(RerankerBase):
    def __init__(self):
        pass

    def rerank_docs(self, query: str, docs: list[dict], rerank_topk: int = 10) -> list[dict]:
        documents = [
            DOCUMENT_TEMPLATE.format(doc=d.get("text", str(d)), suffix=SUFFIX) for d in docs
        ]

        response = requests.post(URL,
                                json={
                                    "model": MODEL_NAME,
                                    "query": QUERY_TEMPLATE.format(prefix=PREFIX, instruction=INSTRUCTION, query=query),
                                    "documents": documents,
                                    "top_n": rerank_topk
                                })
        response = response.json()

        ranked_results = []

        api_results = response.get("results", [])

        for res in api_results:
            idx = res["index"]
            score = res["relevance_score"]
            original_doc = docs[idx]
            
            original_doc["score"] = score
            ranked_results.append(original_doc)

        if not ranked_results:
            return docs[:rerank_topk]

        return ranked_results
    
    def rerank_batch(self, queries: list[str], docs: list[list[dict]], rerank_topk: int = 10, num_gpus: int = 1) -> list[list[dict]]:
        return []
