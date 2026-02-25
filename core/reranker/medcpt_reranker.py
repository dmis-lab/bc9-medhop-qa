import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from reranker_base import RerankerBase
from concurrent.futures import ProcessPoolExecutor
import torch
from itertools import islice
from math import ceil

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def chunked_iterable(iterable, size):
    it = iter(iterable)
    for first in it:
        yield [first, *islice(it, size - 1)]

def process_chunk(device_id, query_chunk, doc_chunk, rerank_topk):
    reranker = MedCptReranker(device=f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    results = []
    for query, doc_list in zip(query_chunk, doc_chunk):
        results.append(reranker.rerank_docs(query, doc_list, rerank_topk))
    return results


class MedCptReranker(RerankerBase):
    def __init__(self, device = 'cpu'):
        super().__init__(model_name_or_path='ncbi/MedCPT-Cross-Encoder', device=device)

    def rerank_docs(self, query: str, docs: list[dict], rerank_topk: int = 10) -> list[dict]:
        try:
            torch.cuda.empty_cache()

            pairs = [[query, d['title']+ '\n'+ d["text"]] for d in docs]
            encoded = self.tokenizer(
                pairs,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=512,
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                scores = self.model(**encoded).logits.squeeze(1).cpu()

            scored_docs = []
            for doc, score in zip(docs, scores.tolist()):
                scored = doc.copy()
                scored["score"] = score
                scored_docs.append(scored)

            scored_docs.sort(key=lambda x: x["score"], reverse=True)
            return scored_docs[:rerank_topk]
        except Exception as e:
            print('Reranker Error!', e)
            return []
    
    def rerank_batch(self, queries: list[str], docs: list[list[dict]], rerank_topk: int = 10, num_gpus: int = 1) -> list[list[dict]]:
        if len(queries) != len(docs):
            raise ValueError("The number of queries must match the number of document lists.")

        chunk_size = ceil(len(queries) / num_gpus)
        query_chunks = list(chunked_iterable(queries, chunk_size))
        doc_chunks = list(chunked_iterable(docs, chunk_size))

        results = []
        with ProcessPoolExecutor(max_workers=num_gpus) as executor:
            futures = [
                executor.submit(process_chunk, device_id, query_chunks[device_id], doc_chunks[device_id], rerank_topk)
                for device_id in range(num_gpus)
            ]
            for future in futures:
                results.extend(future.result())

        return results