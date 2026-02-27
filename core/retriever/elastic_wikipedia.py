import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from retriever_base import RetrieverBase
from elasticsearch import Elasticsearch
from tqdm import tqdm
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class ElasticWikipedia(RetrieverBase):
    def __init__(self):
        super().__init__()

    def load_retriever(self):
        self.client = Elasticsearch(
            "http://localhost:9234",
            basic_auth=("id", "pw"),
            verify_certs=False,
        )

    def search(self, query, top_k=3):
        request = {
            "query": {
                "bool": {
                    "must": {"match": {"chunk_text": query}},
                }
            },
            "size": top_k,
        }

        response = self.client.search(index="wikipedia_chunks", body=request)

        return [
            {
                'title': hit['_source']['chunk_text'].split('\n')[0].strip(),
                'text': ' '.join(hit['_source']['chunk_text'].split('\n')[1:]).strip(),
                'chunk_id': hit['_id'],
            }
            for hit in response['hits']['hits']
        ]

    def batch_search(self, queries, top_k=3):
        results = []

        for query in tqdm(queries, desc="Elastic Batch Search"):
            search_result = self.search(query, top_k)
            results.append(search_result)

        return results
