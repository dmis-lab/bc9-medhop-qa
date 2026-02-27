import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from retriever_base import RetrieverBase
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import asyncio
import aiohttp

class DenseWikipedia(RetrieverBase):
    def __init__(self, url="http://localhost:8080/search"):
        super().__init__()
        self.url = url
        self.session = None

    def load_retriever(self):
        pass

    def search(self, query, top_k=3):
        async def _do_request():
            async with aiohttp.ClientSession() as session:
                payload = {"query": query, "top_k": top_k}

                try:
                    async with session.post(self.url, json=payload) as response:
                        if response.status != 200:
                            text = await response.text()
                            print(f"[Error] Request failed: {text}")
                            return None

                        result = await response.json()

                        return [
                            {
                                'title': doc['title'],
                                'text': doc['content'],
                                'chunk_id': doc['id'],
                            }
                            for doc in result.get('results', [])
                        ]
                except Exception as e:
                    print(f"[Exception] Request error: {e}")
                    return None

        return asyncio.run(_do_request())

    def batch_search(self, queries, top_k=3):
        pass

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()
