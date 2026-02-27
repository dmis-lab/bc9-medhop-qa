from core.medhop_dataset import MedHopDataset
from core.llm.gpt_5_2 import GPT5_2
from core.llm.o3 import O3
from core.llm.gpt_4o import GPT4o
from core.llm.gemini import Gemini
from core.retriever.elastic_wikipedia import ElasticWikipedia
from core.retriever.dense_wikipedia import DenseWikipedia
from core.reranker.medcpt_reranker import MedCptReranker
from prompt.rationale import RationalePrompt
from utils.normalize_long_answer import normalize_long_answer
from tqdm import tqdm
import pandas as pd
import torch
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed


torch.set_num_threads(32)

medhop = MedHopDataset()
medhop.load_csv("data/MedHopQA_Test_Dataset.csv")

# llm = GPT4o()
# llm = O3()
# llm = GPT5_2(effort="high", verbosity="high")
llm = Gemini(model_name="gemini-3-pro-preview", use_google_search=False)

# retriever = DenseWikipedia()
retriever = ElasticWikipedia()

OUTPUT_CSV = "output/result_rationale.csv"
initial_df = pd.DataFrame(columns=["QIDX", "Question", "rationale_short", "rationale_long"])
initial_df.to_csv(OUTPUT_CSV, index=False)

num_gpus = torch.cuda.device_count()
pools = []

def init_model_on_worker(device_id):
    torch.cuda.set_device(device_id)
    global reranker
    reranker = MedCptReranker(device_id)

for gpu_id in range(num_gpus):
    pool = ProcessPoolExecutor(
        max_workers=1,
        initializer=init_model_on_worker,
        initargs=(gpu_id,)
    )
    pools.append(pool)

def rerank_on_gpu(question, docs):
    torch.cuda.empty_cache()
    scored = reranker.rerank_docs(question, docs, 200)

    return scored

def process_row(row):
    qidx = row["qidx"]
    question = row["question"]

    system_prompt = RationalePrompt.get_transform_system_prompt()
    user_prompt = RationalePrompt.get_transform_user_prompt(question)
    rationale_result = llm.generate_response_single(system_prompt, user_prompt)
    contexts = retriever.search(rationale_result, top_k=1000)

    pool_idx = process_row.next_pool
    process_row.next_pool = (pool_idx + 1) % num_gpus

    future = pools[pool_idx].submit(rerank_on_gpu, question, contexts)

    return qidx, question, future

process_row.next_pool = 0


def generate_parallel(devset, max_workers=8):
    with ThreadPoolExecutor(max_workers=max_workers) as thread_pool:
        thread_futs = [
            thread_pool.submit(process_row, row)
            for row in devset
        ]
        for tf in tqdm(as_completed(thread_futs),
                       total=len(thread_futs),
                       desc="Generating short answers"):

            qidx, question, rerank_fut = tf.result()
            scored_docs = rerank_fut.result()
            if len(scored_docs) == 0:
                continue

            system_prompt = RationalePrompt.get_answer_system_prompt()
            user_prompt = RationalePrompt.get_answer_user_prompt(question, scored_docs)
            long_answer = llm.generate_response_single(system_prompt, user_prompt)
            short_answer = normalize_long_answer(long_answer)

            result_data = {
                "QIDX": int(qidx),
                "Question": question,
                "rationale_short": short_answer,
                "rationale_long": long_answer,
            }

            df_row = pd.DataFrame([result_data])
            df_row.to_csv(OUTPUT_CSV, mode="a", header=False, index=False)

generate_parallel(medhop.data, max_workers=5)
