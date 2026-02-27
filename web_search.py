from core.medhop_dataset import MedHopDataset
from core.llm.gemini import Gemini
from prompt.web_search import WebSearchPrompt
from utils.normalize_long_answer import normalize_long_answer
from tqdm import tqdm
import pandas as pd


medhop = MedHopDataset()
medhop.load_csv("data/MedHopQA_Test_Dataset.csv")

llm = Gemini(model_name="gemini-3-pro-preview")

OUTPUT_CSV = "output/result_web_search.csv"
initial_df = pd.DataFrame(columns=["QIDX", "Question", "web_short", "web_long"])
initial_df.to_csv(OUTPUT_CSV, index=False)

for row in tqdm(medhop.data):
    try:
        question = row["question"]

        system_prompt = WebSearchPrompt.get_answer_system_prompt()
        user_prompt = WebSearchPrompt.get_answer_user_prompt(question)
        long_answer = llm.generate_response_single(system_prompt, user_prompt)
        short_answer = normalize_long_answer(long_answer)

        result_data = {
            "QIDX": int(row["qidx"]),
            "Question": question,
            "web_short": short_answer,
            "web_long": long_answer,
        }

        df_row = pd.DataFrame([result_data])
        df_row.to_csv(OUTPUT_CSV, mode="a", header=False, index=False)
    except Exception as e:
        print(f"[Error] QIDX: {row.get('qidx', 'unknown')} failed. Error: {e}")
