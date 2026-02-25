from core.llm.gpt_5_2 import GPT5_2
from core.llm.o3 import O3
from core.llm.gpt_4o import GPT4o
from core.llm.gemini import Gemini
from prompt.decision_making import DecisionMakingPrompt
from utils.normalize_long_answer import normalize_long_answer
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed


final_df = pd.read_csv("output/result_rag.csv")

# llm = GPT4o()
# llm = O3()
# llm = Gemini(model_name="gemini-3-pro-preview")
llm = GPT5_2(effort="high", verbosity="high")

OUTPUT_CSV_PATH = "output/result_decision_making.csv"
initial_df = pd.DataFrame(columns=["QIDX", "Question", "Short Answer", "Long Answer"])
initial_df.to_csv(OUTPUT_CSV_PATH, index=False)

def generate_parallel(max_workers=2):
    results = []

    def process_single(row):
        qidx = row["QIDX"]
        question = row["Question"]

        system_prompt = DecisionMakingPrompt.get_answer_system_prompt()
        user_prompt = DecisionMakingPrompt.get_answer_user_prompt(row)
        long_answer = llm.generate_response_single(system_prompt, user_prompt)
        short_answer = normalize_long_answer(long_answer)

        return {
            "QIDX": int(qidx),
            "Question": question,
            "Short Answer": short_answer,
            "Long Answer": long_answer,
        }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single, row) for idx, row in final_df.iterrows()]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Generating answers"):
            result = fut.result()
            results.append(result)

            df_row = pd.DataFrame([result])
            df_row.to_csv(OUTPUT_CSV_PATH, mode="a", header=False, index=False)

if __name__ == "__main__":
    generate_parallel(max_workers=5)
