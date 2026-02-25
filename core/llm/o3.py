import os
import json
import re
from openai import OpenAI
import time

API_KEY = ""
BATCH_INPUT_PATH = "/home02/d002a03/yein/workspace/medhopqa/jongmyung/test/batch_input_file.jsonl"
BATCH_OUTPUT_PATH = "/home02/d002a03/yein/workspace/medhopqa/jongmyung/test/batch_outputput_file.jsonl"

class O3():
    def __init__(self):
        self.model_name = 'o3'

        self.client = OpenAI(api_key=API_KEY)
        self.id = 0

    def generate_response(self, system_message, user_prompts):
        self.id = 0

        self.make_batch(system_message, user_prompts)
        batch = self.create_batch()
        self.polling_result(batch)

        return self.process_result()

    def generate_response_single(self, system_message, user_prompt):
        messages = [
            {"role": "system",  "content": system_message},
            {"role": "user",    "content": user_prompt},
        ]

        resp = self.client.chat.completions.create(
            model='o3',
            messages=messages,
        )
        return resp.choices[0].message.content.strip()

    def make_batch(self, system_message, user_prompts):
        with open(BATCH_INPUT_PATH, "w") as file:
            for idx, user_prompt in enumerate(user_prompts):
                entry = self.make_entry(system_message, user_prompt)
                file.write(json.dumps(entry) + "\n")

    def make_entry(self, system_message, user_prompt):
        self.id += 1

        return {
            "custom_id": f"case-{self.id}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "o3",
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt}
                ],
            }
        }
    
    def create_batch(self):
        batch_input_file = self.client.files.create(file=open(BATCH_INPUT_PATH, "rb"), purpose="batch")
        batch_input_file_id = batch_input_file.id

        batch = self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": "medhopqa batch"}
        )

        print(f"Batch created with ID: {batch.id}")

        return batch

    def polling_result(self, batch):
        batch_id = batch.id

        while True:
            status = self.client.batches.retrieve(batch_id).status
            if status == "completed":
                output_file = self.client.files.content(self.client.batches.retrieve(batch_id).output_file_id)
                with open(BATCH_OUTPUT_PATH, "w") as out_file:
                    out_file.write(output_file.text)
                print("Batch processing complete. Results saved to:", BATCH_OUTPUT_PATH)
                break
            elif status in ["failed", "cancelled", "expired"]:
                print("Batch processing failed with status:", status)
                break
            else:
                print("Batch status:", self.client.batches.retrieve(batch_id).request_counts)
                time.sleep(30)

    def process_result(self):
        with open(BATCH_OUTPUT_PATH, "r") as file:
            batch_output = [json.loads(line) for line in file]

            return [result["response"]["body"]["choices"][0]["message"]["content"].strip() for result in batch_output]