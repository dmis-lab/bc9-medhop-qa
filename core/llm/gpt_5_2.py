import os
import json
import re
from openai import OpenAI
import time

BATCH_INPUT_PATH = "./gpt_5.2_batch_input_file.jsonl"
BATCH_OUTPUT_PATH = "./gpt5.2_batch_outputput_file.jsonl"

class GPT5_2():
    def __init__(self, effort='none', verbosity='medium'):
        self.model_name = 'gpt-5.2'

        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.id = 0
        self.effort = effort
        self.verbosity = verbosity

    def generate_response(self, system_message, user_prompts):
        self.id = 0

        self.make_batch(system_message, user_prompts)
        batch = self.create_batch()
        self.polling_result(batch)

        return self.process_result()

    def generate_response_single(self, system_message, user_prompt):
        resp = self.client.responses.create(
            model=self.model_name,
            instructions=system_message,
            input=user_prompt,
            max_output_tokens=4096,
            reasoning={
                'effort': self.effort,
            },
            text={
                'verbosity': self.verbosity,
            }
        )
        return resp.output_text

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
            "url": "/v1/responses",
            "body": {
                "model": self.model_name,
                "instructions": system_message,
                "input": [
                    {"role": "user", "content": user_prompt}
                ],
                "max_output_tokens": 4096,
                "reasoning": {
                    "effort": self.effort
                },
                "text": {
                    "verbosity": self.verbosity
                }
            }
        }
    
    def create_batch(self):
        batch_input_file = self.client.files.create(file=open(BATCH_INPUT_PATH, "rb"), purpose="batch")
        batch_input_file_id = batch_input_file.id
        
        batch = self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/responses",
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
                time.sleep(30)  # Poll every 30 seconds

    def process_result(self):
        with open(BATCH_OUTPUT_PATH, "r") as file:
            batch_outputs = [json.loads(line) for line in file]

            results = []
            for batch_output in batch_outputs:
                outputs = batch_output["response"]["body"]["output"]
                messages = [output for output in outputs if output['type'] == 'message']
                content = messages[0]['content'][0]['text'] if len(messages) else ''
                results += [content.strip()]

            return results