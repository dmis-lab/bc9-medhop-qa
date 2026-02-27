from google import genai
from google.genai import types
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
import os

safety_settings = [
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=types.HarmBlockThreshold.BLOCK_NONE
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE
    ),
]

google_search_tool = Tool(
    google_search = GoogleSearch()
)

class Gemini():
    def __init__(self, model_name='gemini-2.5-flash-preview-05-20', use_google_search = True):
        self.model_name = model_name
        self.client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

        self.tools = []
        if use_google_search:
            self.tools = [google_search_tool]

    def generate_response_single(self, system_message, user_prompt):
        prompt = self._build_chat_prompt([
            {'role': 'system', 'content': system_message},
            {'role': 'user', 'content': user_prompt}
        ])

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    candidate_count=1,
                    max_output_tokens=8092,
                    temperature=1.0,
                    safety_settings=safety_settings,
                    tools=self.tools,
                    response_modalities=["TEXT"],
                )
            )

            return response.text if response.text else ('FAILED', [])
        except Exception as e:
            print('Error!', e)
            return ('ERROR', [])

    def _build_chat_prompt(self, messages: list[dict[str,str]]) -> str:
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)