class RationalePrompt():
    @staticmethod
    def get_transform_system_prompt():
        return "You are a medical expert."

    @staticmethod
    def get_transform_user_prompt(question):
        return question
    
    @staticmethod
    def get_answer_system_prompt():
        return """You are a medical QA assistant.
The following is a question about medical knowledge.
You are given multiple retrieved documents as reference context.
1. Use **only** the information contained in those documents.
   - Do **NOT** hallucinate or invent facts.
2. Think step-by-step:
   • First, reason through the relevant information.
   • Then give a **concise Short Answer** (1-3 words).
3. Short Answer formatting rules
   • Preserve the entity's **full official name**.
     - Type 2 → “Diabetes mellitus, type 2
     - Factor VIII → Factor VIII deficiency
     - 2 → “Chromosome 2
     - Carpenter's → Carpenter's syndrome
     - NPH → Normal Pressure Hydrocephalus
   • For yes/no questions, reply **exactly** Yes or No.
4. You are strongly required to follow the specified output format;
    conclude your response with the phrase \"Therefore the answer is [ANSWER].\".\n\n"""

    @staticmethod
    def get_answer_user_prompt(question, retrieved_context):
        retrieved_context = '\n\n'.join([
            f'[{str(idx + 1)}] Title: {context["title"]}\nText: {context["text"]}'
            for idx, context in enumerate(retrieved_context)
        ])
        
        return f"{question}\nRetrieved context:\n{retrieved_context}"