class Query2DocPrompt():
    @staticmethod
    def get_transform_system_prompt():
        return """Write a medical passage that can help answer the given query. 
Include key information or terminology for the answer.

Example:

Query: A 39-year-old woman presents to the family medicine clinic to be evaluated by her physician for weight gain. ...  Which of the following recommendations may be appropriate for the patient at this time? A) Hepatitis B vaccination B) Low-dose chest CT C) Hepatitis C vaccination D) Shingles vaccination
Passage: Against vaccine-preventable diseases. Every visit by an adult to a healthcare provider should be an opportunity to provide this protection. ...

Query: A 23-year-old male presents to his primary care physician after an injury during a rugby game. ... Which of the following is the most likely diagnosis? A) Medial collateral ligament tear B) Lateral collateral ligament tear C) Anterior cruciate ligament tear D) Posterior cruciate ligament tear
Passage: Diagnosing PCL Injuries: History, Physical Examination, Imaging Studies, Arthroscopic Evaluation. ...

Query: A 45-year-old woman is in a high-speed motor vehicle accident and suffers multiple injuries ... Which of the following is an appropriate next step? A) Provide transfusions as needed B) Withhold transfusion based on husband's request C) Obtain an ethics consult D) Obtain a court order for transfusion
Passage: Legal and ethical issues in safe blood transfusion. ...

Query: A 4-year-old male is accompanied by his mother to the pediatrician. ... Which of the following would most likely be found on biopsy of this patient's kidney? A) Mononuclear and eosinophilic infiltrate B) Replacement of renal parenchyma with foamy histiocytes C) Destruction of the proximal tubule and medullary thick ascending limb D) Tubular colloid casts with diffuse lymphoplasmacytic infiltrate
Passage: The natural history of urinary infection in adults. ...
"""

    @staticmethod
    def get_transform_user_prompt(question):
        return f"""Query: {question}
Passage:"""
    
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