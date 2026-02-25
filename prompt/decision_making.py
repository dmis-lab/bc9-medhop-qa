class DecisionMakingPrompt():
    @staticmethod
    def get_answer_system_prompt():
        return """You are an expert decision-making assistant.

Please carefully analyze these three answers and choose the **best** one. Follow these steps:

1. **Identify the key decision criteria** (e.g., accuracy, relevance to the question, logical consistency, completeness).  
2. **Compare each candidate answer** step-by-step with respect to these criteria. Clearly show your reasoning process.  
3. **Decide whether any of the three answers is appropriate and accurate enough to be selected.**  
   - If **one** candidate is suitable, select it and explain why it is better than the other two.  
   - If **none** of the candidates are suitable, provide your own short and long answers that directly address the question, and explain why the candidates were not appropriate.  
4. **Output the final decision** as the final selected long answer or your own generated long answer.

When you output an answer, please follow these rules:

- Show your full reasoning process as a step-by-step explanation before providing the final decision.
- Short Answer formatting rules
   • Preserve the entity's **full official name**.
     - Type 2 → “Diabetes mellitus, type 2
     - Factor VIII → Factor VIII deficiency
     - 2 → “Chromosome 2
     - Carpenter's → Carpenter's syndrome
     - NPH → Normal Pressure Hydrocephalus
   • For yes/no questions, reply **exactly** Yes or No.
- If you generate your own answers, do so with the same thoroughness and format.
- You are strongly required to follow the specified output format; conclude your response with the phrase \"Therefore the answer is [ANSWER].\"
"""

    @staticmethod
    def get_answer_user_prompt(row):
        return f"""
Here is the original question:
{row['Question']}

Below are three candidate answers, each with a short and long version:

1. Candidate A:
- Short Answer: {row['web_short']}
- Long Answer: {row['web_long']}

2. Candidate B:
- Short Answer: {row['rationale_short']}
- Long Answer: {row['rationale_long']}

3. Candidate C:
- Short Answer: {row['q2d_short']}
- Long Answer: {row['q2d_long']}

"""