import inspect
import re
from typing import List, Literal, Union

from pydantic import BaseModel, Field


def build_system_prompt(instruction: str, example: str, pydantic_schema: str = ""):
    base_prompt = f"""You are a helpful assistant.

{instruction}

{example}"""
    if pydantic_schema:
        base_prompt += f"""

You must format your output as a JSON value that adheres to the following schema:
        
{pydantic_schema}"""
    return base_prompt


class NumberSchemaPrompt:
    instruction = """
Your task is to answer the question based on the provided context.
Follow these steps:
1. Carefully analyze the question to understand what is being asked. Identify key entities, numbers, and relationships mentioned.
2. Read through the context thoroughly. Look for information relevant to the question.
3. If numeric data is needed, extract the exact values from the context. Perform any necessary calculations (currency conversion, summing values, etc.).
4. Formulate your final answer based on the evidence found in the context.

Answer rules:
- If the answer is a number, use ONLY digits and decimal points (e.g., 1000000 or 1000000.50)
- For negative numbers, use a minus sign (e.g., -1000000)
- If the question asks for a count/quantity, return just the number
- If information is insufficient or not found in the context, return "N/A"
- Do not include currency symbols, units, commas, or any text formatting
"""
    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(description="Detailed step-by-step analysis of the answer with at least 5 steps and at least 150 words.")
        reasoning_summary: str = Field(description="Concise summary of the step-by-step reasoning process. Around 50 words.")
        relevant_pages: List[int] = Field(description="List of page numbers where relevant information was found")
        final_answer: Union[float, Literal["N/A"]] = Field(description="Final numeric answer or 'N/A' if not available")
    
    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)
    example = """
Example:
Question: "What was the revenue of the company in 2023?"
Context: "The company reported revenue of $1,000,000 for fiscal year 2023."
Answer:
```
{
  "step_by_step_analysis": "1. The question asks for revenue in 2023.\\n2. Found revenue figure in context.\\n3. Value is $1,000,000.\\n4. Removed currency symbol and commas.\\n5. Final answer is 1000000",
  "reasoning_summary": "Revenue of $1,000,000 was reported for 2023.",
  "relevant_pages": [5],
  "final_answer": 1000000
}
```"""
    system_prompt = build_system_prompt(instruction, example)
    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)


class BooleanSchemaPrompt:
    instruction = """
Your task is to answer yes/no questions based on the provided context.
Follow these steps:
1. Carefully analyze the question to understand what fact or event is being asked about.
2. Read through the context thoroughly looking for direct or indirect evidence.
3. If the answer is clearly supported by the context, answer "Yes" or "No".
4. If the information is not found or unclear, answer "N/A".

Answer rules:
- Answer must be exactly "Yes", "No", or "N/A"
- Base your answer only on the provided context
"""
    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(description="Detailed step-by-step analysis with at least 5 steps and 150 words.")
        reasoning_summary: str = Field(description="Concise summary around 50 words.")
        relevant_pages: List[int] = Field(description="Pages where relevant information was found")
        final_answer: Union[Literal["Yes", "No"], Literal["N/A"]]
    
    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)
    example = """
Example:
Question: "Did the company announce any acquisitions?"
Context: "In Q3, the company completed the acquisition of TechCorp for $50M."
Answer:
```
{
  "step_by_step_analysis": "1. The question asks about acquisitions.\\n2. Found mention of acquisition in context.\\n3. TechCorp was acquired for $50M in Q3.\\n4. This confirms acquisition occurred.\\n5. Answer is Yes",
  "reasoning_summary": "The company acquired TechCorp in Q3.",
  "relevant_pages": [12],
  "final_answer": "Yes"
}
```"""
    system_prompt = build_system_prompt(instruction, example)
    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)


class NameSchemaPrompt:
    instruction = """
Your task is to extract company names or identify specific named entities from the context.
Follow these steps:
1. Identify what company name or entity is being asked for.
2. Search the context for the exact name as it appears in the document.
3. Extract the name exactly as written (preserve spelling, capitalization, punctuation).
4. If not found, return "N/A".

Answer rules:
- Return the exact name as it appears in the context
- Do not modify capitalization or spelling
- If multiple companies are mentioned, return the one that answers the question
"""
    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(description="Detailed step-by-step analysis with at least 5 steps and 150 words.")
        reasoning_summary: str = Field(description="Concise summary around 50 words.")
        relevant_pages: List[int] = Field(description="Pages where relevant information was found")
        final_answer: Union[str, Literal["N/A"]]
    
    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)
    example = """
Example:
Question: "Which subsidiary was sold in 2023?"
Context: "The company sold its subsidiary TechSolutions Inc. in March 2023."
Answer:
```
{
  "step_by_step_analysis": "1. The question asks about a sold subsidiary.\\n2. Found mention of subsidiary sale in context.\\n3. Subsidiary name is 'TechSolutions Inc.'\\n4. Sale occurred in March 2023.\\n5. Answer is TechSolutions Inc.",
  "reasoning_summary": "TechSolutions Inc. was sold in March 2023.",
  "relevant_pages": [8],
  "final_answer": "TechSolutions Inc."
}
```"""
    system_prompt = build_system_prompt(instruction, example)
    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)


class ExplanationSchemaPrompt:
    """Open-ended explanation: how / what / why / describe / explain questions."""

    instruction = """
Your task is to answer the question using clear, accurate prose based only on the provided context.
Follow these steps:
1. Identify what the question is asking (definition, mechanism, comparison, cause-effect, etc.).
2. Read the context and extract only information that directly supports an answer.
3. Synthesize a coherent explanation; use the document's terminology when it matches the question.
4. If the context does not contain enough information, say so honestly.

Answer rules:
- Put the direct, user-facing answer in final_answer as one or more short paragraphs (plain text, not JSON inside the string).
- Use step_by_step_analysis for structured reasoning (cite pages and quotes where helpful).
- Do not invent facts beyond the context.
- If the context contains ANY relevant definitions, mechanisms, or facts (even partial), you MUST synthesize final_answer from that material. Use "N/A" only when the context truly has zero usable information for the question.
- Always return valid JSON with all four keys populated whenever possible; do not leave reasoning_summary or step_by_step_analysis empty if you provide an answer.
"""
    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(
            description="Detailed step-by-step analysis with at least 5 steps and at least 150 words."
        )
        reasoning_summary: str = Field(description="Concise summary of the reasoning, around 50 words.")
        relevant_pages: List[int] = Field(description="Page numbers where the most relevant information was found.")
        final_answer: Union[str, Literal["N/A"]] = Field(
            description="Complete explanatory answer in prose, or 'N/A' if the context does not support an answer."
        )

    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)
    example = """
Example:
Question: "What are adiabatic temperature changes?"
Context: "Such temperature changes are called adiabatic changes because no heat is added to, or withdrawn from, the air."
Answer:
```
{
  "step_by_step_analysis": "1. The question asks for a definition of adiabatic temperature changes.\\n2. The context states that these are temperature changes with no heat exchange with the surroundings.\\n3. The passage names them 'adiabatic changes' explicitly.\\n4. I synthesize this into a short definition for the user.\\n5. No contradictory information appears in the excerpt.",
  "reasoning_summary": "Adiabatic changes are temperature changes in air without heat added or removed.",
  "relevant_pages": [61],
  "final_answer": "Adiabatic temperature changes are changes in air temperature that occur without heat being added to or removed from the air parcel (e.g., from expansion or compression as the air moves)."
}
```"""
    system_prompt = build_system_prompt(instruction, example)
    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)


class AnswerSchemaFixPrompt:
    system_prompt = """You are a JSON formatter.
Your task is to format raw LLM response into a valid JSON object.
Your answer should always start with '{' and end with '}'
Your answer should contain only json string, without any preambles, comments, or triple backticks."""

    @staticmethod
    def format_user_prompt(system_prompt_content: str, response_content: str) -> str:
        return f"""Here is the system prompt that defines schema of the json object and provides an example of answer with valid schema:

{system_prompt_content}

---

Here is the LLM response that not following the schema and needs to be properly formatted:

{response_content}"""


class RerankingPrompt:
    system_prompt_rerank_single_block = """
You are a RAG (Retrieval-Augmented Generation) retrievals ranker.
You will receive a query and retrieved text block related to that query. Your task is to evaluate and score the block based on its relevance to the query provided.
Instructions:
1. Reasoning:
   Analyze the block by identifying key information and how it relates to the query. Consider whether the block provides direct answers, partial insights, or background context relevant to the query. Explain your reasoning in a few sentences, referencing specific elements of the block to justify your evaluation. Avoid assumptions—focus solely on the content provided.
2. Relevance Score (0 to 1, in increments of 0.1):
   0 = Completely Irrelevant: The block has no connection or relation to the query.
   0.1 = Virtually Irrelevant: Only a very slight or vague connection to the query.
   0.2 = Very Slightly Relevant: Contains an extremely minimal or tangential connection.
   0.3 = Slightly Relevant: Addresses a very small aspect of the query but lacks substantive detail.
   0.4 = Somewhat Relevant: Contains partial information that is somewhat related but not comprehensive.
   0.5 = Moderately Relevant: Addresses the query but with limited or partial relevance.
   0.6 = Fairly Relevant: Provides relevant information, though lacking depth or specificity.
   0.7 = Relevant: Clearly relates to the query, offering substantive but not fully comprehensive information.
   0.8 = Very Relevant: Strongly relates to the query and provides significant information.
   0.9 = Highly Relevant: Almost completely answers the query with detailed and specific information.
   1 = Perfectly Relevant: Directly and comprehensively answers the query with all the necessary specific information.
3. Additional Guidance:
   - Objectivity: Evaluate block based only on their content relative to the query.
   - Clarity: Be clear and concise in your justifications.
   - No assumptions: Do not infer information beyond what's explicitly stated in the block.
"""
    system_prompt_rerank_multiple_blocks = """
You are a RAG (Retrieval-Augmented Generation) retrievals ranker.
You will receive a query and several retrieved text blocks related to that query. Your task is to evaluate and score each block based on its relevance to the query provided.
Instructions:
1. Reasoning:
   Analyze the block by identifying key information and how it relates to the query. Consider whether the block provides direct answers, partial insights, or background context relevant to the query. Explain your reasoning in a few sentences, referencing specific elements of the block to justify your evaluation. Avoid assumptions—focus solely on the content provided.
2. Relevance Score (0 to 1, in increments of 0.1):
   0 = Completely Irrelevant: The block has no connection or relation to the query.
   0.1 = Virtually Irrelevant: Only a very slight or vague connection to the query.
   0.2 = Very Slightly Relevant: Contains an extremely minimal or tangential connection.
   0.3 = Slightly Relevant: Addresses a very small aspect of the query but lacks substantive detail.
   0.4 = Somewhat Relevant: Contains partial information that is somewhat related but not comprehensive.
   0.5 = Moderately Relevant: Addresses the query but with limited or partial relevance.
   0.6 = Fairly Relevant: Provides relevant information, though lacking depth or specificity.
   0.7 = Relevant: Clearly relates to the query, offering substantive but not fully comprehensive information.
   0.8 = Very Relevant: Strongly relates to the query and provides significant information.
   0.9 = Highly Relevant: Almost completely answers the query with detailed and specific information.
   1 = Perfectly Relevant: Directly and comprehensively answers the query with all the necessary specific information.
3. Additional Guidance:
   - Objectivity: Evaluate blocks based only on their content relative to the query.
   - Clarity: Be clear and concise in your justifications.
   - No assumptions: Do not infer information beyond what's explicitly stated in the block.
"""


class RetrievalRankingSingleBlock(BaseModel):
    reasoning: str = Field(description="Analysis of the block, identifying key information and how it relates to the query")
    relevance_score: float = Field(description="Relevance score from 0 to 1, where 0 is Completely Irrelevant and 1 is Perfectly Relevant")


class RetrievalRankingMultipleBlocks(BaseModel):
    block_rankings: List[RetrievalRankingSingleBlock] = Field(
        description="A list of text blocks and their associated relevance scores."
    )
