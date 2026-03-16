import os
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor

from src.prompts import RerankingPrompt, RetrievalRankingSingleBlock, RetrievalRankingMultipleBlocks


class LLMReranker:
    """Rerank retrieved documents using LLM."""
    
    def __init__(self):
        self.llm = self.set_up_llm()
        self.system_prompt_rerank_single_block = RerankingPrompt.system_prompt_rerank_single_block
        self.system_prompt_rerank_multiple_blocks = RerankingPrompt.system_prompt_rerank_multiple_blocks
        self.schema_for_single_block = RetrievalRankingSingleBlock
        self.schema_for_multiple_blocks = RetrievalRankingMultipleBlocks

    def set_up_llm(self):
        load_dotenv()
        llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return llm

    def get_rank_for_single_block(self, query, retrieved_document):
        """Get relevance score for single document."""
        user_prompt = f'Here is the query:\n"{query}"\n\nHere is the retrieved text block:\n"""\n{retrieved_document}\n"""'
        
        completion = self.llm.beta.chat.completions.parse(
            model="gpt-4o-mini-2024-07-18",
            temperature=0,
            messages=[
                {"role": "system", "content": self.system_prompt_rerank_single_block},
                {"role": "user", "content": user_prompt},
            ],
            response_format=self.schema_for_single_block
        )
        response = completion.choices[0].message.parsed
        return response.model_dump()

    def get_rank_for_multiple_blocks(self, query, retrieved_documents):
        """Rank multiple documents in single call."""
        formatted_blocks = "\n\n---\n\n".join(
            [f'Block {i+1}:\n\n"""\n{text}\n"""' for i, text in enumerate(retrieved_documents)]
        )
        user_prompt = (
            f'Here is the query: "{query}"\n\n'
            "Here are the retrieved text blocks:\n"
            f"{formatted_blocks}\n\n"
            f"You should provide exactly {len(retrieved_documents)} rankings, in order."
        )
        
        completion = self.llm.beta.chat.completions.parse(
            model="gpt-4o-mini-2024-07-18",
            temperature=0,
            messages=[
                {"role": "system", "content": self.system_prompt_rerank_multiple_blocks},
                {"role": "user", "content": user_prompt},
            ],
            response_format=self.schema_for_multiple_blocks
        )
        response = completion.choices[0].message.parsed
        return response.model_dump()

    def rerank_documents(self, query: str, documents: list, 
                         documents_batch_size: int = 4, 
                         llm_weight: float = 0.7):
        """
        Rerank documents using LLM.
        Combines vector similarity and LLM relevance scores.
        """
        doc_batches = [
            documents[i:i + documents_batch_size] 
            for i in range(0, len(documents), documents_batch_size)
        ]
        vector_weight = 1 - llm_weight
        
        if documents_batch_size == 1:
            def process_single_doc(doc):
                ranking = self.get_rank_for_single_block(query, doc['text'])
                doc_with_score = doc.copy()
                doc_with_score["relevance_score"] = ranking["relevance_score"]
                doc_with_score["combined_score"] = round(
                    llm_weight * ranking["relevance_score"] + vector_weight * doc['distance'],
                    4
                )
                return doc_with_score
            
            with ThreadPoolExecutor() as executor:
                all_results = list(executor.map(process_single_doc, documents))
        else:
            def process_batch(batch):
                texts = [doc['text'] for doc in batch]
                rankings = self.get_rank_for_multiple_blocks(query, texts)
                results = []
                block_rankings = rankings.get('block_rankings', [])
                
                # Handle missing rankings
                if len(block_rankings) < len(batch):
                    for i in range(len(block_rankings), len(batch)):
                        block_rankings.append({
                            "relevance_score": 0.0,
                            "reasoning": "Default ranking due to missing LLM response"
                        })
                
                for doc, rank in zip(batch, block_rankings):
                    doc_with_score = doc.copy()
                    doc_with_score["relevance_score"] = rank["relevance_score"]
                    doc_with_score["combined_score"] = round(
                        llm_weight * rank["relevance_score"] + vector_weight * doc['distance'],
                        4
                    )
                    results.append(doc_with_score)
                return results
            
            with ThreadPoolExecutor() as executor:
                batch_results = list(executor.map(process_batch, doc_batches))
            
            all_results = []
            for batch in batch_results:
                all_results.extend(batch)
        
        # Sort by combined score descending
        all_results.sort(key=lambda x: x["combined_score"], reverse=True)
        return all_results
