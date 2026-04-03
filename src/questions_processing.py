import json
import os
import re
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict, Optional, Literal
from dotenv import load_dotenv
from pydantic import BaseModel
from tqdm import tqdm

from src.retrieval import HybridRetriever
from src.multi_path_retrieval import MultiPathRetriever, MultiPathRetrieverWithRAPTOR
from src.prompts import (
    NumberSchemaPrompt, BooleanSchemaPrompt, NameSchemaPrompt,
    AnswerSchemaFixPrompt
)
from src.github_models_client import UnifiedLLMClient


class OpenAIProcessor:
    """Handles all LLM interactions - supports OpenAI and GitHub Models."""
    
    def __init__(self, api_provider: str = "openai", model: str = "o3-mini-2025-01-31"):
        load_dotenv()
        self.api_provider = api_provider
        self.model = model
        self.client = UnifiedLLMClient(api_provider=api_provider, model=model)
        self.response_data = {}

    def get_answer_from_rag_context(self, question: str, rag_context: any, 
                                    schema: str, model: str) -> dict:
        """Generate answer using retrieved context."""
        schema_prompts = {
            "number": NumberSchemaPrompt(),
            "boolean": BooleanSchemaPrompt(),
            "name": NameSchemaPrompt(),
            "text": NameSchemaPrompt()  # Default to name schema for general text
        }
        prompt_obj = schema_prompts.get(schema, NumberSchemaPrompt())
        
        # Format context
        if isinstance(rag_context, list):
            context_text = "\n\n---\n\n".join([
                f"Page {doc.get('page', 'N/A')}:\n{doc.get('text', '')}"
                for doc in rag_context
            ])
        else:
            context_text = str(rag_context)
        
        messages = [
            {"role": "system", "content": prompt_obj.system_prompt_with_schema},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"}
        ]
        
        try:
            # Use unified client for both OpenAI and GitHub Models
            parsed = self.client.parse_structured_output(
                messages=messages,
                response_schema=prompt_obj.AnswerSchema,
                temperature=0.3
            )
            self.response_data = {
                "model": self.client.model,
                "provider": self.api_provider
            }
            return parsed
        except Exception as e:
            # Try to fix malformed response
            try:
                raw_response = self.client.get_completion(
                    messages=messages,
                    temperature=0.3
                )
                return self.fix_answer_schema(raw_response, prompt_obj.system_prompt_with_schema)
            except:
                return {"final_answer": "N/A", "error": str(e), "step_by_step_analysis": "", "reasoning_summary": ""}

    def fix_answer_schema(self, response: str, system_prompt: str) -> dict:
        """Fix malformed JSON responses."""
        fix_prompt = AnswerSchemaFixPrompt.user_prompt.format(
            system_prompt=system_prompt,
            response=response
        )
        try:
            messages = [
                {"role": "system", "content": AnswerSchemaFixPrompt.system_prompt},
                {"role": "user", "content": fix_prompt}
            ]
            # Use unified client
            content = self.client.get_completion(
                messages=messages,
                temperature=0.3
            )
            return json.loads(content)
        except:
            return {"final_answer": "N/A", "error": "Failed to parse", "step_by_step_analysis": "", "reasoning_summary": ""}


class QuestionsProcessor:
    """Main question processing orchestrator."""
    
    def __init__(self, questions_file: Optional[Path], vector_db_dir: Path,
                 documents_dir: Path, chunked_reports_dir: Path = None, markdown_reports_dir: Path = None,
                 run_config=None, bm25_dir: Path = None):
        self.questions_file = questions_file
        self.questions = self._load_questions(questions_file) if questions_file else []
        self.openai_processor = OpenAIProcessor(
            api_provider=run_config.api_provider if run_config else "openai",
            model=run_config.answering_model if run_config else "o3-mini-2025-01-31"
        )
        
        # Use multi-path retriever if enabled, otherwise use standard HybridRetriever
        if run_config and run_config.use_multi_path:
            from pathlib import Path as PathType
            bm25_path = bm25_dir or (PathType(vector_db_dir).parent / "bm25_indices")
            self.retriever = MultiPathRetriever(
                vector_db_dir=vector_db_dir,
                bm25_dir=bm25_path,
                documents_dir=documents_dir,
                rrf_k=run_config.rrf_k
            )
            self.use_multi_path = True
        else:
            chunked_path = chunked_reports_dir or (Path(vector_db_dir).parent / "chunked_reports")
            self.retriever = HybridRetriever(vector_db_dir, chunked_path, documents_dir)
            self.use_multi_path = False
        
        self.markdown_reports_dir = markdown_reports_dir
        self.run_config = run_config
        self.answer_details = []

    def _load_questions(self, questions_file: Path) -> List[Dict]:
        """Load questions from JSON file."""
        with open(questions_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _determine_schema(self, question_text: str) -> str:
        """Determine answer schema type from question."""
        question_lower = question_text.lower()
        
        # Check for boolean indicators
        boolean_starters = ['did ', 'was ', 'were ', 'have ', 'has ', 'is ', 'are ', 'do ', 'does ', 'can ', 'could ']
        if any(question_lower.startswith(starter) for starter in boolean_starters):
            return "boolean"
        
        # Check for name/entity questions
        if any(phrase in question_lower for phrase in ['what is the name', 'which ', 'who ', 'what was the name']):
            return "name"
        
        # Check for number indicators
        number_indicators = ['how much', 'how many', 'what is the amount', 'what was the amount', 
                           'what is the value', 'what was the value', 'what is the number',
                           'what percentage', 'what is the total', 'what was the total']
        if any(phrase in question_lower for phrase in number_indicators):
            return "number"
        
        return "text"  # Default to text

    def get_answer_for_document(self, sha1_name: str, question: str, 
                                 document_name: str = None) -> dict:
        """Get answer for a single document query."""
        schema = self._determine_schema(question)
        
        # Retrieve context
        if self.run_config and self.run_config.full_context and self.markdown_reports_dir:
            rag_context = self.retriever.vector_retriever.retrieve_all_pages(sha1_name)
        elif self.use_multi_path:
            # Use multi-path retrieval
            rag_context = self.retriever.retrieve_by_document(
                sha1_name=sha1_name,
                query=question,
                semantic_top_n=self.run_config.semantic_top_n if self.run_config else 20,
                lexical_top_n=self.run_config.lexical_top_n if self.run_config else 20,
                fusion_top_k=self.run_config.fusion_top_k if self.run_config else 20,
                use_reranking=self.run_config.llm_reranking if self.run_config else False,
                llm_reranking_sample_size=self.run_config.llm_reranking_sample_size if self.run_config else 28,
                top_n=self.run_config.top_n_retrieval if self.run_config else 6
            )
        else:
            # Use standard hybrid retrieval
            rag_context = self.retriever.retrieve_by_document(
                sha1_name=sha1_name,
                query=question,
                llm_reranking_sample_size=self.run_config.llm_reranking_sample_size if self.run_config else 28,
                top_n=self.run_config.top_n_retrieval if self.run_config else 6,
                llm_weight=0.7,
                return_parent_pages=self.run_config.parent_document_retrieval if self.run_config else False
            )
        
        # Generate answer
        answer = self.openai_processor.get_answer_from_rag_context(
            question=question,
            rag_context=rag_context,
            schema=schema,
            model=self.run_config.answering_model if self.run_config else "o3-mini-2025-01-31"
        )
        
        # Add metadata
        answer["schema"] = schema
        answer["document_name"] = document_name or sha1_name
        answer["retrieved_context"] = rag_context[:3] if isinstance(rag_context, list) else []
        if self.use_multi_path:
            answer["retrieval_method"] = "multi_path"
        
        return answer

    def query_single(self, question: str, sha1_name: Optional[str] = None) -> dict:
        """Query a single question (for WebUI)."""
        if sha1_name:
            # Query specific document
            return self.get_answer_for_document(sha1_name, question)
        else:
            # Query all documents and return best answer
            all_docs = self.retriever.get_all_documents()
            if not all_docs:
                return {"final_answer": "N/A", "error": "No documents available"}
            
            # For now, query the first document
            # In a more advanced version, could query all and pick best
            return self.get_answer_for_document(all_docs[0]["sha1_name"], question, all_docs[0].get("document_name"))

    def process_all_questions(self, output_path: str = 'questions_with_answers.json',
                              team_email: str = "rag@system.local", 
                              submission_name: str = "RAG System", 
                              submission_file: bool = False, 
                              pipeline_details: str = ""):
        """Process all questions."""
        processed_questions = []
        
        for question_data in tqdm(self.questions, desc="Processing questions"):
            question_text = question_data["text"]
            schema = question_data.get("kind", self._determine_schema(question_text))
            
            # Get document name from question or use first available
            sha1_name = question_data.get("sha1_name")
            if not sha1_name:
                all_docs = self.retriever.get_all_documents()
                if all_docs:
                    sha1_name = all_docs[0]["sha1_name"]
            
            if sha1_name:
                answer = self.get_answer_for_document(sha1_name, question_text)
            else:
                answer = {"final_answer": "N/A", "error": "No document specified"}
            
            processed_question = {
                "question": question_text,
                "answer": answer.get("final_answer", "N/A"),
                "full_response": answer,
                "schema": schema
            }
            processed_questions.append(processed_question)
        
        # Save results
        result = {
            "questions": processed_questions,
            "statistics": {
                "total": len(processed_questions),
                "answered": sum(1 for q in processed_questions if q["answer"] != "N/A")
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return result
