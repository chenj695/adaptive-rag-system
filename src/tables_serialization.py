import asyncio
import json
import os
import threading
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm
import tiktoken

logger = logging.getLogger(__name__)


class TableSerialization:
    """Pydantic schemas for table serialization."""
    
    system_prompt = (
        "You are a table serialization agent.\n"
        "Your task is to create a set of contextually independent blocks of information based on the provided table and surrounding text.\n"
        "These blocks must be totally context-independent because they will be used as separate chunk to populate database."
    )
    
    class SerializedInformationBlock(BaseModel):
        subject_core_entity: str = Field(
            description="A primary focus of what this block is about. Usually located in a row header."
        )
        information_block: str = Field(description=(
            "Detailed information about the chosen core subject from tables and additional texts. Information SHOULD include:\n"
            "1. All related header information\n"
            "2. All related units and their descriptions\n"
            "    2.1. If header is Total, always write additional context about what this total represents!\n"
            "3. All additional info for context enrichment to make ensure complete context-independency.\n"
            "SKIPPING ANY VALUABLE INFORMATION WILL BE HEAVILY PENALIZED!"
        ))
    
    class TableBlocksCollection(BaseModel):
        subject_core_entities_list: List[str] = Field(
            description="A complete list of core entities. Keep in mind, empty headers are possible."
        )
        relevant_headers_list: List[str] = Field(
            description="A list of ALL headers relevant to the subject."
        )
        information_blocks: List["TableSerialization.SerializedInformationBlock"] = Field(
            description="Complete list of fully described context-independent information blocks"
        )


class AsyncTableSerializer:
    """Async table serialization using OpenAI API."""
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.encoding = tiktoken.get_encoding("o200k_base")
        self.logger = logging.getLogger(__name__)

    async def async_serialize_tables(self, json_report: Dict,
                                      requests_filepath: str,
                                      results_filepath: str) -> Dict:
        """Serialize all tables in a report asynchronously."""
        tables = json_report.get("tables", [])
        
        for i, table in enumerate(tables):
            if "markdown" not in table:
                continue
            
            serialized = await self._serialize_single_table(table, json_report.get("metainfo", {}).get("sha1_name", ""))
            if serialized:
                json_report["tables"][i]["serialized"] = serialized
        
        return json_report

    async def _serialize_single_table(self, table: Dict, report_name: str) -> Dict:
        """Convert single table to serialized blocks."""
        try:
            completion = await self.client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": TableSerialization.system_prompt},
                    {"role": "user", "content": f"Table markdown:\n{table.get('markdown', '')}"}
                ],
                response_format=TableSerialization.TableBlocksCollection
            )
            
            result = completion.choices[0].message.parsed
            return result.model_dump()
        except Exception as e:
            self.logger.error(f"Error serializing table in {report_name}: {e}")
            return None

    def process_file(self, json_path: Path) -> None:
        """Process single report file."""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_report = json.load(f)
            
            thread_id = threading.get_ident()
            requests_filepath = f'./temp/async_llm_requests_{thread_id}.jsonl'
            results_filepath = f'./temp/async_llm_results_{thread_id}.jsonl'
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                updated_report = loop.run_until_complete(
                    self.async_serialize_tables(
                        json_report,
                        requests_filepath=requests_filepath,
                        results_filepath=results_filepath
                    )
                )
            finally:
                loop.close()
                # Cleanup temp files
                try:
                    os.remove(requests_filepath)
                    os.remove(results_filepath)
                except FileNotFoundError:
                    pass
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(updated_report, f, indent=2, ensure_ascii=False)
                
        except json.JSONDecodeError as e:
            self.logger.error("JSON Error in %s: %s", json_path.name, str(e))
            raise
        except Exception as e:
            self.logger.error("Error processing %s: %s", json_path.name, str(e))
            raise

    def process_directory_parallel(self, input_dir: Path, max_workers: int = 5):
        """Process all JSON files in parallel."""
        self.logger.info("Starting parallel table serialization...")
        json_files = list(input_dir.glob("*.json"))
        
        if not json_files:
            self.logger.warning("No JSON files found in %s", input_dir)
            return
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            with tqdm(total=len(json_files), desc="Processing files") as pbar:
                futures = []
                for json_file in json_files:
                    future = executor.submit(self.process_file, json_file)
                    future.add_done_callback(lambda p: pbar.update(1))
                    futures.append(future)
                
                # Wait for completion
                while futures:
                    done_futures = []
                    for future in futures:
                        if future.done():
                            done_futures.append(future)
                            try:
                                future.result()
                            except Exception as e:
                                self.logger.error(str(e))
                    for future in done_futures:
                        futures.remove(future)
                    time.sleep(0.1)
        
        self.logger.info("Table serialization completed!")
