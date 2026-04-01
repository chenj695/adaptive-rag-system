import json
import tiktoken
from pathlib import Path
from typing import List, Dict, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter


class TextSplitter:
    """Split reports into chunks for vectorization."""
    
    def count_tokens(self, string: str, encoding_name="o200k_base") -> int:
        """Count tokens using tiktoken."""
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(string))
    
    def _get_serialized_tables_by_page(self, tables: List[Dict]) -> Dict[int, List[Dict]]:
        """Group serialized tables by page number."""
        tables_by_page = {}
        for table in tables:
            if 'serialized' not in table:
                continue
            page = table['page']
            if page not in tables_by_page:
                tables_by_page[page] = []
            
            table_text = "\n".join(
                block["information_block"]
                for block in table["serialized"]["information_blocks"]
            )
            tables_by_page[page].append({
                "page": page,
                "text": table_text,
                "table_id": table["table_id"],
                "length_tokens": self.count_tokens(table_text)
            })
        return tables_by_page
    
    def _split_page(self, page: Dict[str, any], 
                    chunk_size: int = 300, 
                    chunk_overlap: int = 50) -> List[Dict[str, any]]:
        """Split page text into chunks."""
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4o",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        chunks = text_splitter.split_text(page['text'])
        chunks_with_meta = []
        
        for chunk in chunks:
            chunks_with_meta.append({
                "page": page['page'],
                "length_tokens": self.count_tokens(chunk),
                "text": chunk
            })
        
        return chunks_with_meta
    
    def _split_report(self, file_content: Dict[str, any],
                      serialized_tables_report_path: Optional[Path] = None) -> Dict[str, any]:
        """Split entire report into chunks."""
        chunks = []
        chunk_id = 0
        tables_by_page = {}
        
        if serialized_tables_report_path is not None:
            with open(serialized_tables_report_path, 'r', encoding='utf-8') as f:
                parsed_report = json.load(f)
            tables_by_page = self._get_serialized_tables_by_page(parsed_report.get('tables', []))
        
        # Handle both formats
        content = file_content.get('content', {})
        if isinstance(content, dict):
            pages = content.get('pages', [])
        else:
            pages = content
            
        for page in pages:
            page_chunks = self._split_page(page)
            for chunk in page_chunks:
                chunk['id'] = chunk_id
                chunk['type'] = 'content'
                chunk_id += 1
                chunks.append(chunk)
            
            # Add serialized table chunks if available
            if tables_by_page and page['page'] in tables_by_page:
                for table in tables_by_page[page['page']]:
                    table['id'] = chunk_id
                    table['type'] = 'serialized_table'
                    chunk_id += 1
                    chunks.append(table)
        
        # Ensure content is a dict with chunks
        if isinstance(file_content.get('content'), dict):
            file_content['content']['chunks'] = chunks
        else:
            file_content['content'] = {'pages': pages, 'chunks': chunks}
        
        return file_content
    
    def split_all_reports(self, all_report_dir: Path, output_dir: Path, 
                          serialized_tables_dir: Optional[Path] = None):
        """Split all reports in directory."""
        all_report_paths = list(all_report_dir.glob("*.json"))
        
        for report_path in all_report_paths:
            serialized_tables_path = None
            if serialized_tables_dir is not None:
                serialized_tables_path = serialized_tables_dir / report_path.name
                if not serialized_tables_path.exists():
                    print(f"Warning: Could not find serialized tables for {report_path.name}")
            
            with open(report_path, 'r', encoding='utf-8') as file:
                report_data = json.load(file)
            
            updated_report = self._split_report(report_data, serialized_tables_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(output_dir / report_path.name, 'w', encoding='utf-8') as file:
                json.dump(updated_report, file, indent=2, ensure_ascii=False)
        
        print(f"Split {len(all_report_paths)} files")
