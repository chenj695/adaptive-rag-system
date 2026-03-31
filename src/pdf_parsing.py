"""
PDF Parsing module - uses PyPDF2 only (no Hugging Face downloads required)
"""
import json
import logging
import time
import hashlib
from pathlib import Path
from typing import List, Optional
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

# Only use PyPDF2 - no docling to avoid Hugging Face downloads
try:
    from PyPDF2 import PdfReader
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

_log = logging.getLogger(__name__)


def _process_chunk(chunk: List[Path], output_dir: Path, metadata_lookup: dict, debug_data_path: Path):
    """Process a chunk of PDFs in a separate process."""
    parser = PdfParser(
        doc_dir=None,
        output_dir=output_dir,
        metadata_lookup=metadata_lookup,
        debug_data_path=debug_data_path
    )
    for pdf_path in chunk:
        parser.parse_single_pdf(pdf_path)
    return f"Processed {len(chunk)} PDFs"


class PdfParser:
    """PDF parser using PyPDF2 - works offline without AI model downloads."""
    
    def __init__(self, doc_dir: Path, output_dir: Path, pdf_backend: str = None,
                 num_threads: int = 4, metadata_lookup: dict = None, 
                 debug_data_path: Path = None):
        if not PYPDF2_AVAILABLE:
            raise ImportError("PyPDF2 is required. Run: pip install PyPDF2")
        
        self.doc_dir = doc_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_threads = num_threads
        self.metadata_lookup = metadata_lookup or {}
        self.debug_data_path = debug_data_path
        
        _log.info("Using PyPDF2 parser (no external downloads required)")

    def parse_single_pdf(self, pdf_path: Path) -> dict:
        """Parse a single PDF file using PyPDF2."""
        _log.info(f"Processing {pdf_path.name}")
        
        reader = PdfReader(str(pdf_path))
        num_pages = len(reader.pages)
        
        # Generate sha1_name from filename
        sha1_name = hashlib.sha1(pdf_path.name.encode()).hexdigest()[:16]
        
        # Extract text from each page
        pages_content = []
        chunks = []
        
        for page_num, page in enumerate(reader.pages, 1):
            try:
                text = page.extract_text() or ""
                pages_content.append({
                    'page': page_num,
                    'content': [{
                        'type': 'text',
                        'text': text,
                        'bbox': {}
                    }],
                    'page_dimensions': {}
                })
                
                # Create chunks for embedding
                if text.strip():
                    # Split long text into smaller chunks
                    words = text.split()
                    chunk_size = 300
                    overlap = 50
                    
                    for i in range(0, len(words), chunk_size - overlap):
                        chunk_words = words[i:i + chunk_size]
                        chunk_text = ' '.join(chunk_words)
                        if chunk_text.strip():
                            chunks.append({
                                'page': page_num,
                                'text': chunk_text
                            })
                        
                        # Limit chunks per page to avoid too many
                        if len(chunks) > 100:
                            break
                            
            except Exception as e:
                _log.warning(f"Failed to extract page {page_num}: {e}")
                pages_content.append({
                    'page': page_num,
                    'content': [],
                    'page_dimensions': {}
                })
        
        assembled_report = {
            'metainfo': {
                'sha1_name': sha1_name,
                'filename': pdf_path.name,
                'pages_amount': num_pages,
                'text_blocks_amount': sum(len(p['content']) for p in pages_content),
                'tables_amount': 0,
                'pictures_amount': 0,
                'equations_amount': 0,
                'footnotes_amount': 0,
                'document_name': pdf_path.stem
            },
            'content': {
                'pages': pages_content,
                'chunks': chunks
            },
            'tables': [],
            'pictures': []
        }
        
        # Save output
        output_path = self.output_dir / f"{sha1_name}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(assembled_report, f, indent=2, ensure_ascii=False)
        
        return assembled_report

    def parse_pdfs_sequential(self):
        """Parse all PDFs sequentially."""
        if self.doc_dir is None:
            return
        pdf_files = list(self.doc_dir.glob("*.pdf"))
        for pdf_path in pdf_files:
            self.parse_single_pdf(pdf_path)

    def parse_pdfs_parallel(self, optimal_workers: int = 10, chunk_size: int = None):
        """Parse PDFs in parallel using multiple processes."""
        if self.doc_dir is None:
            return
            
        input_doc_paths = list(self.doc_dir.glob("*.pdf"))
        total_pdfs = len(input_doc_paths)
        
        if total_pdfs == 0:
            _log.warning("No PDF files found")
            return
            
        _log.info(f"Starting parallel processing of {total_pdfs} documents")
        
        cpu_count = multiprocessing.cpu_count()
        if optimal_workers is None:
            optimal_workers = min(cpu_count, total_pdfs)
        if chunk_size is None:
            chunk_size = max(1, total_pdfs // optimal_workers)
        
        chunks = [input_doc_paths[i : i + chunk_size] for i in range(0, total_pdfs, chunk_size)]
        start_time = time.time()
        processed_count = 0
        
        with ProcessPoolExecutor(max_workers=optimal_workers) as executor:
            futures = [
                executor.submit(
                    _process_chunk,
                    chunk,
                    self.output_dir,
                    self.metadata_lookup,
                    self.debug_data_path
                )
                for chunk in chunks
            ]
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    processed_count += int(result.split()[1])
                    _log.info(f"{'#'*50}\n{result} ({processed_count}/{total_pdfs} total)\n{'#'*50}")
                except Exception as e:
                    _log.error(f"Error processing chunk: {str(e)}")
                    raise
        
        elapsed_time = time.time() - start_time
        _log.info(f"Parallel processing completed in {elapsed_time:.2f} seconds.")


# Keep JsonReportProcessor for backward compatibility
class JsonReportProcessor:
    def __init__(self, metadata_lookup: dict = None, debug_data_path: Path = None):
        self.metadata_lookup = metadata_lookup or {}
        self.debug_data_path = debug_data_path

    def assemble_report(self, conv_result, normalized_data=None):
        """Assemble the report - PyPDF2 already returns assembled format."""
        return conv_result

    def debug_data(self, data):
        """Save debug data if path is set."""
        if self.debug_data_path is None:
            return
        doc_name = data.get('metainfo', {}).get('sha1_name', 'unknown')
        path = self.debug_data_path / f"{doc_name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
