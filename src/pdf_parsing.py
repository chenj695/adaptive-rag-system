import json
import logging
import time
from pathlib import Path
from typing import List, Optional
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.settings import settings
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from tabulate import tabulate
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

_log = logging.getLogger(__name__)


def _process_chunk(chunk: List[Path], pdf_backend, output_dir: Path, 
                   num_threads: int, metadata_lookup: dict, debug_data_path: Path):
    """Process a chunk of PDFs in a separate process."""
    parser = PdfParser(
        doc_dir=None,
        output_dir=output_dir,
        pdf_backend=pdf_backend,
        num_threads=num_threads,
        metadata_lookup=metadata_lookup,
        debug_data_path=debug_data_path
    )
    for pdf_path in chunk:
        parser.parse_single_pdf(pdf_path)
    return f"Processed {len(chunk)} PDFs"


class PdfParser:
    def __init__(self, doc_dir: Path, output_dir: Path, pdf_backend: str = "pypdfium2",
                 num_threads: int = 4, metadata_lookup: dict = None, 
                 debug_data_path: Path = None):
        self.doc_dir = doc_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_backend = pdf_backend
        self.num_threads = num_threads
        self.metadata_lookup = metadata_lookup or {}
        self.debug_data_path = debug_data_path
        
        # Configure docling settings
        settings.perf.doc_batch_size = 1
        settings.perf.page_batch_size = 1
        
        # Initialize converter with proper backend
        # Use PyPdfiumDocumentBackend class directly instead of string
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=None,
                    backend=PyPdfiumDocumentBackend
                )
            }
        )
        
        self.json_processor = JsonReportProcessor(
            metadata_lookup=metadata_lookup,
            debug_data_path=debug_data_path
        )

    def parse_single_pdf(self, pdf_path: Path) -> ConversionResult:
        """Parse a single PDF file."""
        _log.info(f"Processing {pdf_path.name}")
        result = self.converter.convert(pdf_path)
        
        assembled_report = self.json_processor.assemble_report(result)
        
        output_path = self.output_dir / f"{assembled_report['metainfo']['sha1_name']}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(assembled_report, f, indent=2, ensure_ascii=False)
        
        return result

    def parse_pdfs_sequential(self):
        """Parse all PDFs sequentially."""
        pdf_files = list(self.doc_dir.glob("*.pdf"))
        for pdf_path in pdf_files:
            self.parse_single_pdf(pdf_path)

    def parse_pdfs_parallel(self, optimal_workers: int = 10, chunk_size: int = None):
        """Parse PDFs in parallel using multiple processes."""
        input_doc_paths = list(self.doc_dir.glob("*.pdf"))
        total_pdfs = len(input_doc_paths)
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
                    self.pdf_backend,
                    self.output_dir,
                    self.num_threads,
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


class JsonReportProcessor:
    def __init__(self, metadata_lookup: dict = None, debug_data_path: Path = None):
        self.metadata_lookup = metadata_lookup or {}
        self.debug_data_path = debug_data_path

    def assemble_report(self, conv_result, normalized_data=None):
        """Assemble the report from conversion result."""
        data = normalized_data if normalized_data is not None else conv_result.document.export_to_dict()
        assembled_report = {}
        assembled_report['metainfo'] = self.assemble_metainfo(data)
        assembled_report['content'] = self.assemble_content(data)
        assembled_report['tables'] = self.assemble_tables(conv_result.document.tables, data)
        assembled_report['pictures'] = self.assemble_pictures(data)
        self.debug_data(data)
        return assembled_report

    def assemble_metainfo(self, data):
        """Extract metadata from document."""
        metainfo = {}
        sha1_name = data['origin']['filename'].rsplit('.', 1)[0]
        metainfo['sha1_name'] = sha1_name
        metainfo['filename'] = data['origin']['filename']
        metainfo['pages_amount'] = len(data.get('pages', []))
        metainfo['text_blocks_amount'] = len(data.get('texts', []))
        metainfo['tables_amount'] = len(data.get('tables', []))
        metainfo['pictures_amount'] = len(data.get('pictures', []))
        metainfo['equations_amount'] = len(data.get('equations', []))
        metainfo['footnotes_amount'] = len([t for t in data.get('texts', []) if t.get('label') == 'footnote'])
        
        if self.metadata_lookup and sha1_name in self.metadata_lookup:
            csv_meta = self.metadata_lookup[sha1_name]
            metainfo['document_name'] = csv_meta.get('document_name', sha1_name)
        else:
            metainfo['document_name'] = sha1_name
        
        return metainfo

    def assemble_content(self, data):
        """Assemble document content by pages."""
        pages = {}
        
        for text_item in data.get('texts', []):
            if 'prov' in text_item and text_item['prov']:
                page_num = text_item['prov'][0]['page_no']
                if page_num not in pages:
                    pages[page_num] = {
                        'page': page_num,
                        'content': [],
                        'page_dimensions': text_item['prov'][0].get('bbox', {})
                    }
                pages[page_num]['content'].append({
                    'type': text_item.get('label', 'text'),
                    'text': text_item.get('text', ''),
                    'bbox': text_item['prov'][0].get('bbox', {})
                })
        
        sorted_pages = [pages[page_num] for page_num in sorted(pages.keys())]
        return sorted_pages

    def assemble_tables(self, tables, data):
        """Assemble tables from document."""
        assembled_tables = []
        for i, table in enumerate(tables):
            table_json_obj = table.model_dump()
            table_md = self._table_to_md(table_json_obj)
            table_html = table.export_to_html()
            table_data = data['tables'][i]
            table_page_num = table_data['prov'][0]['page_no']
            table_bbox = table_data['prov'][0]['bbox']
            table_bbox = [
                table_bbox['l'],
                table_bbox['t'],
                table_bbox['r'],
                table_bbox['b']
            ]
            nrows = table_data['data']['num_rows']
            ncols = table_data['data']['num_cols']
            ref_num = table_data['self_ref'].split('/')[-1]
            ref_num = int(ref_num)
            
            table_obj = {
                'table_id': ref_num,
                'page': table_page_num,
                'bbox': table_bbox,
                '#-rows': nrows,
                '#-cols': ncols,
                'markdown': table_md,
                'html': table_html,
                'json': table_json_obj
            }
            assembled_tables.append(table_obj)
        return assembled_tables

    def _table_to_md(self, table):
        """Convert table to markdown format."""
        table_data = []
        for row in table['data']['grid']:
            table_row = [cell['text'] for cell in row]
            table_data.append(table_row)
        
        if len(table_data) > 1 and len(table_data[0]) > 0:
            try:
                md_table = tabulate(table_data[1:], headers=table_data[0], tablefmt="github")
            except ValueError:
                md_table = tabulate(table_data[1:], headers=table_data[0], tablefmt="github", disable_numparse=True)
        else:
            md_table = tabulate(table_data, tablefmt="github")
        return md_table

    def assemble_pictures(self, data):
        """Assemble pictures from document."""
        assembled_pictures = []
        for i, picture in enumerate(data.get('pictures', [])):
            children_list = self._process_picture_block(picture, data)
            ref_num = picture['self_ref'].split('/')[-1]
            ref_num = int(ref_num)
            picture_page_num = picture['prov'][0]['page_no']
            picture_bbox = picture['prov'][0]['bbox']
            picture_bbox = [
                picture_bbox['l'],
                picture_bbox['t'],
                picture_bbox['r'],
                picture_bbox['b']
            ]
            picture_obj = {
                'picture_id': ref_num,
                'page': picture_page_num,
                'bbox': picture_bbox,
                'children': children_list,
            }
            assembled_pictures.append(picture_obj)
        return assembled_pictures

    def _process_picture_block(self, picture, data):
        """Process picture children."""
        children_list = []
        for item in picture.get('children', []):
            if isinstance(item, dict) and '$ref' in item:
                ref = item['$ref']
                ref_type, ref_num = ref.split('/')[-2:]
                ref_num = int(ref_num)
                if ref_type == 'texts':
                    content_item = self._process_text_reference(ref_num, data)
                    if content_item:
                        children_list.append(content_item)
        return children_list

    def _process_text_reference(self, ref_num, data):
        """Process text reference."""
        if ref_num < len(data.get('texts', [])):
            text_item = data['texts'][ref_num]
            return {
                'type': text_item.get('label', 'text'),
                'text': text_item.get('text', '')
            }
        return None

    def debug_data(self, data):
        """Save debug data if path is set."""
        if self.debug_data_path is None:
            return
        doc_name = data.get('name', 'unknown')
        path = self.debug_data_path / f"{doc_name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
