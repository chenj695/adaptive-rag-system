from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from pyprojroot import here
import logging

_log = logging.getLogger(__name__)


@dataclass
class RunConfig:
    use_serialized_tables: bool = False
    parent_document_retrieval: bool = False
    llm_reranking: bool = False
    parallel_requests: int = 1
    llm_reranking_sample_size: int = 28
    top_n_retrieval: int = 6
    full_context: bool = False
    submission_name: str = ""
    pipeline_details: str = ""
    api_provider: str = "openai"
    answering_model: str = "o3-mini-2025-01-31"
    config_suffix: str = ""


# Configuration presets
base_config = RunConfig()

parent_document_retrieval_config = RunConfig(
    parent_document_retrieval=True
)

max_config = RunConfig(
    use_serialized_tables=True,
    parent_document_retrieval=True,
    llm_reranking=True,
    parallel_requests=5,
    submission_name="RAG System v.3",
    pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + reranking + SO CoT; llm = o3-mini"
)

max_no_ser_tab_config = RunConfig(
    use_serialized_tables=False,
    parent_document_retrieval=True,
    llm_reranking=True,
    parallel_requests=5,
    submission_name="RAG System v.4",
    pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + reranking + SO CoT; llm = o3-mini; no serialized tables"
)

max_nst_o3m_config = RunConfig(
    use_serialized_tables=False,
    parent_document_retrieval=True,
    llm_reranking=True,
    parallel_requests=5,
    submission_name="RAG System v.5",
    pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + reranking + SO CoT; llm = o3-mini",
    answering_model="o3-mini-2025-01-31",
    config_suffix="_max_nst_o3m"
)

max_st_o3m_config = RunConfig(
    use_serialized_tables=True,
    parent_document_retrieval=True,
    llm_reranking=True,
    parallel_requests=5,
    submission_name="RAG System v.6",
    pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + reranking + SO CoT; llm = o3-mini; with serialized tables",
    answering_model="o3-mini-2025-01-31",
    config_suffix="_max_st_o3m"
)

gemini_thinking_config = RunConfig(
    use_serialized_tables=False,
    parent_document_retrieval=True,
    llm_reranking=False,
    parallel_requests=1,
    full_context=True,
    submission_name="RAG System v.8",
    pipeline_details="Custom pdf parsing + Full Context + Router + SO CoT + SO reparser; llm = gemini-2.0-flash-thinking-exp-01-21",
    api_provider="gemini",
    answering_model="gemini-2.0-flash-thinking-exp-01-21",
    config_suffix="_gemini_thinking_fc"
)

configs = {
    "base": base_config,
    "pdr": parent_document_retrieval_config,
    "max": max_config,
    "max_no_ser_tab": max_no_ser_tab_config,
    "max_nst_o3m": max_nst_o3m_config,
    "max_st_o3m": max_st_o3m_config,
    "gemini_thinking": gemini_thinking_config
}

preprocess_configs = {
    "ser_tab": RunConfig(use_serialized_tables=True),
    "no_ser_tab": RunConfig(use_serialized_tables=False)
}


class Pipeline:
    def __init__(self, root_path: Path, run_config: Optional[RunConfig] = None):
        self.root_path = Path(root_path)
        self.run_config = run_config or RunConfig()
        
        # Define paths
        self.data_dir = self.root_path / "data"
        self.pdf_dir = self.data_dir / "pdf_reports"
        self.debug_dir = self.data_dir / "debug"
        self.databases_dir = self.data_dir / "databases"
        self.parsed_reports_dir = self.debug_dir / "data_01_parsed_reports"
        self.merged_reports_dir = self.debug_dir / "data_02_merged_reports"
        self.markdown_reports_dir = self.debug_dir / "data_03_reports_markdown"
        self.chunked_reports_dir = self.databases_dir / "chunked_reports"
        self.vector_dbs_dir = self.databases_dir / "vector_dbs"

    @staticmethod
    def download_docling_models():
        """Download required docling models."""
        from docling.datamodel.document import ConversionResult
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()

    def parse_pdf_reports(self, parallel: bool = True, chunk_size: int = 2, max_workers: int = 10):
        """Parse PDF reports using docling."""
        from src.pdf_parsing import PdfParser
        parser = PdfParser(
            doc_dir=self.pdf_dir,
            output_dir=self.parsed_reports_dir,
            debug_data_path=self.debug_dir / "data_01_parsed_reports_debug"
        )
        if parallel:
            parser.parse_pdfs_parallel(optimal_workers=max_workers, chunk_size=chunk_size)
        else:
            parser.parse_pdfs_sequential()

    def serialize_tables(self, max_workers: int = 5):
        """Serialize tables in parsed reports."""
        from src.tables_serialization import AsyncTableSerializer
        processor = AsyncTableSerializer()
        processor.process_directory_parallel(self.parsed_reports_dir, max_workers)

    def merge_reports(self):
        """Convert parsed reports to simpler page-based JSON format."""
        from src.parsed_reports_merging import ReportsProcessor
        processor = ReportsProcessor(use_serialized_tables=self.run_config.use_serialized_tables)
        processor.process_reports(self.parsed_reports_dir, self.merged_reports_dir)

    def export_reports_to_markdown(self):
        """Export reports to markdown for full-context processing."""
        from src.parsed_reports_merging import ReportsProcessor
        processor = ReportsProcessor(use_serialized_tables=self.run_config.use_serialized_tables)
        processor.export_reports_to_markdown(self.merged_reports_dir, self.markdown_reports_dir)

    def chunk_reports(self):
        """Split reports into chunks for vectorization."""
        from src.text_splitter import TextSplitter
        splitter = TextSplitter()
        splitter.split_all_reports(
            self.merged_reports_dir,
            self.chunked_reports_dir,
            serialized_tables_dir=self.parsed_reports_dir if self.run_config.use_serialized_tables else None
        )

    def create_vector_dbs(self):
        """Create FAISS vector databases from chunked reports."""
        from src.ingestion import VectorDBIngestor
        ingestor = VectorDBIngestor()
        ingestor.process_reports(self.chunked_reports_dir, self.vector_dbs_dir)

    def process_parsed_reports(self):
        """Run full preprocessing pipeline."""
        self.merge_reports()
        self.export_reports_to_markdown()
        self.chunk_reports()
        self.create_vector_dbs()

    def process_questions(self):
        """Process all questions from questions.json."""
        from src.questions_processing import QuestionsProcessor
        questions_file = self.data_dir / "questions.json"
        documents_dir = self.merged_reports_dir
        output_path = self.data_dir / f"answers{self.run_config.config_suffix}.json"
        processor = QuestionsProcessor(
            questions_file=questions_file,
            vector_db_dir=self.vector_dbs_dir,
            documents_dir=documents_dir,
            markdown_reports_dir=self.markdown_reports_dir,
            run_config=self.run_config
        )
        processor.process_all_questions(
            output_path=str(output_path),
            team_email="rag@system.local",
            submission_name=self.run_config.submission_name,
            submission_file=True,
            pipeline_details=self.run_config.pipeline_details
        )

    def query_single(self, question: str, document_name: Optional[str] = None) -> dict:
        """Query a single question (for WebUI)."""
        from src.questions_processing import QuestionsProcessor
        
        # Create a temporary RunConfig for single queries
        processor = QuestionsProcessor(
            questions_file=None,
            vector_db_dir=self.vector_dbs_dir,
            documents_dir=self.merged_reports_dir,
            markdown_reports_dir=self.markdown_reports_dir,
            run_config=self.run_config
        )
        
        return processor.query_single(question, document_name)


if __name__ == "__main__":
    root_path = here()
    pipeline = Pipeline(root_path, run_config=max_nst_o3m_config)
