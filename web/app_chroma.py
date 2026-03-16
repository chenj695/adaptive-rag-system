"""Flask web app with Chroma support.

This is an alternative to app.py that uses Chroma instead of FAISS.
"""

import os
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import Pipeline, configs
from src.ingestion_chroma import ChromaRetriever, ChromaIngestor


def create_app(use_chroma: bool = True):
    """Create Flask app with Chroma support.
    
    Args:
        use_chroma: If True, use Chroma. If False, use FAISS.
    """
    app = Flask(__name__, 
                template_folder='templates',
                static_folder='static')
    CORS(app)
    
    # Configuration
    app.config['UPLOAD_FOLDER'] = Path(__file__).parent.parent / 'data' / 'uploads'
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
    app.config['ALLOWED_EXTENSIONS'] = {'pdf'}
    app.config['USE_CHROMA'] = use_chroma
    
    # Ensure directories exist
    root_path = Path(__file__).parent.parent
    (root_path / 'data' / 'pdf_reports').mkdir(parents=True, exist_ok=True)
    (root_path / 'data' / 'debug').mkdir(parents=True, exist_ok=True)
    (root_path / 'data' / 'chroma_db').mkdir(parents=True, exist_ok=True)
    (app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    pipeline = Pipeline(root_path, run_config=configs["max_nst_o3m"])
    
    # Chroma paths
    chroma_persist_dir = root_path / 'data' / 'chroma_db'
    
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
    
    @app.route('/')
    def index():
        return render_template('index.html')
    
    @app.route('/api/documents', methods=['GET'])
    def get_documents():
        """Get list of all available documents."""
        try:
            if app.config['USE_CHROMA']:
                try:
                    retriever = ChromaRetriever(chroma_persist_dir)
                    documents = retriever.get_all_documents()
                except Exception:
                    documents = []
            else:
                from src.retrieval import HybridRetriever
                retriever = HybridRetriever(
                    pipeline.vector_dbs_dir,
                    pipeline.merged_reports_dir
                )
                documents = retriever.get_all_documents()
            
            return jsonify({
                'success': True,
                'documents': documents,
                'backend': 'chroma' if app.config['USE_CHROMA'] else 'faiss'
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/upload', methods=['POST'])
    def upload_file():
        """Upload PDF files."""
        if 'files' not in request.files:
            return jsonify({'success': False, 'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        uploaded = []
        errors = []
        
        for file in files:
            if file.filename == '':
                continue
            
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                save_path = app.config['UPLOAD_FOLDER'] / filename
                file.save(save_path)
                uploaded.append(filename)
            else:
                errors.append(f"{file.filename}: Invalid file type")
        
        return jsonify({
            'success': True,
            'uploaded': uploaded,
            'errors': errors
        })
    
    @app.route('/api/process', methods=['POST'])
    def process_documents():
        """Run the RAG pipeline on uploaded documents."""
        try:
            import shutil
            
            # Move uploaded files to pdf_reports
            upload_dir = app.config['UPLOAD_FOLDER']
            pdf_dir = pipeline.pdf_dir
            
            uploaded_files = list(upload_dir.glob('*.pdf'))
            if not uploaded_files:
                return jsonify({'success': False, 'error': 'No PDF files to process'}), 400
            
            for file in uploaded_files:
                dest = pdf_dir / file.name
                shutil.copy2(file, dest)
            
            # Run pipeline
            pipeline.parse_pdf_reports(parallel=True, max_workers=4)
            pipeline.merge_reports()
            pipeline.chunk_reports()
            
            # Use Chroma or FAISS for indexing
            if app.config['USE_CHROMA']:
                ingestor = ChromaIngestor(chroma_persist_dir)
                ingestor.process_reports(pipeline.chunked_reports_dir)
            else:
                pipeline.create_vector_dbs()
            
            return jsonify({
                'success': True,
                'message': f'Processed {len(uploaded_files)} documents with {"Chroma" if app.config["USE_CHROMA"] else "FAISS"}'
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/query', methods=['POST'])
    def query():
        """Query the RAG system."""
        try:
            data = request.get_json()
            question = data.get('question', '').strip()
            sha1_name = data.get('sha1_name')  # Optional: specific document
            
            if not question:
                return jsonify({'success': False, 'error': 'Question is required'}), 400
            
            # Use Chroma or FAISS for retrieval
            if app.config['USE_CHROMA']:
                retriever = ChromaRetriever(chroma_persist_dir)
                
                if sha1_name:
                    context_results = retriever.retrieve_by_document(question, sha1_name, n_results=6)
                else:
                    context_results = retriever.retrieve(question, n_results=6)
                
                # Format context for LLM
                rag_context = [
                    {
                        "page": r["metadata"].get("page", 0),
                        "text": r["text"],
                        "distance": r["distance"],
                        "document": r["metadata"].get("document_name", "Unknown")
                    }
                    for r in context_results
                ]
                
                # Generate answer using existing pipeline logic
                from src.questions_processing import OpenAIProcessor
                from src.pipeline import RunConfig
                
                processor = OpenAIProcessor(model="o3-mini-2025-01-31")
                
                schema = _determine_schema(question)
                answer = processor.get_answer_from_rag_context(
                    question=question,
                    rag_context=rag_context,
                    schema=schema,
                    model="o3-mini-2025-01-31"
                )
                
                return jsonify({
                    'success': True,
                    'answer': answer.get('final_answer', 'N/A'),
                    'reasoning': answer.get('reasoning_summary', ''),
                    'analysis': answer.get('step_by_step_analysis', ''),
                    'pages': answer.get('relevant_pages', []),
                    'context': [
                        {
                            'page': ctx.get('page', 0),
                            'text': ctx.get('text', '')[:500] + '...' if len(ctx.get('text', '')) > 500 else ctx.get('text', ''),
                            'document': ctx.get('document', 'Unknown')
                        }
                        for ctx in rag_context
                    ],
                    'schema': schema,
                    'backend': 'chroma'
                })
            else:
                # Use original FAISS-based pipeline
                answer = pipeline.query_single(question, sha1_name)
                
                return jsonify({
                    'success': True,
                    'answer': answer.get('final_answer', 'N/A'),
                    'reasoning': answer.get('reasoning_summary', ''),
                    'analysis': answer.get('step_by_step_analysis', ''),
                    'pages': answer.get('relevant_pages', []),
                    'context': [
                        {
                            'page': ctx.get('page', 0),
                            'text': ctx.get('text', '')[:500] + '...' if len(ctx.get('text', '')) > 500 else ctx.get('text', '')
                        }
                        for ctx in answer.get('retrieved_context', [])
                    ],
                    'schema': answer.get('schema', 'unknown'),
                    'backend': 'faiss'
                })
                
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/stats', methods=['GET'])
    def get_stats():
        """Get system statistics."""
        try:
            stats = {
                'pdf_files': len(list(pipeline.pdf_dir.glob('*.pdf'))),
                'parsed_reports': len(list(pipeline.parsed_reports_dir.glob('*.json'))) if pipeline.parsed_reports_dir.exists() else 0,
                'use_chroma': app.config['USE_CHROMA']
            }
            
            if app.config['USE_CHROMA']:
                try:
                    ingestor = ChromaIngestor(chroma_persist_dir)
                    chroma_stats = ingestor.get_stats()
                    stats['chroma_chunks'] = chroma_stats['total_chunks']
                    stats['chroma_documents'] = chroma_stats['unique_documents']
                except:
                    stats['chroma_chunks'] = 0
                    stats['chroma_documents'] = 0
            else:
                stats['vector_dbs'] = len(list(pipeline.vector_dbs_dir.glob('*.faiss'))) if pipeline.vector_dbs_dir.exists() else 0
            
            return jsonify({'success': True, 'stats': stats})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/clear', methods=['POST'])
    def clear_data():
        """Clear all processed data."""
        try:
            import shutil
            
            dirs_to_clear = [
                pipeline.pdf_dir,
                pipeline.parsed_reports_dir,
                pipeline.merged_reports_dir,
                pipeline.chunked_reports_dir,
                pipeline.vector_dbs_dir,
                app.config['UPLOAD_FOLDER']
            ]
            
            if app.config['USE_CHROMA']:
                dirs_to_clear.append(chroma_persist_dir)
            
            for dir_path in dirs_to_clear:
                if dir_path.exists():
                    shutil.rmtree(dir_path)
                    dir_path.mkdir(parents=True, exist_ok=True)
            
            return jsonify({'success': True, 'message': 'All data cleared'})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    return app


def _determine_schema(question_text: str) -> str:
    """Determine answer schema type from question."""
    question_lower = question_text.lower()
    
    boolean_starters = ['did ', 'was ', 'were ', 'have ', 'has ', 'is ', 'are ', 'do ', 'does ', 'can ', 'could ']
    if any(question_lower.startswith(starter) for starter in boolean_starters):
        return "boolean"
    
    if any(phrase in question_lower for phrase in ['what is the name', 'which ', 'who ', 'what was the name']):
        return "name"
    
    number_indicators = ['how much', 'how many', 'what is the amount', 'what was the amount', 
                       'what is the value', 'what was the value', 'what is the number',
                       'what percentage', 'what is the total', 'what was the total']
    if any(phrase in question_lower for phrase in number_indicators):
        return "number"
    
    return "text"


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--faiss', action='store_true', help='Use FAISS instead of Chroma')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()
    
    app = create_app(use_chroma=not args.faiss)
    
    backend = "Chroma" if not args.faiss else "FAISS"
    print(f"Starting RAG Web UI with {backend} backend")
    print(f"URL: http://{args.host}:{args.port}")
    
    app.run(debug=True, host=args.host, port=args.port)
