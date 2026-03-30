import os
import sys
import logging
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import Pipeline, configs, RunConfig
from src.retrieval import HybridRetriever
from src.multi_path_retrieval import MultiPathRetriever

_log = logging.getLogger(__name__)


def create_app():
    app = Flask(__name__, 
                template_folder='templates',
                static_folder='static')
    CORS(app)
    
    # Configuration
    app.config['UPLOAD_FOLDER'] = Path(__file__).parent.parent / 'data' / 'uploads'
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
    app.config['ALLOWED_EXTENSIONS'] = {'pdf'}
    
    # Ensure directories exist
    root_path = Path(__file__).parent.parent
    (root_path / 'data' / 'pdf_reports').mkdir(parents=True, exist_ok=True)
    (root_path / 'data' / 'debug').mkdir(parents=True, exist_ok=True)
    (root_path / 'data' / 'databases').mkdir(parents=True, exist_ok=True)
    (app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)
    
    # Initialize pipeline with multi-path config
    pipeline = Pipeline(root_path, run_config=configs["multi_path"])
    
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
    
    @app.route('/')
    def index():
        return render_template('index.html')
    
    @app.route('/api/documents', methods=['GET'])
    def get_documents():
        """Get list of all available documents."""
        try:
            # Check if vector databases exist
            if not pipeline.vector_dbs_dir.exists() or not list(pipeline.vector_dbs_dir.glob("*.faiss")):
                return jsonify({
                    'success': True,
                    'documents': [],
                    'retrieval_mode': 'none',
                    'message': 'No processed documents found. Please upload and process PDFs first.'
                })
            
            # Try multi-path retriever first (if BM25 indices exist), fallback to HybridRetriever
            bm25_dir = pipeline.databases_dir / "bm25_indices"
            has_bm25 = bm25_dir.exists() and any(bm25_dir.glob("*.pkl"))
            
            if has_bm25:
                try:
                    retriever = MultiPathRetriever(
                        pipeline.vector_dbs_dir,
                        bm25_dir,
                        pipeline.merged_reports_dir
                    )
                    documents = retriever.get_all_documents()
                    return jsonify({
                        'success': True,
                        'documents': documents,
                        'retrieval_mode': 'multi_path'
                    })
                except Exception as e:
                    _log.warning(f"MultiPathRetriever failed: {e}, falling back to HybridRetriever")
            
            # Fallback to HybridRetriever
            retriever = HybridRetriever(
                pipeline.vector_dbs_dir,
                pipeline.merged_reports_dir
            )
            documents = retriever.get_all_documents()
            return jsonify({
                'success': True,
                'documents': documents,
                'retrieval_mode': 'standard'
            })
        except Exception as e:
            import logging
            logging.error(f"Error in get_documents: {e}")
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
            pipeline.create_vector_dbs()
            pipeline.create_bm25_indices()  # Create BM25 for multi-path
            
            return jsonify({
                'success': True,
                'message': f'Processed {len(uploaded_files)} documents with multi-path indices',
                'retrieval_mode': 'multi_path'
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/query', methods=['POST'])
    def query():
        """Query the RAG system using multi-path retrieval."""
        try:
            data = request.get_json()
            question = data.get('question', '').strip()
            sha1_name = data.get('sha1_name')  # Optional: specific document
            
            if not question:
                return jsonify({'success': False, 'error': 'Question is required'}), 400
            
            # Check if documents have been processed
            if not pipeline.vector_dbs_dir.exists() or not list(pipeline.vector_dbs_dir.glob("*.faiss")):
                return jsonify({
                    'success': False, 
                    'error': 'No processed documents found. Please upload and process PDFs first.'
                }), 400
            
            # Generate answer using the pipeline (it handles multi-path vs standard internally)
            answer = pipeline.query_single(question, sha1_name)
            
            # Determine retrieval method from answer or pipeline config
            retrieval_method = answer.get('retrieval_method', 'standard')
            if hasattr(pipeline.run_config, 'use_multi_path') and pipeline.run_config.use_multi_path:
                retrieval_method = 'multi_path'
            
            return jsonify({
                'success': True,
                'answer': answer.get('final_answer', 'N/A'),
                'reasoning': answer.get('reasoning_summary', ''),
                'analysis': answer.get('step_by_step_analysis', ''),
                'pages': answer.get('relevant_pages', []),
                'retrieval_method': retrieval_method,
                'context': [
                    {
                        'page': ctx.get('page', 0),
                        'text': ctx.get('text', '')[:500] + '...' if len(ctx.get('text', '')) > 500 else ctx.get('text', '')
                    }
                    for ctx in answer.get('retrieved_context', [])
                ],
                'schema': answer.get('schema', 'unknown')
            })
        except Exception as e:
            _log.error(f"Error in query: {e}")
            import traceback
            _log.error(traceback.format_exc())
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
                'vector_dbs': len(list(pipeline.vector_dbs_dir.glob('*.faiss'))) if pipeline.vector_dbs_dir.exists() else 0,
                'bm25_indices': len(list(pipeline.databases_dir.glob('bm25_indices/*.pkl'))) if (pipeline.databases_dir / 'bm25_indices').exists() else 0,
                'retrieval_mode': 'multi_path' if (pipeline.databases_dir / 'bm25_indices').exists() else 'standard'
            }
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
                pipeline.databases_dir / 'bm25_indices',
                app.config['UPLOAD_FOLDER']
            ]
            
            for dir_path in dirs_to_clear:
                if dir_path.exists():
                    shutil.rmtree(dir_path)
                    dir_path.mkdir(parents=True, exist_ok=True)
            
            return jsonify({'success': True, 'message': 'All data cleared'})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/retrieval-info', methods=['GET'])
    def get_retrieval_info():
        """Get information about the retrieval system."""
        try:
            bm25_dir = pipeline.databases_dir / "bm25_indices"
            has_multi_path = bm25_dir.exists() and any(bm25_dir.glob("*.pkl"))
            
            return jsonify({
                'success': True,
                'info': {
                    'mode': 'multi_path' if has_multi_path else 'standard',
                    'paths': [
                        'Semantic (FAISS dense vectors)',
                        'Lexical (BM25 keyword matching)',
                        'RRF Fusion (Reciprocal Rank Fusion)'
                    ] if has_multi_path else ['Standard Hybrid Retrieval'],
                    'rrf_k': 60 if has_multi_path else None,
                    'bm25_indices': len(list(bm25_dir.glob("*.pkl"))) if has_multi_path else 0
                }
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
