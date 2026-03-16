import os
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import Pipeline, configs
from src.retrieval import HybridRetriever


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
    
    # Initialize pipeline
    pipeline = Pipeline(root_path, run_config=configs["max_nst_o3m"])
    
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
    
    @app.route('/')
    def index():
        return render_template('index.html')
    
    @app.route('/api/documents', methods=['GET'])
    def get_documents():
        """Get list of all available documents."""
        try:
            retriever = HybridRetriever(
                pipeline.vector_dbs_dir,
                pipeline.merged_reports_dir
            )
            documents = retriever.get_all_documents()
            return jsonify({
                'success': True,
                'documents': documents
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
            pipeline.create_vector_dbs()
            
            return jsonify({
                'success': True,
                'message': f'Processed {len(uploaded_files)} documents'
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
            
            # Use the pipeline's query method
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
                'schema': answer.get('schema', 'unknown')
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
                'vector_dbs': len(list(pipeline.vector_dbs_dir.glob('*.faiss'))) if pipeline.vector_dbs_dir.exists() else 0
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
                app.config['UPLOAD_FOLDER']
            ]
            
            for dir_path in dirs_to_clear:
                if dir_path.exists():
                    shutil.rmtree(dir_path)
                    dir_path.mkdir(parents=True, exist_ok=True)
            
            return jsonify({'success': True, 'message': 'All data cleared'})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
