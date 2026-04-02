import os
import sys
import shutil
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# 将项目根目录添加到系统路径，确保能导入 src 模块
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.pipeline import Pipeline, configs
from src.retrieval import HybridRetriever

def create_app():
    app = Flask(__name__, template_folder='templates', static_folder='static')
    CORS(app)

    # 路径配置
    root_path = Path(__file__).parent.parent
    app.config['UPLOAD_FOLDER'] = root_path / 'data' / 'uploads'
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB
    app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

    # 确保必要的目录存在
    (root_path / 'data' / 'pdf_reports').mkdir(parents=True, exist_ok=True)
    (root_path / 'data' / 'debug').mkdir(parents=True, exist_ok=True)
    (root_path / 'data' / 'databases').mkdir(parents=True, exist_ok=True)
    (app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)

    # 初始化 RAG 流水线
    # 注意：确保 configs 中存在对应的配置键
    pipeline = Pipeline(root_path, run_config=configs.get("max_nst_o3m", {}))

    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/api/documents', methods=['GET'])
    def get_documents():
        """获取所有已处理文档的列表，用于前端渲染文档库卡片。"""
        try:
            # 如果向量库目录不存在或没有索引文件，返回空
            if not pipeline.vector_dbs_dir.exists() or not any(pipeline.vector_dbs_dir.glob("*.faiss")):
                return jsonify({'success': True, 'documents': []})

            retriever = HybridRetriever(
                pipeline.vector_dbs_dir, 
                pipeline.merged_reports_dir
            )
            # 获取文档数据
            raw_docs = retriever.get_all_documents()
            
            # 格式化数据以匹配 app.js 中的渲染逻辑
            documents = []
            for doc in raw_docs:
                documents.append({
                    'sha1_name': doc.get('sha1_name'),
                    'document_name': doc.get('document_name') or doc.get('filename', 'Unknown Document'),
                    'pages_amount': doc.get('pages_amount', 0)
                })
                
            return jsonify({'success': True, 'documents': documents})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/upload', methods=['POST'])
    def upload_file():
        """处理文件上传。"""
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
                errors.append(f"{file.filename}: Invalid file type (PDF only)")

        return jsonify({'success': True, 'uploaded': uploaded, 'errors': errors})

    @app.route('/api/process', methods=['POST'])
    def process_documents():
        """运行 RAG 流水线：解析、合并、分块并创建向量库。"""
        try:
            upload_dir = app.config['UPLOAD_FOLDER']
            pdf_dir = pipeline.pdf_dir
            uploaded_files = list(upload_dir.glob('*.pdf'))

            if not uploaded_files:
                return jsonify({'success': False, 'error': 'No PDF files found in upload folder'}), 400

            # 移动文件到流水线输入目录
            for file in uploaded_files:
                dest = pdf_dir / file.name
                shutil.copy2(file, dest)
                file.unlink() # 移动后删除上传暂存区文件

            # 执行流水线步骤
            pipeline.parse_pdf_reports(parallel=True, max_workers=4)
            pipeline.merge_reports()
            pipeline.chunk_reports()
            pipeline.create_vector_dbs()

            return jsonify({
                'success': True, 
                'message': f'Successfully processed {len(uploaded_files)} documents.'
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/query', methods=['POST'])
    def query():
        """查询 RAG 系统并返回答案和上下文。"""
        try:
            data = request.get_json()
            question = data.get('question', '').strip()
            sha1_name = data.get('sha1_name') # 可选：指定特定文档

            if not question:
                return jsonify({'success': False, 'error': 'Question is required'}), 400

            if not pipeline.vector_dbs_dir.exists() or not any(pipeline.vector_dbs_dir.glob("*.faiss")):
                return jsonify({'success': False, 'error': 'Please process documents first.'}), 400

            answer = pipeline.query_single(question, sha1_name)
            
            return jsonify({
                'success': True,
                'answer': answer.get('final_answer', 'No answer generated'),
                'reasoning': answer.get('reasoning_summary', ''),
                'analysis': answer.get('step_by_step_analysis', ''),
                'pages': answer.get('relevant_pages', []),
                'context': [
                    {
                        'page': ctx.get('page', 0),
                        'text': ctx.get('text', '')
                    } for ctx in answer.get('retrieved_context', [])
                ],
                'schema': answer.get('schema', 'general')
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/stats', methods=['GET'])
    def get_stats():
        """获取系统状态数据。"""
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
        """清理所有已上传和处理的数据。"""
        try:
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
            return jsonify({'success': True, 'message': 'All data has been cleared.'})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
