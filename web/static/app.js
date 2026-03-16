// RAG System Web UI JavaScript

// State
let selectedFiles = [];
let isProcessing = false;

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const fileList = document.getElementById('fileList');
const uploadBtn = document.getElementById('uploadBtn');
const processBtn = document.getElementById('processBtn');
const processSpinner = document.getElementById('processSpinner');
const progressBar = document.getElementById('progressBar');
const progressFill = document.getElementById('progressFill');
const documentList = document.getElementById('documentList');
const documentSelect = document.getElementById('documentSelect');
const refreshDocsBtn = document.getElementById('refreshDocs');
const clearBtn = document.getElementById('clearBtn');
const questionInput = document.getElementById('questionInput');
const queryBtn = document.getElementById('queryBtn');
const querySpinner = document.getElementById('querySpinner');
const answerCard = document.getElementById('answerCard');
const exampleBtns = document.querySelectorAll('.example-btn');

// Toast notifications
function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    const icons = {
        success: '✅',
        error: '❌',
        info: 'ℹ️'
    };
    
    toast.innerHTML = `
        <span>${icons[type]}</span>
        <span>${message}</span>
    `;
    
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100%)';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// File Upload Handlers
uploadArea.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = Array.from(e.dataTransfer.files).filter(f => f.name.endsWith('.pdf'));
    addFiles(files);
});

fileInput.addEventListener('change', (e) => {
    addFiles(Array.from(e.target.files));
});

function addFiles(files) {
    files.forEach(file => {
        if (!selectedFiles.find(f => f.name === file.name)) {
            selectedFiles.push(file);
        }
    });
    updateFileList();
}

function updateFileList() {
    fileList.innerHTML = '';
    
    selectedFiles.forEach((file, index) => {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        fileItem.innerHTML = `
            <span class="file-icon">📄</span>
            <span class="file-name">${file.name}</span>
            <span class="file-size">${formatFileSize(file.size)}</span>
            <button class="remove-btn" data-index="${index}">×</button>
        `;
        fileList.appendChild(fileItem);
    });
    
    // Add remove handlers
    document.querySelectorAll('.remove-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const index = parseInt(e.target.dataset.index);
            selectedFiles.splice(index, 1);
            updateFileList();
        });
    });
    
    uploadBtn.disabled = selectedFiles.length === 0;
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Upload Files
uploadBtn.addEventListener('click', async () => {
    if (selectedFiles.length === 0) return;
    
    const formData = new FormData();
    selectedFiles.forEach(file => formData.append('files', file));
    
    uploadBtn.disabled = true;
    uploadBtn.textContent = 'Uploading...';
    
    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            showToast(`Uploaded ${data.uploaded.length} files successfully`, 'success');
            selectedFiles = [];
            updateFileList();
            processBtn.disabled = false;
            loadDocuments();
            loadStats();
        } else {
            showToast(data.error || 'Upload failed', 'error');
        }
    } catch (error) {
        showToast('Upload failed: ' + error.message, 'error');
    } finally {
        uploadBtn.disabled = false;
        uploadBtn.textContent = 'Upload Files';
    }
});

// Process Documents
processBtn.addEventListener('click', async () => {
    if (isProcessing) return;
    
    isProcessing = true;
    processBtn.disabled = true;
    processSpinner.classList.remove('hidden');
    progressBar.classList.remove('hidden');
    progressFill.style.width = '10%';
    
    try {
        const response = await fetch('/api/process', {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (data.success) {
            progressFill.style.width = '100%';
            showToast(data.message, 'success');
            loadDocuments();
            loadStats();
        } else {
            showToast(data.error || 'Processing failed', 'error');
        }
    } catch (error) {
        showToast('Processing failed: ' + error.message, 'error');
    } finally {
        isProcessing = false;
        processBtn.disabled = false;
        processSpinner.classList.add('hidden');
        setTimeout(() => {
            progressBar.classList.add('hidden');
            progressFill.style.width = '0%';
        }, 1000);
    }
});

// Load Documents
async function loadDocuments() {
    try {
        const response = await fetch('/api/documents');
        const data = await response.json();
        
        if (data.success) {
            // Update document list
            if (data.documents.length === 0) {
                documentList.innerHTML = '<p class="empty">No documents processed yet</p>';
                documentSelect.innerHTML = '<option value="">All Documents</option>';
            } else {
                documentList.innerHTML = data.documents.map(doc => `
                    <div class="document-item">
                        <span class="doc-icon">📄</span>
                        <div class="doc-info">
                            <span class="doc-name">${doc.document_name || doc.filename}</span>
                            <span class="doc-meta">${doc.pages_amount} pages</span>
                        </div>
                    </div>
                `).join('');
                
                // Update select dropdown
                documentSelect.innerHTML = `
                    <option value="">All Documents</option>
                    ${data.documents.map(doc => `
                        <option value="${doc.sha1_name}">${doc.document_name || doc.filename}</option>
                    `).join('')}
                `;
            }
        }
    } catch (error) {
        console.error('Failed to load documents:', error);
    }
}

// Load Statistics
async function loadStats() {
    try {
        const response = await fetch('/api/stats');
        const data = await response.json();
        
        if (data.success) {
            document.getElementById('statPdfs').textContent = data.stats.pdf_files;
            document.getElementById('statParsed').textContent = data.stats.parsed_reports;
            document.getElementById('statVectors').textContent = data.stats.vector_dbs;
        }
    } catch (error) {
        console.error('Failed to load stats:', error);
    }
}

// Refresh Documents
refreshDocsBtn.addEventListener('click', () => {
    loadDocuments();
    loadStats();
    showToast('Documents refreshed', 'info');
});

// Clear Data
clearBtn.addEventListener('click', async () => {
    if (!confirm('Are you sure you want to clear all data?')) return;
    
    try {
        const response = await fetch('/api/clear', { method: 'POST' });
        const data = await response.json();
        
        if (data.success) {
            showToast(data.message, 'success');
            loadDocuments();
            loadStats();
            answerCard.classList.add('hidden');
        }
    } catch (error) {
        showToast('Clear failed: ' + error.message, 'error');
    }
});

// Query
queryBtn.addEventListener('click', async () => {
    const question = questionInput.value.trim();
    if (!question) {
        showToast('Please enter a question', 'error');
        return;
    }
    
    const sha1_name = documentSelect.value;
    
    queryBtn.disabled = true;
    querySpinner.classList.remove('hidden');
    answerCard.classList.add('hidden');
    
    try {
        const response = await fetch('/api/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, sha1_name })
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayAnswer(data);
        } else {
            showToast(data.error || 'Query failed', 'error');
        }
    } catch (error) {
        showToast('Query failed: ' + error.message, 'error');
    } finally {
        queryBtn.disabled = false;
        querySpinner.classList.add('hidden');
    }
});

function displayAnswer(data) {
    // Schema badge
    document.getElementById('answerSchema').textContent = data.schema || 'text';
    
    // Answer text
    document.getElementById('answerText').textContent = data.answer || 'N/A';
    
    // Reasoning
    document.getElementById('reasoningText').textContent = data.reasoning || 'No reasoning provided';
    
    // Analysis
    document.getElementById('analysisText').textContent = data.analysis || 'No analysis provided';
    
    // Context
    const contextList = document.getElementById('contextList');
    if (data.context && data.context.length > 0) {
        contextList.innerHTML = data.context.map((ctx, idx) => `
            <div class="context-item">
                <div class="context-header">
                    <span class="context-page">Page ${ctx.page}</span>
                    <span class="context-score">#${idx + 1}</span>
                </div>
                <div class="context-text">${escapeHtml(ctx.text)}</div>
            </div>
        `).join('');
    } else {
        contextList.innerHTML = '<p class="empty">No context retrieved</p>';
    }
    
    // Pages
    const pageTags = document.getElementById('pageTags');
    if (data.pages && data.pages.length > 0) {
        pageTags.innerHTML = data.pages.map(p => `
            <span class="page-tag">Page ${p}</span>
        `).join('');
    } else {
        pageTags.innerHTML = '<span class="page-tag">N/A</span>';
    }
    
    answerCard.classList.remove('hidden');
    answerCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Example Questions
exampleBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        questionInput.value = btn.dataset.question;
        questionInput.focus();
    });
});

// Initialize
loadDocuments();
loadStats();
