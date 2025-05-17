document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const previewContainer = document.getElementById('previewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const removeImage = document.getElementById('removeImage');
    const predictButton = document.getElementById('predictButton');
    const resultContainer = document.getElementById('resultContainer');
    const resultIcon = document.getElementById('resultIcon');
    const predictionText = document.getElementById('predictionText');
    const confidenceText = document.getElementById('confidenceText');
    const confidenceBar = document.getElementById('confidenceBar');
    const newAnalysisButton = document.getElementById('newAnalysisButton');
    const loadingContainer = document.getElementById('loadingContainer');
    const resultsFilter = document.getElementById('resultsFilter');
    const resultsList = document.getElementById('resultsList');

    // Drag and drop handlers
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    // File input handler
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Handle file selection
    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            showToast('Please select an image file', 'error');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            previewContainer.style.display = 'block';
            dropZone.style.display = 'none';
        };
        reader.readAsDataURL(file);
    }

    // Remove image handler
    removeImage.addEventListener('click', () => {
        imagePreview.src = '';
        previewContainer.style.display = 'none';
        dropZone.style.display = 'block';
        resultContainer.style.display = 'none';
    });

    // Analyze button handler
    predictButton.addEventListener('click', async () => {
        if (!imagePreview.src) return;

        // Show loading state
        loadingContainer.style.display = 'block';
        previewContainer.style.display = 'none';
        resultContainer.style.display = 'none';

        try {
            const response = await fetch('http://localhost:5000/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image: imagePreview.src
                })
            });

            const data = await response.json();
            
            // Update UI with results
            resultIcon.className = `fas ${data.prediction === 'AMD' ? 'fa-exclamation-triangle text-danger' : 'fa-check-circle text-success'}`;
            predictionText.textContent = data.prediction;
            confidenceText.textContent = `Confidence: ${(data.confidence * 100).toFixed(2)}%`;
            confidenceBar.style.width = `${data.confidence * 100}%`;
            confidenceBar.className = `progress-bar ${data.prediction === 'AMD' ? 'bg-danger' : 'bg-success'}`;

            // Show results
            loadingContainer.style.display = 'none';
            resultContainer.style.display = 'block';

            // Add to results history
            addToResultsHistory(data);

        } catch (error) {
            showToast('Error analyzing image. Please try again.', 'error');
            loadingContainer.style.display = 'none';
            previewContainer.style.display = 'block';
        }
    });

    // New analysis button handler
    newAnalysisButton.addEventListener('click', () => {
        resultContainer.style.display = 'none';
        previewContainer.style.display = 'block';
    });

    // Results filter handler
    resultsFilter.addEventListener('change', filterResults);

    // Add result to history
    function addToResultsHistory(data) {
        const resultItem = document.createElement('div');
        resultItem.className = 'result-item fade-in';
        resultItem.dataset.status = data.prediction.toLowerCase();

        resultItem.innerHTML = `
            <div class="result-date">${new Date().toLocaleString()}</div>
            <div class="result-image">
                <img src="${imagePreview.src}" alt="Result">
            </div>
            <div class="result-details">
                <div class="result-status ${data.prediction.toLowerCase()}">
                    ${data.prediction}
                </div>
                <div class="result-confidence">
                    Confidence: ${(data.confidence * 100).toFixed(2)}%
                </div>
                <div class="result-actions">
                    <button class="btn btn-sm btn-outline-primary" onclick="viewDetails(this)">
                        <i class="fas fa-eye me-1"></i>View Details
                    </button>
                    <button class="btn btn-sm btn-outline-danger" onclick="deleteResult(this)">
                        <i class="fas fa-trash me-1"></i>Delete
                    </button>
                </div>
            </div>
        `;

        resultsList.insertBefore(resultItem, resultsList.firstChild);
        filterResults();
    }

    // Filter results
    function filterResults() {
        const filter = resultsFilter.value;
        const items = resultsList.getElementsByClassName('result-item');

        Array.from(items).forEach(item => {
            if (filter === 'all' || item.dataset.status === filter) {
                item.style.display = 'flex';
            } else {
                item.style.display = 'none';
            }
        });
    }

    // View result details
    window.viewDetails = function(button) {
        const resultItem = button.closest('.result-item');
        const image = resultItem.querySelector('img').src;
        const status = resultItem.querySelector('.result-status').textContent;
        const confidence = resultItem.querySelector('.result-confidence').textContent;

        // Show details in a modal or expand the result item
        showToast(`Viewing details for ${status} result`, 'info');
    };

    // Delete result
    window.deleteResult = function(button) {
        const resultItem = button.closest('.result-item');
        resultItem.remove();
        showToast('Result deleted', 'success');
    };

    // Toast notification
    function showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast align-items-center text-white bg-${type === 'error' ? 'danger' : type} border-0`;
        toast.setAttribute('role', 'alert');
        toast.setAttribute('aria-live', 'assertive');
        toast.setAttribute('aria-atomic', 'true');

        toast.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        `;

        document.body.appendChild(toast);
        const bsToast = new bootstrap.Toast(toast);
        bsToast.show();

        toast.addEventListener('hidden.bs.toast', () => {
            toast.remove();
        });
    }
}); 