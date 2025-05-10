const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');
const uploadIcon = document.getElementById('uploadIcon');
const previewContainer = document.getElementById('previewContainer');
const imagePreview = document.getElementById('imagePreview');
const removeBtn = document.getElementById('removeBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultCard = document.getElementById('resultCard');
const resultIcon = document.getElementById('resultIcon');
const predictionText = document.getElementById('predictionText');
const confidenceText = document.getElementById('confidenceText');

let selectedFile = null;

dropZone.addEventListener('dragover', handleDragOver);
dropZone.addEventListener('dragleave', handleDragLeave);
dropZone.addEventListener('drop', handleDrop);
browseBtn.removeEventListener('click', handleBrowseClick); 
browseBtn.addEventListener('click', handleBrowseClick, false);
fileInput.removeEventListener('change', handleFileSelect);
fileInput.addEventListener('change', handleFileSelect, false);
removeBtn.addEventListener('click', resetUpload);
analyzeBtn.addEventListener('click', analyzeImage);

function handleBrowseClick(e) {
    e.preventDefault();
    e.stopPropagation(); 
    fileInput.click();
}

function handleDragOver(e) {
    e.preventDefault();
    dropZone.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    
    const files = e.dataTransfer.files;
    handleFiles(files);
}

function handleFileSelect(e) {
    e.preventDefault(); 
    const files = e.target.files;
    if (files && files.length > 0) {
        handleFiles(files);
    }
}

function handleFiles(files) {
    if (files.length === 0) return;
    
    const file = files[0];
    
    if (selectedFile && selectedFile.name === file.name) {
        return;
    }

    if (!isValidImageFile(file)) {
        showError('Please upload a valid image file (JPG, JPEG, or PNG)');
        return;
    }

    selectedFile = file; 
    displayImagePreview(file);
}

function isValidImageFile(file) {
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
    return validTypes.includes(file.type);
}

function displayImagePreview(file) {
    const reader = new FileReader();
    
    reader.onload = function(e) {
        imagePreview.src = e.target.result;
        dropZone.style.display = 'none';
        previewContainer.hidden = false;
        analyzeBtn.disabled = false;
    };
    
    reader.readAsDataURL(file);
}

function resetUpload() {
    fileInput.value = '';
    selectedFile = null;
    imagePreview.src = '';
    dropZone.style.display = 'block';
    previewContainer.hidden = true;
    analyzeBtn.disabled = true;
    resultCard.hidden = true;
}

async function analyzeImage() {
    analyzeBtn.disabled = true;
    const btnText = analyzeBtn.querySelector('.btn-text');
    const spinner = analyzeBtn.querySelector('.spinner');
    btnText.textContent = 'Analyzing...';
    spinner.hidden = false;

    try {
        if (!selectedFile) {
            showError('No file selected.');
            return;
        }

        const formData = new FormData();
        formData.append('file', selectedFile);

        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Prediction failed');

        const data = await response.json();

        resultIcon.textContent = data.prediction === 'PNEUMONIA' ? '⚠️' : '✅';  // Match exact string
        predictionText.textContent = data.prediction === 'PNEUMONIA' ? 'Pneumonia Detected' : 'Normal';
        confidenceText.textContent = `${data.confidence}%`;
        predictionText.className = data.prediction === 'PNEUMONIA' ? 'text-error' : 'text-success';
        resultCard.hidden = false;

    } catch (error) {
        console.error(error);
        showError('An error occurred during analysis. Please try again.');
    } finally {
        btnText.textContent = '🔍 Analyze X-ray';
        spinner.hidden = true;
        analyzeBtn.disabled = false;
    }
}

function showError(message) {
    alert(message); 
}
