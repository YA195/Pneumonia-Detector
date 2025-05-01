from flask import Flask, render_template, request, jsonify
import os
import io
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from typing import Dict, Any

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model() -> torch.nn.Module:
    model_path = os.path.join(os.path.dirname(__file__), 'Best_resnet50.pth')
    print(f"Loading model from: {model_path}")

    model = models.resnet50(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(model.fc.in_features, 1)
    )
    
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

try:
    model = load_model()
    print("Model loaded successfully")
except Exception as e:
    app.logger.error(f"Model loading failed: {e}")
    raise RuntimeError(f"Model loading failed: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_image() -> tuple[Dict[str, Any], int]:
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        img = Image.open(io.BytesIO(file.read()))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(img)
            probability = torch.sigmoid(output)
            prediction = (probability > 0.5).float()
            confidence = probability.item() if prediction.item() == 1 else 1 - probability.item()
            confidence = round(confidence * 100, 2)

        result = "PNEUMONIA" if prediction.item() == 1 else "NORMAL"

        print(f"Raw output: {output.item()}")
        print(f"Probability: {probability.item()}")
        print(f"Prediction: {result}")
        print(f"Confidence: {confidence}%")

        return jsonify({
            'prediction': result,
            'confidence': confidence
        }), 200

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': f"Prediction failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5126)
