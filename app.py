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

def load_models() -> list:
    model_paths = {
        'densenet': 'best_densenet_model.pth',
        'mobilenet': 'Best_mobilenet.pth',
        'efficientnet': 'best_efficientnet_model.pth',
        'vgg': 'best_vgg16_model.pth'
    }

    models_list = []

    model_path = os.path.join(os.path.dirname(__file__), model_paths['densenet'])
    model = models.densenet121(pretrained=False)
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(model.classifier.in_features, 1)
    )
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    models_list.append(model)

    model_path = os.path.join(os.path.dirname(__file__), model_paths['mobilenet'])
    model = models.mobilenet_v3_small(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.features[11:13].named_parameters():
        param.requires_grad = True
    for name, param in model.features[10].named_parameters():
        if 'bn' in name:
            param.requires_grad = True
    num_features = model.classifier[0].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(num_features, 1)
    )
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    models_list.append(model)

    model_path = os.path.join(os.path.dirname(__file__), model_paths['efficientnet'])
    model = models.efficientnet_b0(pretrained=False)
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.4),
        torch.nn.Linear(model.classifier[1].in_features, 1)
    )
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    models_list.append(model)

    model_path = os.path.join(os.path.dirname(__file__), model_paths['vgg'])
    model = models.vgg16_bn(pretrained=False)
    model.classifier[6] = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(model.classifier[6].in_features, 1)
    )
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    models_list.append(model)

    return models_list

def ensemble_voting(models_list, img):
    predictions = []
    probabilities = []
    
    for model in models_list:
        with torch.no_grad():
            output = model(img)
            probability = torch.sigmoid(output).item()
            prediction = int(probability > 0.5)
            predictions.append(prediction)
            probabilities.append(probability if prediction == 1 else 1 - probability)
    
    hard_pred = round(sum(predictions) / len(predictions))
    confidence = sum(probabilities) / len(probabilities)
    confidence = round(confidence * 100, 2)
    
    return hard_pred, confidence

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

models_list = load_models()

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

        prediction, confidence = ensemble_voting(models_list, img)
        result = "PNEUMONIA" if prediction == 1 else "NORMAL"

        return jsonify({
            'prediction': result,
            'confidence': confidence
        }), 200

    except Exception as e:
        return jsonify({'error': f"Prediction failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5126)
