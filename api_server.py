"""
API para Detección de Fisuras en Concreto
Servidor Flask con soporte para:
- Predicción de imágenes (POST /predict)
- Streaming en tiempo real via WebSocket
- Interfaz web móvil-friendly

Uso:
    python api_server.py
    
Luego accede desde tu celular a: http://TU_IP:5000
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import base64
import os

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
app = Flask(__name__)
CORS(app)  # Permitir peticiones desde cualquier origen

MODEL_PATH = './modelo_fisuras.pth'
IMAGE_SIZE = 224
CLASSES = ['Sin Fisura', 'Con Fisura']

# Variables globales del modelo
model = None
device = None
transform = None

# ============================================================================
# CARGAR MODELO
# ============================================================================
def load_model():
    global model, device, transform
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Usando: {'GPU (CUDA)' if device.type == 'cuda' else 'CPU'}")
    
    # Transformaciones
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Crear arquitectura
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2)
    )
    
    # Cargar pesos
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Modelo cargado (Val Acc: {checkpoint['val_acc']:.2f}%)")

# ============================================================================
# PREDICCIÓN
# ============================================================================
def predict_image(image):
    """Realiza la predicción en una imagen PIL"""
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    return {
        'class_id': predicted.item(),
        'class_name': CLASSES[predicted.item()],
        'confidence': round(confidence.item() * 100, 2),
        'has_crack': predicted.item() == 1
    }

# ============================================================================
# PÁGINA WEB PRINCIPAL (MÓVIL-FRIENDLY)
# ============================================================================
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>🔍 Detector de Fisuras</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: white;
            padding: 20px;
        }
        
        .container {
            max-width: 500px;
            margin: 0 auto;
        }
        
        h1 {
            text-align: center;
            font-size: 1.5rem;
            margin-bottom: 20px;
            background: linear-gradient(90deg, #00d2ff, #3a7bd5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .camera-section {
            background: rgba(255,255,255,0.1);
            border-radius: 20px;
            padding: 20px;
            margin-bottom: 20px;
        }
        
        #video, #preview {
            width: 100%;
            border-radius: 15px;
            display: block;
        }
        
        #preview { display: none; }
        
        .btn-group {
            display: flex;
            gap: 10px;
            margin-top: 15px;
        }
        
        .btn {
            flex: 1;
            padding: 15px;
            border: none;
            border-radius: 12px;
            font-size: 1rem;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .btn:active { transform: scale(0.95); }
        
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .btn-success {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
        }
        
        .btn-danger {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
        }
        
        .result-card {
            background: rgba(255,255,255,0.1);
            border-radius: 20px;
            padding: 25px;
            text-align: center;
            display: none;
        }
        
        .result-card.show { display: block; }
        
        .result-icon {
            font-size: 4rem;
            margin-bottom: 15px;
        }
        
        .result-text {
            font-size: 1.5rem;
            font-weight: bold;
            margin-bottom: 10px;
        }
        
        .confidence {
            font-size: 1rem;
            color: rgba(255,255,255,0.7);
        }
        
        .progress-bar {
            width: 100%;
            height: 8px;
            background: rgba(255,255,255,0.2);
            border-radius: 4px;
            margin-top: 15px;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s;
        }
        
        .progress-fill.green { background: linear-gradient(90deg, #11998e, #38ef7d); }
        .progress-fill.red { background: linear-gradient(90deg, #f093fb, #f5576c); }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        
        .loading.show { display: block; }
        
        .spinner {
            width: 40px;
            height: 40px;
            border: 4px solid rgba(255,255,255,0.3);
            border-top-color: #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        
        @keyframes spin { to { transform: rotate(360deg); } }
        
        .upload-section {
            background: rgba(255,255,255,0.05);
            border: 2px dashed rgba(255,255,255,0.3);
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            margin-top: 20px;
        }
        
        input[type="file"] { display: none; }
        
        .realtime-toggle {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            margin-top: 15px;
        }
        
        .toggle {
            width: 50px;
            height: 26px;
            background: rgba(255,255,255,0.2);
            border-radius: 13px;
            cursor: pointer;
            position: relative;
            transition: background 0.3s;
        }
        
        .toggle.active { background: #38ef7d; }
        
        .toggle::after {
            content: '';
            position: absolute;
            width: 22px;
            height: 22px;
            background: white;
            border-radius: 50%;
            top: 2px;
            left: 2px;
            transition: left 0.3s;
        }
        
        .toggle.active::after { left: 26px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏗️ Detector de Fisuras en Concreto</h1>
        
        <div class="camera-section">
            <video id="video" autoplay playsinline></video>
            <img id="preview" alt="Preview">
            
            <div class="btn-group">
                <button class="btn btn-primary" id="captureBtn">📸 Capturar</button>
                <button class="btn btn-success" id="analyzeBtn" style="display:none;">🔍 Analizar</button>
                <button class="btn btn-danger" id="retakeBtn" style="display:none;">🔄 Otra</button>
            </div>
            
            <div class="realtime-toggle">
                <span>Tiempo real</span>
                <div class="toggle" id="realtimeToggle"></div>
            </div>
        </div>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Analizando imagen...</p>
        </div>
        
        <div class="result-card" id="resultCard">
            <div class="result-icon" id="resultIcon"></div>
            <div class="result-text" id="resultText"></div>
            <div class="confidence" id="resultConfidence"></div>
            <div class="progress-bar">
                <div class="progress-fill" id="progressFill"></div>
            </div>
        </div>
        
        <div class="upload-section">
            <p>📁 O sube una imagen</p>
            <input type="file" id="fileInput" accept="image/*">
            <button class="btn btn-primary" onclick="document.getElementById('fileInput').click()" style="margin-top:10px;">
                Seleccionar archivo
            </button>
        </div>
    </div>
    
    <canvas id="canvas" style="display:none;"></canvas>
    
    <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const preview = document.getElementById('preview');
        const captureBtn = document.getElementById('captureBtn');
        const analyzeBtn = document.getElementById('analyzeBtn');
        const retakeBtn = document.getElementById('retakeBtn');
        const loading = document.getElementById('loading');
        const resultCard = document.getElementById('resultCard');
        const realtimeToggle = document.getElementById('realtimeToggle');
        const fileInput = document.getElementById('fileInput');
        
        let stream = null;
        let realtimeMode = false;
        let realtimeInterval = null;
        
        // Iniciar cámara
        async function startCamera() {
            try {
                stream = await navigator.mediaDevices.getUserMedia({
                    video: { facingMode: 'environment' }
                });
                video.srcObject = stream;
            } catch (err) {
                alert('No se pudo acceder a la cámara: ' + err.message);
            }
        }
        
        // Capturar frame
        function captureFrame() {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);
            return canvas.toDataURL('image/jpeg', 0.8);
        }
        
        // Analizar imagen
        async function analyzeImage(imageData) {
            loading.classList.add('show');
            resultCard.classList.remove('show');
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: imageData })
                });
                
                const result = await response.json();
                showResult(result);
            } catch (err) {
                alert('Error al analizar: ' + err.message);
            } finally {
                loading.classList.remove('show');
            }
        }
        
        // Mostrar resultado
        function showResult(result) {
            const icon = result.has_crack ? '⚠️' : '✅';
            const text = result.has_crack ? 'FISURA DETECTADA' : 'SIN FISURA';
            const color = result.has_crack ? 'red' : 'green';
            
            document.getElementById('resultIcon').textContent = icon;
            document.getElementById('resultText').textContent = text;
            document.getElementById('resultText').style.color = result.has_crack ? '#f5576c' : '#38ef7d';
            document.getElementById('resultConfidence').textContent = `Confianza: ${result.confidence}%`;
            
            const progressFill = document.getElementById('progressFill');
            progressFill.style.width = result.confidence + '%';
            progressFill.className = 'progress-fill ' + color;
            
            resultCard.classList.add('show');
        }
        
        // Modo captura
        captureBtn.addEventListener('click', () => {
            if (realtimeMode) return;
            
            const imageData = captureFrame();
            preview.src = imageData;
            preview.style.display = 'block';
            video.style.display = 'none';
            
            captureBtn.style.display = 'none';
            analyzeBtn.style.display = 'block';
            retakeBtn.style.display = 'block';
        });
        
        // Analizar
        analyzeBtn.addEventListener('click', () => {
            analyzeImage(preview.src);
        });
        
        // Retomar
        retakeBtn.addEventListener('click', () => {
            preview.style.display = 'none';
            video.style.display = 'block';
            
            captureBtn.style.display = 'block';
            analyzeBtn.style.display = 'none';
            retakeBtn.style.display = 'none';
            
            resultCard.classList.remove('show');
        });
        
        // Modo tiempo real
        realtimeToggle.addEventListener('click', () => {
            realtimeMode = !realtimeMode;
            realtimeToggle.classList.toggle('active');
            
            if (realtimeMode) {
                captureBtn.style.display = 'none';
                realtimeInterval = setInterval(() => {
                    const imageData = captureFrame();
                    analyzeImage(imageData);
                }, 1000); // Cada 1 segundo
            } else {
                captureBtn.style.display = 'block';
                clearInterval(realtimeInterval);
            }
        });
        
        // Subir archivo
        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (event) => {
                    preview.src = event.target.result;
                    preview.style.display = 'block';
                    video.style.display = 'none';
                    analyzeImage(event.target.result);
                };
                reader.readAsDataURL(file);
            }
        });
        
        // Iniciar
        startCamera();
    </script>
</body>
</html>
'''

# ============================================================================
# RUTAS DE LA API
# ============================================================================
@app.route('/')
def index():
    """Página principal con interfaz móvil"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint para predicción de imágenes"""
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No se proporcionó imagen'}), 400
        
        # Decodificar imagen base64
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Realizar predicción
        result = predict_image(image)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check"""
    return jsonify({
        'status': 'ok',
        'device': str(device),
        'model_loaded': model is not None
    })

# ============================================================================
# EJECUTAR SERVIDOR
# ============================================================================
if __name__ == '__main__':
    import socket
    
    # Obtener IP local
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    print("\n" + "="*60)
    print("🚀 SERVIDOR DE DETECCIÓN DE FISURAS")
    print("="*60)
    
    load_model()
    
    print("\n📱 Accede desde tu celular a:")
    print(f"   http://{local_ip}:5000")
    print(f"   http://localhost:5000 (local)")
    print("\n⚡ API Endpoints:")
    print("   POST /predict - Envía imagen para análisis")
    print("   GET /health - Estado del servidor")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
