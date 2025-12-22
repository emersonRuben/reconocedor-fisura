"""
Predicción de fisuras en concreto
Usando el modelo entrenado con PyTorch + CUDA

Modos de uso:
1. Predecir una imagen: python predecir.py --imagen ruta/imagen.jpg
2. Tiempo real con cámara: python predecir.py --camara
3. Predecir carpeta de imágenes: python predecir.py --carpeta ruta/carpeta
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np
import argparse
import os
from pathlib import Path

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
MODEL_PATH = './modelo_fisuras.pth'
IMAGE_SIZE = 224
CLASSES = ['Sin Fisura ✓', 'Con Fisura ✗']
COLORS = [(0, 255, 0), (0, 0, 255)]  # Verde para sin fisura, Rojo para con fisura

# ============================================================================
# CARGAR MODELO
# ============================================================================
def load_model(model_path, device):
    """Carga el modelo entrenado"""
    print(f"📂 Cargando modelo desde: {model_path}")
    
    # Crear arquitectura del modelo
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
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Modelo cargado (Val Acc: {checkpoint['val_acc']:.2f}%)")
    return model

# ============================================================================
# TRANSFORMACIONES
# ============================================================================
def get_transform():
    """Transformaciones para la predicción"""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

# ============================================================================
# PREDICCIÓN DE UNA IMAGEN
# ============================================================================
def predict_image(model, image_path, device, transform):
    """Predice si una imagen tiene fisura o no"""
    # Cargar imagen
    image = Image.open(image_path).convert('RGB')
    
    # Aplicar transformaciones
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predicción
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    class_idx = predicted.item()
    conf = confidence.item() * 100
    
    return class_idx, conf, CLASSES[class_idx]

# ============================================================================
# PREDICCIÓN EN TIEMPO REAL
# ============================================================================
def realtime_prediction(model, device, transform, camera_id=0):
    """Predicción en tiempo real usando la cámara"""
    print("\n🎥 Iniciando cámara...")
    print("   Presiona 'q' para salir")
    print("   Presiona 's' para guardar captura")
    
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print("❌ Error: No se pudo abrir la cámara")
        return
    
    # Configurar resolución
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convertir BGR a RGB para PIL
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Predicción
        input_tensor = transform(pil_image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        class_idx = predicted.item()
        conf = confidence.item() * 100
        label = CLASSES[class_idx]
        color = COLORS[class_idx]
        
        # Dibujar resultado en el frame
        # Fondo semi-transparente para el texto
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 100), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Texto con resultado
        cv2.putText(frame, f"Resultado: {label}", (20, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Confianza: {conf:.1f}%", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Borde de color según predicción
        cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), color, 4)
        
        # Mostrar frame
        cv2.imshow('Deteccion de Fisuras - Presiona Q para salir', frame)
        
        # Controles
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f'captura_{frame_count}.jpg'
            cv2.imwrite(filename, frame)
            print(f"   📸 Captura guardada: {filename}")
            frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Cámara cerrada")

# ============================================================================
# PREDICCIÓN DE CARPETA
# ============================================================================
def predict_folder(model, folder_path, device, transform):
    """Predice todas las imágenes en una carpeta"""
    print(f"\n📁 Procesando carpeta: {folder_path}")
    
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    images = [f for f in Path(folder_path).iterdir() 
              if f.suffix.lower() in extensions]
    
    if not images:
        print("❌ No se encontraron imágenes en la carpeta")
        return
    
    print(f"   Encontradas {len(images)} imágenes\n")
    
    results = {'positive': 0, 'negative': 0}
    
    for img_path in images:
        class_idx, conf, label = predict_image(model, str(img_path), device, transform)
        
        if class_idx == 1:
            results['positive'] += 1
            print(f"   🔴 {img_path.name}: {label} ({conf:.1f}%)")
        else:
            results['negative'] += 1
            print(f"   🟢 {img_path.name}: {label} ({conf:.1f}%)")
    
    print(f"\n📊 Resumen:")
    print(f"   Sin fisuras: {results['negative']}")
    print(f"   Con fisuras: {results['positive']}")

# ============================================================================
# VISUALIZAR PREDICCIÓN DE UNA IMAGEN
# ============================================================================
def visualize_prediction(image_path, class_idx, confidence, label):
    """Muestra la imagen con la predicción"""
    import matplotlib.pyplot as plt
    
    img = Image.open(image_path)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)
    
    color = 'green' if class_idx == 0 else 'red'
    ax.set_title(f'{label}\nConfianza: {confidence:.1f}%', 
                 fontsize=16, fontweight='bold', color=color)
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Predicción de fisuras en concreto')
    parser.add_argument('--imagen', type=str, help='Ruta a una imagen para predecir')
    parser.add_argument('--carpeta', type=str, help='Ruta a carpeta con imágenes')
    parser.add_argument('--camara', action='store_true', help='Usar cámara en tiempo real')
    parser.add_argument('--camara-id', type=int, default=0, help='ID de la cámara (default: 0)')
    parser.add_argument('--modelo', type=str, default=MODEL_PATH, help='Ruta al modelo')
    
    args = parser.parse_args()
    
    # Verificar que se especificó un modo
    if not any([args.imagen, args.carpeta, args.camara]):
        print("❌ Debes especificar un modo:")
        print("   --imagen ruta/imagen.jpg")
        print("   --carpeta ruta/carpeta")
        print("   --camara")
        return
    
    # Configurar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️ Usando: {'GPU (CUDA)' if device.type == 'cuda' else 'CPU'}")
    
    # Verificar modelo
    if not os.path.exists(args.modelo):
        print(f"❌ Error: No se encontró el modelo en {args.modelo}")
        return
    
    # Cargar modelo
    model = load_model(args.modelo, device)
    transform = get_transform()
    
    # Ejecutar modo seleccionado
    if args.imagen:
        if not os.path.exists(args.imagen):
            print(f"❌ Error: No se encontró la imagen {args.imagen}")
            return
        
        class_idx, conf, label = predict_image(model, args.imagen, device, transform)
        
        print(f"\n{'='*50}")
        print(f"📷 Imagen: {args.imagen}")
        print(f"🔍 Resultado: {label}")
        print(f"📊 Confianza: {conf:.1f}%")
        print(f"{'='*50}")
        
        # Mostrar imagen (opcional)
        try:
            visualize_prediction(args.imagen, class_idx, conf, label)
        except:
            pass
    
    elif args.carpeta:
        if not os.path.isdir(args.carpeta):
            print(f"❌ Error: No se encontró la carpeta {args.carpeta}")
            return
        predict_folder(model, args.carpeta, device, transform)
    
    elif args.camara:
        realtime_prediction(model, device, transform, args.camara_id)

if __name__ == "__main__":
    main()
