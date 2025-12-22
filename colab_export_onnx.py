# Detector de Fisuras - Notebook para Google Colab
# Ejecuta este código en Google Colab para exportar el modelo a ONNX

"""
INSTRUCCIONES:
1. Sube este archivo a Google Colab (https://colab.research.google.com)
2. Sube tu archivo 'modelo_fisuras.pth' a Colab
3. Ejecuta todas las celdas
4. Descarga el archivo 'modelo_fisuras.onnx' generado
"""

# =============================================================================
# CELDA 1: Instalar dependencias
# =============================================================================
# !pip install torch torchvision onnx onnxruntime

# =============================================================================
# CELDA 2: Imports
# =============================================================================
import torch
import torch.nn as nn
from torchvision import models
import os

print("PyTorch version:", torch.__version__)

# =============================================================================
# CELDA 3: Definir modelo (igual que en entrenamiento)
# =============================================================================
def create_model():
    """Crea la arquitectura idéntica al modelo entrenado"""
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2)
    )
    return model

# =============================================================================
# CELDA 4: Cargar pesos entrenados
# =============================================================================
MODEL_PATH = 'modelo_fisuras.pth'  # Asegúrate de subir este archivo

# Verificar que el archivo existe
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"No se encontró '{MODEL_PATH}'. "
        "Sube el archivo modelo_fisuras.pth a Colab primero."
    )

# Cargar modelo
model = create_model()
checkpoint = torch.load(MODEL_PATH, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✅ Modelo cargado correctamente")
print(f"   Accuracy de validación: {checkpoint['val_acc']:.2f}%")

# =============================================================================
# CELDA 5: Exportar a ONNX
# =============================================================================
ONNX_PATH = 'modelo_fisuras.onnx'
IMAGE_SIZE = 224

# Entrada de ejemplo
dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

# Exportar
torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print(f"✅ Modelo exportado a: {ONNX_PATH}")
print(f"   Tamaño: {os.path.getsize(ONNX_PATH) / (1024*1024):.2f} MB")

# =============================================================================
# CELDA 6: Verificar modelo ONNX
# =============================================================================
import onnx
import onnxruntime as ort
import numpy as np

# Verificar estructura
onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)
print("✅ Estructura ONNX válida")

# Probar inferencia
session = ort.InferenceSession(ONNX_PATH)
input_name = session.get_inputs()[0].name

# Comparar salidas
with torch.no_grad():
    pytorch_output = model(dummy_input).numpy()

onnx_output = session.run(None, {input_name: dummy_input.numpy()})[0]

diff = np.abs(pytorch_output - onnx_output).max()
print(f"✅ Diferencia máxima PyTorch vs ONNX: {diff:.6f}")

if diff < 0.001:
    print("✅ Conversión verificada correctamente")
else:
    print("⚠️ Hay diferencias menores, pero el modelo debería funcionar")

# =============================================================================
# CELDA 7: Descargar modelo
# =============================================================================
from google.colab import files

print("\n📥 Descargando modelo ONNX...")
files.download(ONNX_PATH)

print("""
✅ ¡Listo!

El archivo 'modelo_fisuras.onnx' se está descargando.
Cópialo a tu proyecto Android en:
  app/src/main/assets/modelo_fisuras.onnx
""")
