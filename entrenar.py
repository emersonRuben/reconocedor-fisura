"""
Entrenamiento de modelo CNN para detección de fisuras en concreto
Usando PyTorch + CUDA

Autor: Mael
Dataset: Imágenes de concreto (Positive=con fisuras, Negative=sin fisuras)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
CONFIG = {
    'data_dir': './concreto',           # Directorio del dataset
    'batch_size': 32,                    # Tamaño del batch
    'num_epochs': 10,                    # Número de épocas
    'learning_rate': 0.001,              # Tasa de aprendizaje
    'train_split': 0.8,                  # 80% para entrenamiento, 20% para validación
    'image_size': 224,                   # Tamaño de entrada (224x224 para ResNet)
    'num_workers': 4,                    # Workers para cargar datos
    'model_save_path': './modelo_fisuras.pth',  # Ruta para guardar el modelo
}

# ============================================================================
# VERIFICAR CUDA
# ============================================================================
def setup_device():
    """Configura el dispositivo (GPU o CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("=" * 60)
        print("🚀 USANDO GPU CON CUDA")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memoria: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print("=" * 60)
    else:
        device = torch.device("cpu")
        print("⚠️ CUDA no disponible, usando CPU (será más lento)")
    return device

# ============================================================================
# TRANSFORMACIONES DE DATOS
# ============================================================================
def get_transforms():
    """Define las transformaciones para entrenamiento y validación"""
    
    # Transformaciones para entrenamiento (con data augmentation)
    train_transform = transforms.Compose([
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # Valores de ImageNet
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Transformaciones para validación (sin augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return train_transform, val_transform

# ============================================================================
# CARGAR DATASET
# ============================================================================
def load_data():
    """Carga y divide el dataset en entrenamiento y validación"""
    print("\n📂 Cargando dataset...")
    
    train_transform, val_transform = get_transforms()
    
    # Cargar el dataset completo
    full_dataset = datasets.ImageFolder(root=CONFIG['data_dir'])
    
    # Mostrar información del dataset
    print(f"   Total de imágenes: {len(full_dataset)}")
    print(f"   Clases: {full_dataset.classes}")
    print(f"   Mapeo de clases: {full_dataset.class_to_idx}")
    
    # Dividir en train y val
    train_size = int(CONFIG['train_split'] * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Para reproducibilidad
    )
    
    # Aplicar transformaciones
    # Necesitamos crear datasets separados con las transformaciones correctas
    train_dataset_with_transform = datasets.ImageFolder(
        root=CONFIG['data_dir'], 
        transform=train_transform
    )
    val_dataset_with_transform = datasets.ImageFolder(
        root=CONFIG['data_dir'], 
        transform=val_transform
    )
    
    # Usar los mismos índices del split
    train_dataset_with_transform = torch.utils.data.Subset(
        train_dataset_with_transform, 
        train_dataset.indices
    )
    val_dataset_with_transform = torch.utils.data.Subset(
        val_dataset_with_transform, 
        val_dataset.indices
    )
    
    print(f"   Imágenes de entrenamiento: {len(train_dataset_with_transform)}")
    print(f"   Imágenes de validación: {len(val_dataset_with_transform)}")
    
    # Crear DataLoaders
    train_loader = DataLoader(
        train_dataset_with_transform,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=True  # Mejora rendimiento con GPU
    )
    
    val_loader = DataLoader(
        val_dataset_with_transform,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader, full_dataset.classes

# ============================================================================
# CREAR MODELO
# ============================================================================
def create_model(num_classes=2, pretrained=True):
    """Crea el modelo usando Transfer Learning con ResNet18"""
    print("\n🧠 Creando modelo...")
    
    # Cargar ResNet18 preentrenado en ImageNet
    if pretrained:
        print("   Usando Transfer Learning (ResNet18 preentrenado)")
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        print("   Creando ResNet18 desde cero (sin pesos preentrenados)")
        model = models.resnet18(weights=None)
    
    # Congelar las capas del modelo base (opcional, para fine-tuning rápido)
    # for param in model.parameters():
    #     param.requires_grad = False
    
    # Modificar la última capa para clasificación binaria
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),  # Regularización
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    
    print(f"   Clases de salida: {num_classes}")
    
    return model

# ============================================================================
# ENTRENAMIENTO
# ============================================================================
def train_model(model, train_loader, val_loader, device, classes):
    """Entrena el modelo"""
    print("\n🏋️ Iniciando entrenamiento...")
    print(f"   Épocas: {CONFIG['num_epochs']}")
    print(f"   Batch size: {CONFIG['batch_size']}")
    print(f"   Learning rate: {CONFIG['learning_rate']}")
    
    # Mover modelo a GPU
    model = model.to(device)
    
    # Definir loss function y optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    # Scheduler para reducir learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    
    # Historial para gráficas
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    
    for epoch in range(CONFIG['num_epochs']):
        print(f"\n{'='*60}")
        print(f"ÉPOCA {epoch+1}/{CONFIG['num_epochs']}")
        print(f"{'='*60}")
        
        # ================== ENTRENAMIENTO ==================
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc=f"Entrenando", leave=False)
        for images, labels in train_bar:
            # Mover datos a GPU
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Estadísticas
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Actualizar barra de progreso
            train_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100 * train_correct / train_total:.2f}%"
            })
        
        train_loss = train_loss / train_total
        train_acc = 100 * train_correct / train_total
        
        # ================== VALIDACIÓN ==================
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Validando", leave=False)
            for images, labels in val_bar:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / val_total
        val_acc = 100 * val_correct / val_total
        
        # Actualizar scheduler
        scheduler.step(val_loss)
        
        # Guardar historial
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Mostrar resultados de la época
        print(f"\n📊 Resultados Época {epoch+1}:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Guardar mejor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'classes': classes
            }, CONFIG['model_save_path'])
            print(f"   ✅ Mejor modelo guardado! (Val Acc: {val_acc:.2f}%)")
    
    return model, history

# ============================================================================
# VISUALIZACIÓN
# ============================================================================
def plot_training_history(history):
    """Genera gráficas del entrenamiento"""
    print("\n📈 Generando gráficas de entrenamiento...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfica de Loss
    axes[0].plot(history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_title('Pérdida durante el Entrenamiento', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Gráfica de Accuracy
    axes[1].plot(history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    axes[1].plot(history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
    axes[1].set_title('Precisión durante el Entrenamiento', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("   Gráfica guardada como 'training_history.png'")

# ============================================================================
# EVALUACIÓN DETALLADA
# ============================================================================
def evaluate_model(model, val_loader, device, classes):
    """Evaluación detallada del modelo con matriz de confusión"""
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    
    print("\n🔍 Evaluación detallada del modelo...")
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluando"):
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Reporte de clasificación
    print("\n" + "="*60)
    print("REPORTE DE CLASIFICACIÓN")
    print("="*60)
    print(classification_report(all_labels, all_preds, target_names=classes))
    
    # Matriz de confusión
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Matriz de Confusión', fontsize=14, fontweight='bold')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("   Matriz guardada como 'confusion_matrix.png'")

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================
def main():
    """Función principal de entrenamiento"""
    print("\n" + "="*60)
    print("🏗️ DETECCIÓN DE FISURAS EN CONCRETO")
    print("   Usando PyTorch + CUDA")
    print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # 1. Configurar dispositivo
    device = setup_device()
    
    # 2. Cargar datos
    train_loader, val_loader, classes = load_data()
    
    # 3. Crear modelo
    model = create_model(num_classes=len(classes))
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parámetros totales: {total_params:,}")
    print(f"   Parámetros entrenables: {trainable_params:,}")
    
    # 4. Entrenar
    model, history = train_model(model, train_loader, val_loader, device, classes)
    
    # 5. Visualizar resultados
    plot_training_history(history)
    
    # 6. Evaluación detallada
    try:
        evaluate_model(model, val_loader, device, classes)
    except ImportError:
        print("\n⚠️ Para ver la matriz de confusión, instala: pip install scikit-learn seaborn")
    
    print("\n" + "="*60)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print(f"   Modelo guardado en: {CONFIG['model_save_path']}")
    print("="*60)

# ============================================================================
# EJECUTAR
# ============================================================================
if __name__ == "__main__":
    main()
