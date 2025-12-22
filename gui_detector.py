"""
GUI para Detección de Fisuras en Concreto
Interfaz gráfica moderna usando CustomTkinter

Funcionalidades:
- Cargar y analizar imágenes
- Análisis en tiempo real con cámara
- Historial de predicciones
- Estadísticas
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import os
import threading
from datetime import datetime

# ============================================================================
# CONFIGURACIÓN DE TEMA
# ============================================================================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ============================================================================
# CONFIGURACIÓN DEL MODELO
# ============================================================================
MODEL_PATH = './modelo_fisuras.pth'
IMAGE_SIZE = 224
CLASSES = ['Sin Fisura', 'Con Fisura']

# ============================================================================
# CLASE PRINCIPAL DE LA APLICACIÓN
# ============================================================================
class FisuraDetectorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Configuración de la ventana
        self.title("🔍 Detector de Fisuras en Concreto")
        self.geometry("1100x750")
        self.minsize(900, 600)
        
        # Variables
        self.model = None
        self.device = None
        self.transform = None
        self.camera_running = False
        self.cap = None
        self.history = []
        
        # Inicializar modelo
        self.init_model()
        
        # Crear interfaz
        self.create_widgets()
        
    def init_model(self):
        """Inicializa el modelo y el dispositivo"""
        # Configurar dispositivo
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Transformaciones
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Cargar modelo si existe
        if os.path.exists(MODEL_PATH):
            self.load_model()
        
    def load_model(self):
        """Carga el modelo entrenado"""
        try:
            # Crear arquitectura
            self.model = models.resnet18(weights=None)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 2)
            )
            
            # Cargar pesos
            checkpoint = torch.load(MODEL_PATH, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()
            
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Error cargando modelo: {str(e)}")
            return False
    
    def create_widgets(self):
        """Crea todos los widgets de la interfaz"""
        # Grid principal
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        self.grid_rowconfigure(0, weight=1)
        
        # ==================== PANEL IZQUIERDO (Principal) ====================
        self.main_frame = ctk.CTkFrame(self, corner_radius=0)
        self.main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)
        
        # Header
        self.header_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        self.title_label = ctk.CTkLabel(
            self.header_frame, 
            text="🏗️ Detector de Fisuras en Concreto",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.title_label.pack(side="left", padx=10)
        
        # Estado del dispositivo
        device_text = "🚀 GPU (CUDA)" if self.device and self.device.type == "cuda" else "💻 CPU"
        self.device_label = ctk.CTkLabel(
            self.header_frame,
            text=device_text,
            font=ctk.CTkFont(size=14),
            text_color="#4CAF50" if "GPU" in device_text else "#FFA500"
        )
        self.device_label.pack(side="right", padx=10)
        
        # Área de imagen
        self.image_frame = ctk.CTkFrame(self.main_frame)
        self.image_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.image_frame.grid_columnconfigure(0, weight=1)
        self.image_frame.grid_rowconfigure(0, weight=1)
        
        self.image_label = ctk.CTkLabel(
            self.image_frame, 
            text="📷 Carga una imagen o inicia la cámara",
            font=ctk.CTkFont(size=16)
        )
        self.image_label.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        
        # Panel de resultado
        self.result_frame = ctk.CTkFrame(self.main_frame, height=120)
        self.result_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        self.result_frame.grid_columnconfigure(0, weight=1)
        self.result_frame.grid_propagate(False)
        
        self.result_label = ctk.CTkLabel(
            self.result_frame,
            text="Esperando análisis...",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.result_label.grid(row=0, column=0, pady=10)
        
        self.confidence_label = ctk.CTkLabel(
            self.result_frame,
            text="",
            font=ctk.CTkFont(size=14)
        )
        self.confidence_label.grid(row=1, column=0)
        
        # Barra de confianza
        self.confidence_bar = ctk.CTkProgressBar(self.result_frame, width=400)
        self.confidence_bar.grid(row=2, column=0, pady=10)
        self.confidence_bar.set(0)
        
        # ==================== PANEL DERECHO (Controles) ====================
        self.control_frame = ctk.CTkFrame(self, width=280)
        self.control_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)
        self.control_frame.grid_propagate(False)
        
        # Título controles
        ctk.CTkLabel(
            self.control_frame,
            text="⚙️ Controles",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=15)
        
        # Separador
        ctk.CTkFrame(self.control_frame, height=2, fg_color="gray50").pack(fill="x", padx=20, pady=5)
        
        # Botón cargar imagen
        self.btn_load = ctk.CTkButton(
            self.control_frame,
            text="📂 Cargar Imagen",
            command=self.load_image,
            height=45,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.btn_load.pack(pady=10, padx=20, fill="x")
        
        # Botón cargar carpeta
        self.btn_folder = ctk.CTkButton(
            self.control_frame,
            text="📁 Analizar Carpeta",
            command=self.analyze_folder,
            height=45,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#6B5B95"
        )
        self.btn_folder.pack(pady=10, padx=20, fill="x")
        
        # Separador
        ctk.CTkFrame(self.control_frame, height=2, fg_color="gray50").pack(fill="x", padx=20, pady=15)
        
        # Botón cámara
        self.btn_camera = ctk.CTkButton(
            self.control_frame,
            text="🎥 Iniciar Cámara",
            command=self.toggle_camera,
            height=50,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#2E8B57"
        )
        self.btn_camera.pack(pady=10, padx=20, fill="x")
        
        # Botón capturar
        self.btn_capture = ctk.CTkButton(
            self.control_frame,
            text="📸 Capturar Frame",
            command=self.capture_frame,
            height=40,
            state="disabled",
            fg_color="#FF6B6B"
        )
        self.btn_capture.pack(pady=5, padx=20, fill="x")
        
        # Separador
        ctk.CTkFrame(self.control_frame, height=2, fg_color="gray50").pack(fill="x", padx=20, pady=15)
        
        # Estadísticas
        ctk.CTkLabel(
            self.control_frame,
            text="📊 Estadísticas de Sesión",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=5)
        
        self.stats_frame = ctk.CTkFrame(self.control_frame, fg_color="transparent")
        self.stats_frame.pack(pady=5, padx=20, fill="x")
        
        self.stats_total = ctk.CTkLabel(self.stats_frame, text="Total: 0", font=ctk.CTkFont(size=12))
        self.stats_total.pack(anchor="w")
        
        self.stats_positive = ctk.CTkLabel(
            self.stats_frame, 
            text="Con fisura: 0", 
            font=ctk.CTkFont(size=12),
            text_color="#FF6B6B"
        )
        self.stats_positive.pack(anchor="w")
        
        self.stats_negative = ctk.CTkLabel(
            self.stats_frame, 
            text="Sin fisura: 0", 
            font=ctk.CTkFont(size=12),
            text_color="#4CAF50"
        )
        self.stats_negative.pack(anchor="w")
        
        # Botón limpiar
        self.btn_clear = ctk.CTkButton(
            self.control_frame,
            text="🗑️ Limpiar Estadísticas",
            command=self.clear_stats,
            height=35,
            fg_color="gray40"
        )
        self.btn_clear.pack(pady=15, padx=20, fill="x")
        
        # Info del modelo
        ctk.CTkFrame(self.control_frame, height=2, fg_color="gray50").pack(fill="x", padx=20, pady=10)
        
        model_status = "✅ Modelo cargado" if self.model else "❌ Modelo no encontrado"
        model_color = "#4CAF50" if self.model else "#FF6B6B"
        
        self.model_label = ctk.CTkLabel(
            self.control_frame,
            text=model_status,
            font=ctk.CTkFont(size=12),
            text_color=model_color
        )
        self.model_label.pack(pady=10)
    
    def predict(self, pil_image):
        """Realiza la predicción en una imagen PIL"""
        if self.model is None:
            return None, 0
        
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        return predicted.item(), confidence.item() * 100
    
    def update_result(self, class_idx, confidence):
        """Actualiza el panel de resultados"""
        if class_idx == 0:  # Sin fisura
            self.result_label.configure(
                text="✅ SIN FISURA",
                text_color="#4CAF50"
            )
            self.confidence_bar.configure(progress_color="#4CAF50")
        else:  # Con fisura
            self.result_label.configure(
                text="⚠️ FISURA DETECTADA",
                text_color="#FF6B6B"
            )
            self.confidence_bar.configure(progress_color="#FF6B6B")
        
        self.confidence_label.configure(text=f"Confianza: {confidence:.1f}%")
        self.confidence_bar.set(confidence / 100)
        
        # Actualizar estadísticas
        self.history.append({'class': class_idx, 'confidence': confidence})
        self.update_stats()
    
    def update_stats(self):
        """Actualiza las estadísticas"""
        total = len(self.history)
        positive = sum(1 for h in self.history if h['class'] == 1)
        negative = total - positive
        
        self.stats_total.configure(text=f"Total: {total}")
        self.stats_positive.configure(text=f"Con fisura: {positive}")
        self.stats_negative.configure(text=f"Sin fisura: {negative}")
    
    def clear_stats(self):
        """Limpia las estadísticas"""
        self.history = []
        self.update_stats()
        self.result_label.configure(text="Esperando análisis...", text_color="white")
        self.confidence_label.configure(text="")
        self.confidence_bar.set(0)
    
    def load_image(self):
        """Carga y analiza una imagen"""
        if self.model is None:
            messagebox.showerror("Error", "Modelo no cargado")
            return
        
        file_path = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[
                ("Imágenes", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("Todos los archivos", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Cargar imagen
                pil_image = Image.open(file_path).convert('RGB')
                
                # Mostrar imagen
                self.display_image(pil_image)
                
                # Predecir
                class_idx, confidence = self.predict(pil_image)
                self.update_result(class_idx, confidence)
                
            except Exception as e:
                messagebox.showerror("Error", f"Error cargando imagen: {str(e)}")
    
    def display_image(self, pil_image, max_size=500):
        """Muestra una imagen en el panel central"""
        # Redimensionar manteniendo aspecto
        ratio = min(max_size / pil_image.width, max_size / pil_image.height)
        new_size = (int(pil_image.width * ratio), int(pil_image.height * ratio))
        display_img = pil_image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convertir a CTkImage (pasar como light_image y dark_image)
        ctk_image = ctk.CTkImage(light_image=display_img, dark_image=display_img, size=new_size)
        self.image_label.configure(image=ctk_image, text="")
        self.image_label.image = ctk_image  # Mantener referencia
    
    def analyze_folder(self):
        """Analiza todas las imágenes de una carpeta"""
        if self.model is None:
            messagebox.showerror("Error", "Modelo no cargado")
            return
        
        folder_path = filedialog.askdirectory(title="Seleccionar carpeta")
        
        if folder_path:
            extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
            images = [f for f in os.listdir(folder_path) 
                     if os.path.splitext(f)[1].lower() in extensions]
            
            if not images:
                messagebox.showinfo("Info", "No se encontraron imágenes en la carpeta")
                return
            
            # Procesar imágenes
            results = {'positive': 0, 'negative': 0}
            
            for img_name in images:
                img_path = os.path.join(folder_path, img_name)
                pil_image = Image.open(img_path).convert('RGB')
                class_idx, confidence = self.predict(pil_image)
                
                if class_idx == 1:
                    results['positive'] += 1
                else:
                    results['negative'] += 1
                
                self.history.append({'class': class_idx, 'confidence': confidence})
            
            self.update_stats()
            
            messagebox.showinfo(
                "Análisis Completado",
                f"Total analizadas: {len(images)}\n"
                f"Con fisura: {results['positive']}\n"
                f"Sin fisura: {results['negative']}"
            )
    
    def toggle_camera(self):
        """Inicia o detiene la cámara"""
        if self.camera_running:
            self.stop_camera()
        else:
            self.start_camera()
    
    def start_camera(self):
        """Inicia la cámara"""
        if self.model is None:
            messagebox.showerror("Error", "Modelo no cargado")
            return
        
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            messagebox.showerror("Error", "No se pudo abrir la cámara")
            return
        
        self.camera_running = True
        self.btn_camera.configure(text="⏹️ Detener Cámara", fg_color="#DC143C")
        self.btn_capture.configure(state="normal")
        
        # Iniciar hilo de captura
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()
    
    def stop_camera(self):
        """Detiene la cámara"""
        self.camera_running = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.btn_camera.configure(text="🎥 Iniciar Cámara", fg_color="#2E8B57")
        self.btn_capture.configure(state="disabled")
        
        self.image_label.configure(
            image=None, 
            text="📷 Carga una imagen o inicia la cámara"
        )
    
    def camera_loop(self):
        """Loop de captura de la cámara"""
        while self.camera_running:
            ret, frame = self.cap.read()
            
            if not ret:
                break
            
            # Convertir BGR a RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Predecir
            class_idx, confidence = self.predict(pil_image)
            
            # Dibujar resultado en el frame
            color = (0, 255, 0) if class_idx == 0 else (255, 0, 0)
            label = CLASSES[class_idx]
            
            # Agregar texto y borde
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
            cv2.putText(frame, f"{label} - {confidence:.1f}%", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), color[::-1], 4)
            
            # Convertir de nuevo a RGB para mostrar
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Actualizar GUI (thread-safe)
            self.after(0, lambda img=pil_image, idx=class_idx, conf=confidence: 
                      self._update_camera_frame(img, idx, conf))
        
    def _update_camera_frame(self, pil_image, class_idx, confidence):
        """Actualiza el frame de la cámara en el hilo principal"""
        if self.camera_running:
            self.display_image(pil_image)
            
            # Actualizar resultado (sin agregar al historial en cada frame)
            if class_idx == 0:
                self.result_label.configure(text="✅ SIN FISURA", text_color="#4CAF50")
                self.confidence_bar.configure(progress_color="#4CAF50")
            else:
                self.result_label.configure(text="⚠️ FISURA DETECTADA", text_color="#FF6B6B")
                self.confidence_bar.configure(progress_color="#FF6B6B")
            
            self.confidence_label.configure(text=f"Confianza: {confidence:.1f}%")
            self.confidence_bar.set(confidence / 100)
    
    def capture_frame(self):
        """Captura el frame actual"""
        if self.cap and self.camera_running:
            ret, frame = self.cap.read()
            if ret:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"captura_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                
                # Agregar al historial
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                class_idx, confidence = self.predict(pil_image)
                self.history.append({'class': class_idx, 'confidence': confidence})
                self.update_stats()
                
                messagebox.showinfo("Captura", f"Imagen guardada: {filename}")
    
    def on_closing(self):
        """Maneja el cierre de la aplicación"""
        self.stop_camera()
        self.destroy()

# ============================================================================
# EJECUTAR APLICACIÓN
# ============================================================================
if __name__ == "__main__":
    app = FisuraDetectorApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
