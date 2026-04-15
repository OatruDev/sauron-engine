import os
import time
import torch
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from transformers import CLIPVisionModelWithProjection
import warnings

warnings.filterwarnings("ignore")

# Rutas de salida
ONNX_FP32_PATH = "sauron_vision_fp32.onnx"
ONNX_INT8_PATH = "sauron_vision_int8.onnx"

print("==================================================")
print("🧙‍♂️ SAURON EDGE COMPILER - FASE 2")
print("==================================================")

# 1. Cargar solo la parte visual (ahorramos 50% de peso de golpe)
print("\n📦 1. Descargando el Hemisferio Visual (CLIP Vision)...")
# Usamos el modelo base compatible con sentence-transformers
model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

# 2. Crear un tensor "falso" (Dummy) del tamaño exacto que usará el celular
# CLIP usa imágenes de 224x224 con 3 canales (RGB)
print("📏 2. Creando tensor de calibración (Batch: 1, Canales: 3, Tamaño: 224x224)...")
dummy_input = torch.randn(1, 3, 224, 224)

# 3. Exportar a ONNX (Formato universal C++)
print(f"⚙️ 3. Exportando modelo a ONNX FP32 -> {ONNX_FP32_PATH}")
torch.onnx.export(
    model,
    dummy_input,
    ONNX_FP32_PATH,
    export_params=True,
    opset_version=14,          # Opset estable para móviles
    do_constant_folding=True,  # Optimización matemática
    input_names=['pixel_values'],
    output_names=['image_embeds'],
    dynamic_axes={
        'pixel_values': {0: 'batch_size'},
        'image_embeds': {0: 'batch_size'}
    }
)

peso_fp32 = os.path.getsize(ONNX_FP32_PATH) / (1024 * 1024)
print(f"✅ Exportación completada. Peso actual: {peso_fp32:.1f} MB")

# 4. Cuantización a INT8 (La magia negra de compresión)
print(f"\n🗜️ 4. Iniciando Cuantización INT8 -> {ONNX_INT8_PATH}")
print("   (Convirtiendo pesos de 32-bits a 8-bits...)")

t_start = time.time()
quantize_dynamic(
    model_input=ONNX_FP32_PATH,
    model_output=ONNX_INT8_PATH,
    weight_type=QuantType.QUInt8,
    optimize_model=True
)
t_end = time.time()

peso_int8 = os.path.getsize(ONNX_INT8_PATH) / (1024 * 1024)
reduccion = (1 - (peso_int8 / peso_fp32)) * 100

print(f"✅ Cuantización exitosa en {t_end - t_start:.1f} segundos.")
print("==================================================")
print("📊 REPORTE DE COMPRESIÓN:")
print(f"   Modelo Original (FP32) : {peso_fp32:.1f} MB")
print(f"   Modelo Móvil (INT8)    : {peso_int8:.1f} MB")
print(f"   Reducción de tamaño    : {reduccion:.1f} %")
print("==================================================")
print("🚀 SAURON está listo para ser inyectado en ManaFox (Android/iOS).")