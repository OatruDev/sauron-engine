import os
import pickle
import time
import json
import numpy as np
import faiss
import gradio as gr
from sentence_transformers import SentenceTransformer
from PIL import Image
import pandas as pd
from datetime import datetime

# --- CONFIGURACIÓN DE ARCHIVOS ---
DB_PKL = 'sauron_database.pkl'
CSV_LOG = 'bulk_fast_scan.csv'
RLHF_DIR = 'dataset_rlhf' # Nueva carpeta para guardar las fotos reales

# Crear directorio para las imágenes de entrenamiento si no existe
os.makedirs(RLHF_DIR, exist_ok=True)

print("🧠 Cargando Cerebro Visual (CLIP)...")
model = SentenceTransformer('sentence-transformers/clip-ViT-B-32')

print(f"📦 Cargando Base de Datos Local ({DB_PKL})...")
if not os.path.exists(DB_PKL):
    raise FileNotFoundError(f"❌ No encuentro {DB_PKL}.")

with open(DB_PKL, 'rb') as f:
    cards_data = pickle.load(f)

print("⚡ Construyendo Índice FAISS en RAM...")
processed_embeddings = []
for c in cards_data:
    emb = c['embedding']
    if isinstance(emb, str):
        try:
            emb = json.loads(emb)
        except:
            emb = emb.strip("[]").split(",")
            emb = [float(x) for x in emb]
    processed_embeddings.append(emb)

embeddings_matrix = np.array(processed_embeddings).astype('float32')
faiss.normalize_L2(embeddings_matrix) 

dimension = embeddings_matrix.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings_matrix)
print(f"✅ Listo. {index.ntotal} cartas cargadas en RAM.")

# Variable global para retener la imagen actual y sus datos
estado_captura = {"imagen": None, "name": "", "set": ""}

def scan_realtime(imagen):
    global estado_captura
    if imagen is None: return "Esperando video...", ""

    t_start = time.time()
    query_vector = model.encode(imagen).astype('float32').reshape(1, -1)
    faiss.normalize_L2(query_vector)

    D, I = index.search(query_vector, 1)
    t_end = time.time()
    latencia_ms = (t_end - t_start) * 1000

    if D[0][0] < 0.65: # Umbral ligeramente más bajo para fotos reales
        return "Buscando... (Pon la carta frente a la cámara)", ""

    card = cards_data[I[0][0]]
    # Guardamos la imagen en memoria temporalmente por si el usuario la confirma
    estado_captura = {"imagen": imagen, "name": card['name'], "set": card['set_code']}
    
    html_output = f"""
    <div style='text-align:center; background-color: #1a1a1a; color: white; padding: 10px; border-radius: 10px;'>
        <h2 style='margin:0;'>{card['name']}</h2>
        <p style='margin:0; color: #aaa;'>Set: {card['set_code'].upper()}</p>
        <p style='margin:0; color: #4caf50;'>Confianza: {D[0][0]*100:.1f}% | Temp: {latencia_ms:.0f}ms</p>
    </div>
    """
    return html_output, f"{card['name']} [{card['set_code'].upper()}]"

def confirmar_carta():
    global estado_captura
    if not estado_captura["name"] or estado_captura["imagen"] is None: 
        return "❌ Nada que guardar"
    
    name = estado_captura["name"]
    set_code = estado_captura["set"]
    fecha_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. GUARDAR LA IMAGEN REAL PARA EL RLHF
    nombre_archivo = f"{set_code}_{name.replace(' ', '_').replace('/', '-')}_{fecha_str}.jpg"
    ruta_imagen = os.path.join(RLHF_DIR, nombre_archivo)
    estado_captura["imagen"].save(ruta_imagen)
    
    # 2. GUARDAR EN EL CSV
    nuevo_registro = pd.DataFrame([[datetime.now().strftime("%Y-%m-%d %H:%M:%S"), name, set_code, ruta_imagen]], columns=["Fecha", "Nombre", "Set", "Ruta_Imagen"])
    nuevo_registro.to_csv(CSV_LOG, mode='a', index=False, header=not os.path.exists(CSV_LOG))
    
    return f"✅ Guardado: {name} (Foto capturada para RLHF)"

# --- INTERFAZ UI ---
with gr.Blocks(title="SAURON Bulk Scanner") as demo:
    gr.Markdown("# 👁️ SAURON - Escáner Bulk (Fase Recolección RLHF)")
    
    with gr.Row():
        # source="webcam" fuerza la cámara. 
        input_img = gr.Image(sources=["webcam"], streaming=True, type="pil", label="Cámara del Celular")
    
    with gr.Row():
        with gr.Column(scale=2):
            output_html = gr.HTML(label="Resultado")
        with gr.Column(scale=1):
            lbl_confirmacion = gr.Label(value="Esperando...", label="Última Guardada")
            btn_confirmar = gr.Button("💾 CONFIRMAR Y GUARDAR FOTO", variant="success")

    input_img.stream(fn=scan_realtime, inputs=[input_img], outputs=[output_html, lbl_confirmacion], queue=False)
    btn_confirmar.click(fn=confirmar_carta, inputs=[], outputs=[lbl_confirmacion])

if __name__ == "__main__":
    # CRÍTICO: share=True genera un enlace HTTPS seguro.
    # Esto evita que Chrome en Android bloquee el acceso a la cámara trasera.
    demo.launch(share=True)