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
RLHF_DIR = 'dataset_rlhf'

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

# --- ESTADO GLOBAL ---
estado_captura = {"imagen": None, "name": "", "set": ""}

def analizar_foto(imagen):
    global estado_captura
    # Si la imagen es None (porque la acabamos de limpiar), no hacemos nada
    if imagen is None: 
        return "Esperando captura...", "Cámara lista"

    t_start = time.time()
    query_vector = model.encode(imagen).astype('float32').reshape(1, -1)
    faiss.normalize_L2(query_vector)

    D, I = index.search(query_vector, 1)
    t_end = time.time()
    latencia_ms = (t_end - t_start) * 1000

    if D[0][0] < 0.65:
        estado_captura = {"imagen": None, "name": "", "set": ""}
        return "<h3 style='color:red; text-align:center;'>No estoy seguro. Descarta y toma otra foto.</h3>", "❌ Baja confianza"

    card = cards_data[I[0][0]]
    estado_captura = {"imagen": imagen, "name": card['name'], "set": card['set_code']}
    
    html_output = f"""
    <div style='text-align:center; background-color: #1a1a1a; color: white; padding: 10px; border-radius: 10px;'>
        <h2 style='margin:0;'>{card['name']}</h2>
        <p style='margin:0; color: #aaa;'>Set: {card['set_code'].upper()}</p>
        <p style='margin:0; color: #4caf50;'>Confianza: {D[0][0]*100:.1f}% | Temp: {latencia_ms:.0f}ms</p>
    </div>
    """
    return html_output, f"Detectado: {card['name']}"

def confirmar_carta():
    global estado_captura
    if not estado_captura["name"] or estado_captura["imagen"] is None: 
        # Retornamos gr.update() para no afectar la interfaz si no hay nada
        return "❌ Nada que guardar", gr.update(), gr.update()
    
    name = estado_captura["name"]
    set_code = estado_captura["set"]
    fecha_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Guardar foto RLHF
    nombre_archivo = f"{set_code}_{name.replace(' ', '_').replace('/', '-')}_{fecha_str}.jpg"
    ruta_imagen = os.path.join(RLHF_DIR, nombre_archivo)
    estado_captura["imagen"].save(ruta_imagen)
    
    # 2. Guardar en CSV
    nuevo_registro = pd.DataFrame([[datetime.now().strftime("%Y-%m-%d %H:%M:%S"), name, set_code, ruta_imagen]], columns=["Fecha", "Nombre", "Set", "Ruta_Imagen"])
    nuevo_registro.to_csv(CSV_LOG, mode='a', index=False, header=not os.path.exists(CSV_LOG))
    
    # Limpiamos el estado
    estado_captura = {"imagen": None, "name": "", "set": ""}
    
    # Retornamos el mensaje, LIMPIAMOS la imagen (None la reinicia) y limpiamos el HTML
    return f"✅ Guardado: {name}", None, "<h3 style='text-align:center;'>Foto guardada. Cámara reiniciada.</h3>"

def descartar_carta():
    global estado_captura
    estado_captura = {"imagen": None, "name": "", "set": ""}
    # Reiniciamos la cámara devolviendo None a la imagen
    return "🗑️ Descartada", None, "<h3 style='text-align:center;'>Descartado. Cámara reiniciada.</h3>"

# --- INTERFAZ UI ---
with gr.Blocks(title="SAURON RLHF") as demo:
    gr.Markdown("# 👁️ SAURON - Entrenamiento RLHF")
    
    with gr.Row():
        # Quitamos streaming=True. Ahora Gradio esperará a que el usuario "tome la foto"
        input_img = gr.Image(sources=["webcam"], type="pil", label="1. Encuadra la carta y toma la foto")
    
    with gr.Row():
        with gr.Column(scale=2):
            output_html = gr.HTML(value="<h3 style='text-align:center;'>Esperando foto...</h3>", label="2. Resultado")
        with gr.Column(scale=1):
            lbl_confirmacion = gr.Label(value="Esperando...", label="Estado")
            
            with gr.Row():
                btn_descartar = gr.Button("🗑️ DESCARTAR (Mal)", variant="secondary")
                btn_confirmar = gr.Button("💾 CONFIRMAR (Bien)", variant="success")

    # Cuando el usuario toma la foto, se ejecuta el análisis
    input_img.change(fn=analizar_foto, inputs=[input_img], outputs=[output_html, lbl_confirmacion])
    
    # Botones de acción. Ojo: devuelven None a input_img para volver a prender la cámara
    btn_confirmar.click(fn=confirmar_carta, inputs=[], outputs=[lbl_confirmacion, input_img, output_html])
    btn_descartar.click(fn=descartar_carta, inputs=[], outputs=[lbl_confirmacion, input_img, output_html])

if __name__ == "__main__":
    demo.launch(share=True)