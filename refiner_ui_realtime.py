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

# Archivos Locales
DB_PKL = 'sauron_database.pkl'
CSV_LOG = 'bulk_fast_scan.csv'

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

ultima_carta_detectada = {"name": "", "set": ""}

def scan_realtime(imagen):
    global ultima_carta_detectada
    if imagen is None: return "Esperando video...", ""

    t_start = time.time()
    query_vector = model.encode(imagen).astype('float32').reshape(1, -1)
    faiss.normalize_L2(query_vector)

    D, I = index.search(query_vector, 1)
    t_end = time.time()
    latencia_ms = (t_end - t_start) * 1000

    if D[0][0] < 0.70: 
        return "Buscando... (Pon la carta frente a la cámara)", ""

    card = cards_data[I[0][0]]
    ultima_carta_detectada = {"name": card['name'], "set": card['set_code']}
    
    html_output = f"""
    <div style='text-align:center; background-color: #1a1a1a; color: white; padding: 10px; border-radius: 10px;'>
        <h2 style='margin:0;'>{card['name']}</h2>
        <p style='margin:0; color: #aaa;'>Set: {card['set_code'].upper()}</p>
        <p style='margin:0; color: #4caf50;'>Confianza: {D[0][0]*100:.1f}% | Temp: {latencia_ms:.0f}ms</p>
        <img src='{card['image_url']}' style='max-width: 200px; margin-top:10px; border-radius:10px;'>
    </div>
    """
    return html_output, f"{card['name']} [{card['set_code'].upper()}]"

def confirmar_carta():
    global ultima_carta_detectada
    if not ultima_carta_detectada["name"]: return "❌ Nada que guardar"
    
    name = ultima_carta_detectada["name"]
    set_code = ultima_carta_detectada["set"]
    nuevo_registro = pd.DataFrame([[datetime.now().strftime("%Y-%m-%d %H:%M:%S"), name, set_code]], columns=["Fecha", "Nombre", "Set"])
    nuevo_registro.to_csv(CSV_LOG, mode='a', index=False, header=not os.path.exists(CSV_LOG))
    return f"✅ Guardado: {name} ({set_code})"

# --- INTERFAZ UI LIMPIA ---
with gr.Blocks(title="SAURON ManaBox Level") as demo:
    gr.Markdown("# 👁️ SAURON - Escáner Bulk Tiempo Real")
    
    with gr.Row():
        # Limpiamos los parámetros para máxima compatibilidad con Gradio v5+
        input_img = gr.Image(
            sources=["webcam"], 
            streaming=True, 
            type="pil", 
            label="Cámara en Vivo"
        )
    
    with gr.Row():
        with gr.Column(scale=2):
            output_html = gr.HTML(label="Resultado")
        with gr.Column(scale=1):
            lbl_confirmacion = gr.Label(value="Esperando...", label="Última Guardada")
            btn_confirmar = gr.Button("💾 GUARDAR CARTA", variant="success")

    # Conectamos el flujo de video al modelo
    input_img.stream(fn=scan_realtime, inputs=[input_img], outputs=[output_html, lbl_confirmacion], queue=False)
    btn_confirmar.click(fn=confirmar_carta, inputs=[], outputs=[lbl_confirmacion])

if __name__ == "__main__":
    custom_css = ".gradio-container {background-color: #111; color: white;}"
    # Lanzamiento
    demo.launch(server_name="0.0.0.0", server_port=7860, css=custom_css)