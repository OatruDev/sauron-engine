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

# Archivos Locales (Asegúrate de que estén en la misma carpeta en el móvil)
DB_PKL = 'sauron_database.pkl'
CSV_LOG = 'bulk_fast_scan.csv'

print("🧠 Cargando CLIP en el móvil...")
# Usamos el mismo modelo, el F4 GT tiene NPU para acelerar esto
model = SentenceTransformer('sentence-transformers/clip-ViT-B-32')

print(f"📦 Cargando Base de Datos ({DB_PKL})...")
with open(DB_PKL, 'rb') as f:
    cards_data = pickle.load(f)

print("⚡ Construyendo Índice FAISS...")
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
index = faiss.IndexFlatIP(embeddings_matrix.shape[1])
index.add(embeddings_matrix)
print(f"✅ ¡SAURON DESPIERTO EN ANDROID! {index.ntotal} cartas.")

ultima_carta_detectada = {"name": "", "set": ""}

def scan_realtime(imagen):
    global ultima_carta_detectada
    if imagen is None: return "Enfoca una carta...", ""

    t_start = time.time()
    query_vector = model.encode(imagen).astype('float32').reshape(1, -1)
    faiss.normalize_L2(query_vector)

    D, I = index.search(query_vector, 1)
    t_end = time.time()
    latencia_ms = (t_end - t_start) * 1000

    if D[0][0] < 0.75: # Subimos el umbral para evitar falsos positivos en móvil
        return "Buscando...", ""

    card = cards_data[I[0][0]]
    ultima_carta_detectada = {"name": card['name'], "set": card['set_code']}
    
    html_output = f"""
    <div style='text-align:center; background-color: #1a1a1a; color: white; padding: 10px; border-radius: 10px;'>
        <h3 style='margin:0;'>{card['name']}</h3>
        <p style='margin:0; color: #4caf50;'>Confianza: {D[0][0]*100:.1f}% | {latencia_ms:.0f}ms</p>
    </div>
    """
    return html_output, f"{card['name']} [{card['set_code'].upper()}]"

def confirmar_carta():
    global ultima_carta_detectada
    if not ultima_carta_detectada["name"]: return "❌ Vacío"
    
    nuevo_registro = pd.DataFrame([[datetime.now().strftime("%H:%M:%S"), ultima_carta_detectada["name"], ultima_carta_detectada["set"]]], columns=["Hora", "Nombre", "Set"])
    nuevo_registro.to_csv(CSV_LOG, mode='a', index=False, header=not os.path.exists(CSV_LOG))
    return f"✅ Guardado: {ultima_carta_detectada['name']}"

with gr.Blocks(title="SAURON Mobile") as demo:
    gr.Markdown("# 👁️ SAURON Local Mode")
    input_img = gr.Image(sources=["webcam"], streaming=True, type="pil")
    output_html = gr.HTML()
    lbl_confirmacion = gr.Label(value="Listo para escanear")
    btn_confirmar = gr.Button("💾 GUARDAR CARTA", variant="success")

    input_img.stream(fn=scan_realtime, inputs=[input_img], outputs=[output_html, lbl_confirmacion], queue=False)
    btn_confirmar.click(fn=confirmar_carta, inputs=[], outputs=[lbl_confirmacion])

if __name__ == "__main__":
    # Importante: Corremos en localhost (127.0.0.1)
    demo.launch(server_name="127.0.0.1", server_port=7860)