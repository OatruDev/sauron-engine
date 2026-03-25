import os
import pickle
import time
import json
import numpy as np
import faiss
import gradio as gr
from sentence_transformers import SentenceTransformer
from PIL import Image, ImageOps
import pandas as pd
from datetime import datetime

# --- CONFIGURACIÓN ---
DB_PKL = 'sauron_database.pkl'
CSV_LOG = 'bulk_fast_scan.csv'
RLHF_DIR = 'dataset_rlhf'
os.makedirs(RLHF_DIR, exist_ok=True)

print("Cargando modelo CLIP...")
model = SentenceTransformer('sentence-transformers/clip-ViT-B-32')

print(f"Cargando {DB_PKL}...")
with open(DB_PKL, 'rb') as f:
    cards_data = pickle.load(f)

print("Construyendo Índice FAISS...")
processed_embeddings = []
for c in cards_data:
    emb = c['embedding']
    if isinstance(emb, str):
        try: emb = json.loads(emb)
        except:
            emb = emb.strip("[]").split(",")
            emb = [float(x) for x in emb]
    processed_embeddings.append(emb)

embeddings_matrix = np.array(processed_embeddings).astype('float32')
faiss.normalize_L2(embeddings_matrix) 
index = faiss.IndexFlatIP(embeddings_matrix.shape[1])
index.add(embeddings_matrix)
print(f"Listo. {index.ntotal} cartas preparadas.")

# --- BLOQUEO GLOBAL (Evita el salto de frames) ---
ESTADO = {
    "pausado": False,
    "nombre": "",
    "set": "",
    "imagen_temp": None
}

def scan_realtime(imagen, invertir):
    global ESTADO
    
    # Si está pausado, ignoramos cualquier imagen nueva que llegue de la cámara
    if ESTADO["pausado"]:
        return gr.skip(), gr.skip()

    if imagen is None: 
        return "<div style='text-align:center; font-size:14px;'>Cámara inactiva</div>", "Iniciando..."

    if invertir:
        imagen = ImageOps.mirror(imagen)

    query_vector = model.encode(imagen).astype('float32').reshape(1, -1)
    faiss.normalize_L2(query_vector)
    D, I = index.search(query_vector, 1)
    
    # Umbral de detección
    if D[0][0] < 0.65:
        return "<div style='text-align:center; font-size:14px; color:#ff9800;'>Buscando...</div>", "🔴"

    # Se detectó una carta. Bloqueamos el sistema instantáneamente.
    ESTADO["pausado"] = True
    card = cards_data[I[0][0]]
    ESTADO["nombre"] = card['name']
    ESTADO["set"] = card['set_code']
    ESTADO["imagen_temp"] = imagen
    
    html_compacto = f"""
    <div style='text-align:center; background-color:#1e1e1e; color:white; padding:8px; border-radius:5px; border:1px solid #4caf50;'>
        <strong style='font-size:16px;'>{card['name']}</strong><br>
        <span style='font-size:12px; color:#aaa;'>Set: {card['set_code'].upper()} | {(D[0][0]*100):.1f}%</span>
    </div>
    """
    return html_compacto, "¡Pausado!"

def confirmar_carta():
    global ESTADO
    if not ESTADO["nombre"] or ESTADO["imagen_temp"] is None: 
        return gr.skip(), gr.skip()
    
    name = ESTADO["nombre"]
    set_code = ESTADO["set"]
    fecha_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    nombre_archivo = f"{set_code}_{name.replace(' ', '_').replace('/', '-')}_{fecha_str}.jpg"
    ruta_imagen = os.path.join(RLHF_DIR, nombre_archivo)
    ESTADO["imagen_temp"].save(ruta_imagen)
    
    nuevo_registro = pd.DataFrame([[datetime.now().strftime("%Y-%m-%d %H:%M:%S"), name, set_code, ruta_imagen]], columns=["Fecha", "Nombre", "Set", "Ruta_Imagen"])
    nuevo_registro.to_csv(CSV_LOG, mode='a', index=False, header=not os.path.exists(CSV_LOG))
    
    # Liberar el bloqueo para seguir escaneando
    ESTADO["pausado"] = False
    return "<div style='text-align:center; color:#4caf50; font-size:14px;'>✅ Guardado. Siguiente...</div>", "Buscando..."

def rechazar_carta():
    global ESTADO
    # Liberar el bloqueo sin guardar nada
    ESTADO["pausado"] = False
    return "<div style='text-align:center; color:#f44336; font-size:14px;'>❌ Descartado. Siguiente...</div>", "Buscando..."

# CSS compacto para evitar scroll innecesario en móvil
css = """
.gradio-container { padding: 5px !important; }
.gradio-container video { transform: none !important; max-height: 40vh; object-fit: cover; }
button { min-height: 40px !important; font-size: 14px !important; padding: 5px !important; }
"""

with gr.Blocks(title="SAURON UI", css=css) as demo:
    
    with gr.Row():
        input_img = gr.Image(sources=["webcam"], streaming=True, type="pil", show_label=False)
    
    chk_invertir = gr.Checkbox(label="Espejo", value=True, visible=False) # Se mantiene funcional pero oculto para ahorrar espacio
    
    with gr.Row():
        output_html = gr.HTML(value="<div style='text-align:center; font-size:14px;'>Enfoca una carta...</div>")
    
    with gr.Row():
        lbl_estado = gr.Label(value="Iniciando...", show_label=False)
        
    with gr.Row():
        btn_no = gr.Button("❌ NO ES", variant="secondary")
        btn_si = gr.Button("✅ SÍ ES", variant="success")

    # Bucle de procesamiento
    input_img.stream(
        fn=scan_realtime, 
        inputs=[input_img, chk_invertir], 
        outputs=[output_html, lbl_estado], 
        queue=False
    )
    
    # Controles
    btn_si.click(fn=confirmar_carta, outputs=[output_html, lbl_estado])
    btn_no.click(fn=rechazar_carta, outputs=[output_html, lbl_estado])

if __name__ == "__main__":
    demo.launch(share=True)