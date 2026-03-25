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

# --- BLOQUEO GLOBAL ---
ESTADO = {
    "pausado": False,
    "nombre": "",
    "set": "",
    "imagen_temp": None
}

def scan_realtime(imagen):
    global ESTADO
    
    # Si el sistema está esperando tu decisión, ignoramos las imágenes nuevas
    if ESTADO["pausado"]:
        return gr.skip()

    if imagen is None: 
        return gr.skip()

    try:
        query_vector = model.encode(imagen).astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_vector)
        D, I = index.search(query_vector, 1)
        
        # Umbral de detección
        if D[0][0] < 0.65:
            return "<div class='mensaje-buscando'>🔴 Buscando carta...</div>"

        # Se detectó una carta. Bloqueamos el sistema.
        ESTADO["pausado"] = True
        card = cards_data[I[0][0]]
        ESTADO["nombre"] = card['name']
        ESTADO["set"] = card['set_code']
        ESTADO["imagen_temp"] = imagen
        
        html_resultado = f"""
        <div class='mensaje-detectado'>
            <strong>{card['name']}</strong><br>
            <span style='font-size:14px; color:#aaa;'>Set: {card['set_code'].upper()} | Confianza: {(D[0][0]*100):.1f}%</span>
        </div>
        """
        return html_resultado
    
    except Exception as e:
        print(f"Error en procesamiento: {e}")
        return gr.skip()

def confirmar_carta():
    global ESTADO
    if not ESTADO["nombre"] or ESTADO["imagen_temp"] is None: 
        return gr.skip()
    
    name = ESTADO["nombre"]
    set_code = ESTADO["set"]
    fecha_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    nombre_archivo = f"{set_code}_{name.replace(' ', '_').replace('/', '-')}_{fecha_str}.jpg"
    ruta_imagen = os.path.join(RLHF_DIR, nombre_archivo)
    ESTADO["imagen_temp"].save(ruta_imagen)
    
    nuevo_registro = pd.DataFrame([[datetime.now().strftime("%Y-%m-%d %H:%M:%S"), name, set_code, ruta_imagen]], columns=["Fecha", "Nombre", "Set", "Ruta_Imagen"])
    nuevo_registro.to_csv(CSV_LOG, mode='a', index=False, header=not os.path.exists(CSV_LOG))
    
    # Liberar el bloqueo
    ESTADO["pausado"] = False
    return "<div class='mensaje-exito'>✅ Guardado. Siguiente carta...</div>"

def rechazar_carta():
    global ESTADO
    # Liberar el bloqueo sin guardar nada
    ESTADO["pausado"] = False
    return "<div class='mensaje-error'>❌ Descartado. Siguiente carta...</div>"

# --- DISEÑO FIJO Y COMPACTO ---
css = """
.gradio-container { padding: 5px !important; }
.gradio-container video { transform: none !important; max-height: 40vh; object-fit: contain; }

/* Contenedor de altura fija para evitar los saltos de interfaz */
#caja-resultado {
    height: 90px !important;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 5px;
    width: 100%;
}

.mensaje-buscando { color: #ff9800; font-size: 16px; font-weight: bold; }
.mensaje-exito { color: #4caf50; font-size: 16px; font-weight: bold; }
.mensaje-error { color: #f44336; font-size: 16px; font-weight: bold; }
.mensaje-detectado {
    background-color: #1e1e1e;
    color: white;
    padding: 10px;
    border-radius: 8px;
    border: 2px solid #4caf50;
    width: 100%;
    text-align: center;
    font-size: 18px;
}

button { min-height: 50px !important; font-size: 16px !important; font-weight: bold !important; }
"""

with gr.Blocks(title="SAURON UI", css=css) as demo:
    with gr.Row():
        input_img = gr.Image(sources=["webcam"], streaming=True, type="pil", show_label=False)
    
    with gr.Row():
        output_html = gr.HTML(
            value="<div class='mensaje-buscando'>Iniciando sistema...</div>",
            elem_id="caja-resultado" # Aplica la altura fija del CSS
        )
        
    with gr.Row():
        btn_no = gr.Button("❌ NO ES", variant="secondary")
        btn_si = gr.Button("✅ SÍ ES", variant="success")

    # Bucle de procesamiento de video
    input_img.stream(
        fn=scan_realtime, 
        inputs=[input_img], 
        outputs=[output_html], 
        queue=False
    )
    
    # Controles de decisión
    btn_si.click(fn=confirmar_carta, outputs=[output_html])
    btn_no.click(fn=rechazar_carta, outputs=[output_html])

if __name__ == "__main__":
    demo.launch(share=True)