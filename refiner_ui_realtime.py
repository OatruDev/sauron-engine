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

# --- ESTADO Y BLOQUEO GLOBAL ---
ESTADO = {
    "pausado": False,
    "nombre": "",
    "set": "",
    "imagen_temp": None,
    "cooldown_hasta": 0.0  # Temporizador para evitar saltos
}

def scan_realtime(imagen):
    global ESTADO
    
    # 1. Si está pausado o en tiempo de gracia (cooldown), ignoramos la cámara
    if ESTADO["pausado"] or time.time() < ESTADO["cooldown_hasta"]:
        return gr.update()

    if imagen is None: 
        return gr.update()

    try:
        # 2. Inferencia
        query_vector = model.encode(imagen).astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_vector)
        D, I = index.search(query_vector, 1)
        
        # 3. Umbral de detección
        if D[0][0] < 0.65:
            return "<div class='mensaje-buscando'>🔴 Escaneando...</div>"

        # 4. Carta detectada: Bloqueamos el sistema
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
        # Esto evita el SyntaxError en la interfaz si hay un frame corrupto
        print(f"Frame ignorado por error: {e}")
        return gr.update()

def confirmar_carta():
    global ESTADO
    if not ESTADO["nombre"] or ESTADO["imagen_temp"] is None: 
        return gr.update()
    
    name = ESTADO["nombre"]
    set_code = ESTADO["set"]
    fecha_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    nombre_archivo = f"{set_code}_{name.replace(' ', '_').replace('/', '-')}_{fecha_str}.jpg"
    ruta_imagen = os.path.join(RLHF_DIR, nombre_archivo)
    ESTADO["imagen_temp"].save(ruta_imagen)
    
    nuevo_registro = pd.DataFrame([[datetime.now().strftime("%Y-%m-%d %H:%M:%S"), name, set_code, ruta_imagen]], columns=["Fecha", "Nombre", "Set", "Ruta_Imagen"])
    nuevo_registro.to_csv(CSV_LOG, mode='a', index=False, header=not os.path.exists(CSV_LOG))
    
    # Liberar el bloqueo y aplicar 2 segundos de gracia para cambiar la carta
    ESTADO["pausado"] = False
    ESTADO["cooldown_hasta"] = time.time() + 2.0
    return "<div class='mensaje-exito'>✅ Guardado. Retira la carta...</div>"

def rechazar_carta():
    global ESTADO
    # Liberar el bloqueo y aplicar 2 segundos de gracia para retirar la carta
    ESTADO["pausado"] = False
    ESTADO["cooldown_hasta"] = time.time() + 2.0
    return "<div class='mensaje-error'>❌ Descartado. Retira la carta...</div>"

# --- DISEÑO FIJO ---
css = """
.gradio-container { padding: 5px !important; }
.gradio-container video { transform: none !important; max-height: 40vh; object-fit: contain; }

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
            elem_id="caja-resultado"
        )
        
    with gr.Row():
        btn_no = gr.Button("❌ NO ES", variant="secondary")
        btn_si = gr.Button("✅ SÍ ES", variant="success")

    input_img.stream(
        fn=scan_realtime, 
        inputs=[input_img], 
        outputs=[output_html], 
        queue=False
    )
    
    btn_si.click(fn=confirmar_carta, outputs=[output_html])
    btn_no.click(fn=rechazar_carta, outputs=[output_html])

if __name__ == "__main__":
    demo.launch(share=True)