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

print("🧠 Cargando Cerebro Visual (CLIP)...")
model = SentenceTransformer('sentence-transformers/clip-ViT-B-32')

print(f"📦 Cargando Base de Datos Local ({DB_PKL})...")
with open(DB_PKL, 'rb') as f:
    cards_data = pickle.load(f)

print("⚡ Construyendo Índice FAISS en RAM...")
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
print(f"✅ Listo. {index.ntotal} cartas.")

# --- EL MOTOR MANABOX (ESTADO GLOBAL) ---
estado = {
    "pausado": False, 
    "imagen": None, 
    "name": "", 
    "set": "",
    "html_actual": "<h3 style='text-align:center;'>Pon una carta frente a la cámara...</h3>"
}

def scan_realtime(imagen):
    global estado
    
    # 1. Si está pausado esperando tu respuesta, congelamos la pantalla
    if estado["pausado"]:
        return estado["html_actual"], "⚠️ Esperando tu decisión..."

    # 2. Si no hay imagen
    if imagen is None: 
        return "<h3 style='text-align:center;'>Encendiendo cámara...</h3>", "Cámara inactiva"

    # 3. Escaneo en vivo
    t_start = time.time()
    query_vector = model.encode(imagen).astype('float32').reshape(1, -1)
    faiss.normalize_L2(query_vector)
    D, I = index.search(query_vector, 1)
    
    # Si la confianza es muy baja, sigue buscando (sigue el stream)
    if D[0][0] < 0.65:
        return "<h3 style='text-align:center; color:orange;'>Buscando carta...</h3>", "Escaneando en vivo 🔴"

    # 4. ¡Encontró algo! Congelamos el estado
    card = cards_data[I[0][0]]
    estado["pausado"] = True
    estado["imagen"] = imagen
    estado["name"] = card['name']
    estado["set"] = card['set_code']
    
    latencia_ms = (time.time() - t_start) * 1000
    
    estado["html_actual"] = f"""
    <div style='text-align:center; background-color: #003300; color: white; padding: 15px; border-radius: 10px; border: 2px solid #4caf50;'>
        <h2 style='margin:0;'>¿Es {card['name']}?</h2>
        <p style='margin:0; color: #aaa;'>Set: {card['set_code'].upper()}</p>
        <p style='margin:0; color: #4caf50;'>Confianza: {D[0][0]*100:.1f}% | {latencia_ms:.0f}ms</p>
    </div>
    """
    return estado["html_actual"], f"¡Detectado! ¿Es esta?"

def confirmar_carta():
    global estado
    if not estado["name"] or estado["imagen"] is None: 
        return gr.update(), "No hay nada que guardar."
    
    # Guardar datos RLHF y CSV
    name = estado["name"]
    set_code = estado["set"]
    fecha_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    nombre_archivo = f"{set_code}_{name.replace(' ', '_').replace('/', '-')}_{fecha_str}.jpg"
    ruta_imagen = os.path.join(RLHF_DIR, nombre_archivo)
    estado["imagen"].save(ruta_imagen)
    
    nuevo_registro = pd.DataFrame([[datetime.now().strftime("%Y-%m-%d %H:%M:%S"), name, set_code, ruta_imagen]], columns=["Fecha", "Nombre", "Set", "Ruta_Imagen"])
    nuevo_registro.to_csv(CSV_LOG, mode='a', index=False, header=not os.path.exists(CSV_LOG))
    
    # REINICIAR Y SEGUIR BUSCANDO
    estado["pausado"] = False
    estado["html_actual"] = "<h3 style='text-align:center; color:green;'>✅ Guardado. Siguiente carta...</h3>"
    return estado["html_actual"], f"Guardado: {name}"

def rechazar_carta():
    global estado
    # REINICIAR Y SEGUIR BUSCANDO
    estado["pausado"] = False
    estado["html_actual"] = "<h3 style='text-align:center; color:red;'>❌ Descartado. Sigue escaneando...</h3>"
    return estado["html_actual"], "Buscando de nuevo..."

# --- INTERFAZ UI ---
with gr.Blocks(title="SAURON ManaBox Mode") as demo:
    gr.Markdown("# 👁️ SAURON - Escáner Continuo")
    
    with gr.Row():
        # Volvemos a activar streaming=True para lectura en vivo
        input_img = gr.Image(sources=["webcam"], streaming=True, type="pil", label="1. Toca el ícono de la cámara para usar la Trasera")
    
    with gr.Row():
        with gr.Column(scale=2):
            output_html = gr.HTML(value=estado["html_actual"], label="Resultado de la IA")
        with gr.Column(scale=1):
            lbl_estado = gr.Label(value="Iniciando...", label="Estado del Sistema")
            
            with gr.Row():
                btn_no = gr.Button("❌ NO ES", variant="secondary", size="lg")
                btn_si = gr.Button("✅ SÍ ES (Guardar)", variant="success", size="lg")

    # El stream evalúa constantemente. Si se pausa, se salta la evaluación pesada.
    input_img.stream(fn=scan_realtime, inputs=[input_img], outputs=[output_html, lbl_estado], queue=False)
    
    btn_si.click(fn=confirmar_carta, inputs=[], outputs=[output_html, lbl_estado])
    btn_no.click(fn=rechazar_carta, inputs=[], outputs=[output_html, lbl_estado])

if __name__ == "__main__":
    demo.launch(share=True)