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

def scan_realtime(imagen, pausado, invertir):
    # Si el sistema está pausado (esperando decisión del usuario), ignoramos los nuevos frames.
    # gr.skip() es crucial: evita que la interfaz se actualice y salte.
    if pausado:
        return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()

    if imagen is None: 
        return "<h3 style='text-align:center;'>Encendiendo cámara...</h3>", "Cámara inactiva", False, "", ""

    # Corregir el efecto espejo de la cámara de los navegadores móviles
    if invertir:
        imagen = ImageOps.mirror(imagen)

    t_start = time.time()
    query_vector = model.encode(imagen).astype('float32').reshape(1, -1)
    faiss.normalize_L2(query_vector)
    D, I = index.search(query_vector, 1)
    
    if D[0][0] < 0.65:
        return "<h3 style='text-align:center; color:orange;'>Buscando carta...</h3>", "Escaneando en vivo 🔴", False, "", ""

    # ¡Se detectó una carta!
    card = cards_data[I[0][0]]
    latencia_ms = (time.time() - t_start) * 1000
    
    html_actual = f"""
    <div style='text-align:center; background-color: #003300; color: white; padding: 15px; border-radius: 10px; border: 2px solid #4caf50;'>
        <h2 style='margin:0;'>¿Es {card['name']}?</h2>
        <p style='margin:0; color: #aaa;'>Set: {card['set_code'].upper()}</p>
        <p style='margin:0; color: #4caf50;'>Confianza: {D[0][0]*100:.1f}% | {latencia_ms:.0f}ms</p>
    </div>
    """
    # Retornamos pausado = True para detener el escaneo
    return html_actual, "¡Detectado! ¿Es esta?", True, card['name'], card['set_code']

def confirmar_carta(imagen, name, set_code, invertir):
    if not name or imagen is None: 
        return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()
    
    # Invertir la imagen antes de guardarla para que el dataset RLHF sea correcto
    if invertir:
        imagen = ImageOps.mirror(imagen)
        
    fecha_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_archivo = f"{set_code}_{name.replace(' ', '_').replace('/', '-')}_{fecha_str}.jpg"
    ruta_imagen = os.path.join(RLHF_DIR, nombre_archivo)
    imagen.save(ruta_imagen)
    
    nuevo_registro = pd.DataFrame([[datetime.now().strftime("%Y-%m-%d %H:%M:%S"), name, set_code, ruta_imagen]], columns=["Fecha", "Nombre", "Set", "Ruta_Imagen"])
    nuevo_registro.to_csv(CSV_LOG, mode='a', index=False, header=not os.path.exists(CSV_LOG))
    
    # Despausar y limpiar estados
    html_actual = "<h3 style='text-align:center; color:green;'>✅ Guardado. Siguiente carta...</h3>"
    return html_actual, f"Guardado: {name}", False, "", ""

def rechazar_carta():
    # Despausar y limpiar estados
    html_actual = "<h3 style='text-align:center; color:red;'>❌ Descartado. Sigue escaneando...</h3>"
    return html_actual, "Buscando de nuevo...", False, "", ""

# CSS para forzar visualmente que el video no se vea invertido en la pantalla del celular
css = """
.gradio-container video { transform: none !important; }
"""

with gr.Blocks(title="SAURON ManaBox Mode", css=css) as demo:
    gr.Markdown("# 👁️ SAURON - Escáner Continuo")
    
    # Estados ocultos para evitar problemas de concurrencia
    state_pausado = gr.State(False)
    state_name = gr.State("")
    state_set = gr.State("")
    
    with gr.Row():
        input_img = gr.Image(sources=["webcam"], streaming=True, type="pil", label="Cámara en vivo")
    
    with gr.Row():
        chk_invertir = gr.Checkbox(label="Corregir efecto espejo (Mantenlo activado)", value=True)
        
    with gr.Row():
        with gr.Column(scale=2):
            output_html = gr.HTML(value="<h3 style='text-align:center;'>Pon una carta frente a la cámara...</h3>", label="Resultado de la IA")
        with gr.Column(scale=1):
            lbl_estado = gr.Label(value="Iniciando...", label="Estado del Sistema")
            
            with gr.Row():
                btn_no = gr.Button("❌ NO ES", variant="secondary", size="lg")
                btn_si = gr.Button("✅ SÍ ES", variant="success", size="lg")

    # Bucle de procesamiento de video
    input_img.stream(
        fn=scan_realtime, 
        inputs=[input_img, state_pausado, chk_invertir], 
        outputs=[output_html, lbl_estado, state_pausado, state_name, state_set], 
        queue=False
    )
    
    # Controles de decisión
    btn_si.click(
        fn=confirmar_carta, 
        inputs=[input_img, state_name, state_set, chk_invertir], 
        outputs=[output_html, lbl_estado, state_pausado, state_name, state_set]
    )
    
    btn_no.click(
        fn=rechazar_carta, 
        inputs=[], 
        outputs=[output_html, lbl_estado, state_pausado, state_name, state_set]
    )

if __name__ == "__main__":
    demo.launch(share=True)