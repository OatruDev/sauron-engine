import os
import pickle
import time
import json
import numpy as np
import faiss
import cv2
import gradio as gr
from sentence_transformers import SentenceTransformer
import easyocr
from rapidfuzz import process, fuzz
from PIL import Image
import pandas as pd
from datetime import datetime
import warnings

# Suprimir warnings molestos de EasyOCR en la consola
warnings.filterwarnings("ignore", category=FutureWarning)

# --- CONFIGURACIÓN DE ARCHIVOS ---
DB_PKL = 'sauron_database.pkl'
CSV_LOG = 'bulk_fast_scan.csv'
RLHF_DIR = 'dataset_rlhf'
os.makedirs(RLHF_DIR, exist_ok=True)

print("🧠 Cargando Cerebro Visual (CLIP)...")
model = SentenceTransformer('sentence-transformers/clip-ViT-B-32')

print("📖 Cargando Lector de Texto (EasyOCR)...")
# gpu=True acelerará el proceso si tu PC tiene NVIDIA. Si no, usa CPU automáticamente.
reader = easyocr.Reader(['en'], gpu=True, verbose=False)
ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 -,'"

print(f"📦 Cargando Base de Datos Local ({DB_PKL})...")
if not os.path.exists(DB_PKL):
    raise FileNotFoundError(f"❌ No encuentro {DB_PKL}. Ejecuta rescue_data.py si lo perdiste.")

with open(DB_PKL, 'rb') as f:
    cards_data = pickle.load(f)

print("⚡ Construyendo Índice FAISS en RAM...")
processed_embeddings = []
for c in cards_data:
    emb = c['embedding']
    if isinstance(emb, str):
        try: emb = json.loads(emb)
        except: emb = [float(x) for x in emb.strip("[]").split(",")]
    processed_embeddings.append(emb)

embeddings_matrix = np.array(processed_embeddings).astype('float32')
faiss.normalize_L2(embeddings_matrix) 
index = faiss.IndexFlatIP(embeddings_matrix.shape[1])
index.add(embeddings_matrix)
print(f"✅ Listo. {index.ntotal} cartas preparadas en el índice.")

# --- ESTADO GLOBAL (MANABOX MODE & RLHF) ---
ESTADO = {
    "pausado": False,
    "nombre": "",
    "set": "",
    "imagen_temp": None,
    "cooldown_hasta": 0.0,
    "blacklist": set()
}

def scan_realtime(imagen):
    global ESTADO
    
    if ESTADO["pausado"] or time.time() < ESTADO["cooldown_hasta"]:
        return gr.update()

    if imagen is None: 
        return gr.update()

    try:
        t_start = time.time()
        
        # --- 1. PREPARACIÓN VISUAL (CLIP) ---
        img_clip = imagen.copy()
        img_clip.thumbnail((512, 512), Image.Resampling.LANCZOS)

        # --- 2. FASE 1: FILTRO GRUESO (FAISS) ---
        query_vector = model.encode(img_clip).astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_vector)
        # CRÍTICO: Obtenemos el TOP 15 candidatos visuales
        D, I = index.search(query_vector, 15)
        
        # Umbral: Si el candidato #1 es muy malo, no hay carta en pantalla.
        if D[0][0] < 0.65:
            ESTADO["blacklist"].clear() # Limpiamos memoria si retiras la carta
            return "<div class='mensaje-buscando'>🔴 Escaneando...</div>"

        # Extraemos los datos de las 15 cartas
        candidatos = [cards_data[idx] for idx in I[0]]
        nombres_candidatos = [c['name'] for c in candidatos]

        # --- 3. FASE 2: EL ANCLA DE TEXTO (EasyOCR + CLAHE) ---
        img_cv = np.array(imagen) # Convertir PIL a OpenCV (RGB)
        alto, ancho = img_cv.shape[:2]
        
        # Recorte (ROI): Solo el 15% superior de la imagen (donde está el título)
        roi = img_cv[0:max(int(alto * 0.15), 20), 0:ancho]
        
        # Mejora de contraste para textos brillantes/oscuros
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        roi_enhanced = clahe.apply(gray)
        
        # Lectura OCR
        resultados_ocr = reader.readtext(roi_enhanced, detail=0, paragraph=True, allowlist=ALLOWLIST)
        texto_ocr = " ".join(resultados_ocr).strip()

        # --- 4. FASE 3: EL VEREDICTO (Fuzzy Matching) ---
        if texto_ocr:
            # Cruzamos lo que leyó el OCR contra el Top 15 visual
            mejor_coincidencia = process.extractOne(
                query=texto_ocr, 
                choices=nombres_candidatos, 
                scorer=fuzz.WRatio
            )
            nombre_ganador, puntaje_fuzzy, indice_ganador = mejor_coincidencia
            carta_ganadora = candidatos[indice_ganador]
            metodo_log = f"Fusión (OCR: '{texto_ocr}' | Score: {puntaje_fuzzy:.0f})"
        else:
            # Si el OCR queda ciego por reflejos, confiamos en el #1 visual
            carta_ganadora = candidatos[0]
            metodo_log = "Visual Puro (OCR Falló/Sin Texto)"
            
        # --- 5. CONTROL DE BUCLE (Lista Negra) ---
        if carta_ganadora['name'] in ESTADO["blacklist"]:
            return "<div class='mensaje-buscando' style='color:#f44336;'>Buscando alternativas...</div>"

        # --- 6. PAUSA Y RESULTADO (UI) ---
        ESTADO["pausado"] = True
        ESTADO["nombre"] = carta_ganadora['name']
        ESTADO["set"] = carta_ganadora['set_code']
        ESTADO["imagen_temp"] = img_clip # Guardamos versión 512px para RLHF
        
        latencia_ms = (time.time() - t_start) * 1000
        
        html_resultado = f"""
        <div class='mensaje-detectado'>
            <strong>{carta_ganadora['name']}</strong><br>
            <span style='font-size:14px; color:#aaa;'>Set: {carta_ganadora['set_code'].upper()} | Visual: {(D[0][0]*100):.1f}%</span><br>
            <span style='font-size:12px; color:#4caf50;'>{metodo_log} | {latencia_ms:.0f}ms</span>
        </div>
        """
        return html_resultado
    
    except Exception as e:
        print(f"Error procesando frame: {e}")
        return gr.update()

def confirmar_carta():
    global ESTADO
    if not ESTADO["nombre"] or ESTADO["imagen_temp"] is None: 
        return gr.update()
    
    name = ESTADO["nombre"]
    set_code = ESTADO["set"]
    fecha_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Crear carpeta del Set si no existe
    set_dir = os.path.join(RLHF_DIR, set_code)
    os.makedirs(set_dir, exist_ok=True)
    
    # Guardar foto real
    nombre_archivo = f"{name.replace(' ', '_').replace('/', '-')}_{fecha_str}.jpg"
    ruta_imagen = os.path.join(set_dir, nombre_archivo)
    ESTADO["imagen_temp"].save(ruta_imagen)
    
    # Guardar en log CSV
    nuevo_registro = pd.DataFrame([[datetime.now().strftime("%Y-%m-%d %H:%M:%S"), name, set_code, ruta_imagen]], columns=["Fecha", "Nombre", "Set", "Ruta_Imagen"])
    nuevo_registro.to_csv(CSV_LOG, mode='a', index=False, header=not os.path.exists(CSV_LOG))
    
    # Desbloquear y limpiar para la siguiente
    ESTADO["pausado"] = False
    ESTADO["cooldown_hasta"] = time.time() + 1.5
    ESTADO["blacklist"].clear() 
    return "<div class='mensaje-exito'>✅ ¡Guardado! Retira la carta...</div>"

def rechazar_carta():
    global ESTADO
    # Meter carta equivocada a la lista negra
    if ESTADO["nombre"]:
        ESTADO["blacklist"].add(ESTADO["nombre"])
        
    ESTADO["pausado"] = False
    ESTADO["cooldown_hasta"] = time.time() + 0.5 
    return "<div class='mensaje-error'>❌ Descartado. Escaneando la siguiente opción...</div>"

# --- INTERFAZ DE USUARIO COMPACTA ---
css = """
.gradio-container { padding: 5px !important; }
.gradio-container video { transform: none !important; max-height: 40vh; object-fit: contain; }
#caja-resultado { height: 110px !important; display: flex; align-items: center; justify-content: center; margin-bottom: 5px; width: 100%; }
.mensaje-buscando { color: #ff9800; font-size: 16px; font-weight: bold; text-align: center; width: 100%; }
.mensaje-exito { color: #4caf50; font-size: 16px; font-weight: bold; text-align: center; width: 100%; }
.mensaje-error { color: #f44336; font-size: 16px; font-weight: bold; text-align: center; width: 100%; }
.mensaje-detectado { background-color: #1e1e1e; color: white; padding: 10px; border-radius: 8px; border: 2px solid #4caf50; width: 100%; text-align: center; font-size: 16px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
button { min-height: 50px !important; font-size: 16px !important; font-weight: bold !important; border-radius: 8px !important; }
"""

with gr.Blocks(title="SAURON V2", css=css) as demo:
    with gr.Row():
        input_img = gr.Image(sources=["webcam"], streaming=True, type="pil", show_label=False)
    
    with gr.Row():
        output_html = gr.HTML(
            value="<div class='mensaje-buscando'>Iniciando sistema V2...</div>",
            elem_id="caja-resultado"
        )
        
    with gr.Row():
        btn_no = gr.Button("❌ NO ES", variant="secondary")
        btn_si = gr.Button("✅ SÍ ES", variant="success")

    input_img.stream(fn=scan_realtime, inputs=[input_img], outputs=[output_html], queue=False)
    btn_si.click(fn=confirmar_carta, outputs=[output_html])
    btn_no.click(fn=rechazar_carta, outputs=[output_html])

if __name__ == "__main__":
    # share=True asegura el túnel HTTPS para que la cámara del POCO F4 GT funcione
    demo.launch(share=True)