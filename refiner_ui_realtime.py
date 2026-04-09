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
from PIL import Image, ImageOps
import pandas as pd
from datetime import datetime
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# --- CONFIGURACIÓN DE ARCHIVOS ---
DB_PKL = 'sauron_database.pkl'
CSV_LOG = 'bulk_fast_scan.csv'
RLHF_DIR = 'dataset_rlhf'
os.makedirs(RLHF_DIR, exist_ok=True)

# --- PARÁMETROS DEL DETECTOR OPENCV ---
WORK_WIDTH   = 640
CANNY_LOW    = 30
CANNY_HIGH   = 120
DILATE_ITER  = 2
APPROX_COEFF = 0.02
MIN_AREA_RATIO = 0.05
ASPECT_MIN   = 1.25
ASPECT_MAX   = 1.55
OUT_W, OUT_H = 400, 560

# --- INICIALIZACIÓN DE MODELOS ---
print("🧠 Cargando Cerebro Visual (CLIP)...")
model = SentenceTransformer('sentence-transformers/clip-ViT-B-32')

print("📖 Cargando Lector de Texto (EasyOCR)...")
reader = easyocr.Reader(['en'], gpu=True, verbose=False)
ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 -,'"

print(f"📦 Cargando Índice FAISS ({DB_PKL})...")
with open(DB_PKL, 'rb') as f:
    cards_data = pickle.load(f)

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
print(f"✅ Listo. {index.ntotal} cartas en memoria.")

# --- ESTADO GLOBAL ---
ESTADO = {
    "pausado": False,
    "nombre": "",
    "set": "",
    "imagen_temp": None,
    "cooldown_hasta": 0.0,
    "blacklist": set()
}

# --- FUNCIONES DE VISIÓN (OpenCV) ---
def _order_points(pts):
    pts = pts.reshape(4, 2).astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered

def detect_and_rectify_card(frame):
    bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
    h_orig, w_orig = bgr.shape[:2]

    scale = WORK_WIDTH / w_orig
    w_work = WORK_WIDTH
    h_work = int(h_orig * scale)
    working = cv2.resize(bgr, (w_work, h_work), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=DILATE_ITER)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = MIN_AREA_RATIO * w_work * h_work
    best_cnt = None
    best_area = 0.0

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, APPROX_COEFF * peri, True)
        if len(approx) != 4: continue
        
        area = cv2.contourArea(approx)
        if area < min_area: continue

        rect = cv2.minAreaRect(approx)
        w, h = rect[1]
        if w == 0 or h == 0: continue
        
        ratio = max(w, h) / min(w, h)
        if not (ASPECT_MIN <= ratio <= ASPECT_MAX): continue

        if area > best_area:
            best_cnt = approx
            best_area = area

    if best_cnt is None:
        return None

    pts_work = best_cnt.reshape(4, 2).astype(np.float32)
    pts_orig = pts_work * (1.0 / scale)

    src = _order_points(pts_orig)
    dst = np.array([[0, 0], [OUT_W-1, 0], [OUT_W-1, OUT_H-1], [0, OUT_H-1]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(bgr, M, (OUT_W, OUT_H))

    return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))

def crop_title_roi_v2(card_img):
    img_cv = np.array(card_img)
    h, w = img_cv.shape[:2]
    roi_h = max(int(h * 0.15), 20)
    roi = img_cv[0:roi_h, 0:w]
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

# --- PIPELINE PRINCIPAL ---
def scan_realtime(imagen):
    global ESTADO
    
    if ESTADO["pausado"] or time.time() < ESTADO["cooldown_hasta"]:
        return gr.update()

    if imagen is None: return gr.update()

    try:
        t_start = time.time()
        
        # --- EL FIX CRÍTICO: Des-espejar la imagen cruda que manda Gradio ---
        imagen = ImageOps.mirror(imagen)

        # 1. DETECCIÓN Y RECTIFICACIÓN
        card_img = detect_and_rectify_card(imagen)
        if card_img is None:
            ESTADO["blacklist"].clear()
            return "<div class='mensaje-buscando' style='color:#03a9f4;'>Encuadra la carta completa...</div>"

        # 2. FASE VISUAL (FAISS)
        query_vector = model.encode(card_img).astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_vector)
        D, I = index.search(query_vector, 15)
        
        if D[0][0] < 0.65:
            return "<div class='mensaje-buscando'>🔴 Escaneando arte...</div>"

        candidatos = [cards_data[idx] for idx in I[0]]
        nombres_candidatos = [c['name'] for c in candidatos]

        # 3. FASE OCR
        roi_enhanced = crop_title_roi_v2(card_img)
        resultados_ocr = reader.readtext(roi_enhanced, detail=0, paragraph=True, allowlist=ALLOWLIST)
        texto_ocr = " ".join(resultados_ocr).strip()

        # 4. FASE VEREDICTO
        if texto_ocr:
            mejor_coincidencia = process.extractOne(query=texto_ocr, choices=nombres_candidatos, scorer=fuzz.WRatio)
            nombre_ganador, puntaje_fuzzy, indice_ganador = mejor_coincidencia
            carta_ganadora = candidatos[indice_ganador]
            metodo_log = f"OCR: '{texto_ocr}' | Score: {puntaje_fuzzy:.0f}"
        else:
            carta_ganadora = candidatos[0]
            metodo_log = "Visual Puro (OCR Falló)"
            
        if carta_ganadora['name'] in ESTADO["blacklist"]:
            return "<div class='mensaje-buscando' style='color:#f44336;'>Ignorando coincidencias previas...</div>"

        # 5. PAUSA Y RESULTADO
        ESTADO["pausado"] = True
        ESTADO["nombre"] = carta_ganadora['name']
        ESTADO["set"] = carta_ganadora['set_code']
        ESTADO["imagen_temp"] = card_img 
        
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
    if not ESTADO["nombre"] or ESTADO["imagen_temp"] is None: return gr.update()
    
    name, set_code = ESTADO["nombre"], ESTADO["set"]
    fecha_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    set_dir = os.path.join(RLHF_DIR, set_code)
    os.makedirs(set_dir, exist_ok=True)
    
    ruta_imagen = os.path.join(set_dir, f"{name.replace(' ', '_').replace('/', '-')}_{fecha_str}.jpg")
    ESTADO["imagen_temp"].save(ruta_imagen)
    
    nuevo_registro = pd.DataFrame([[datetime.now().strftime("%Y-%m-%d %H:%M:%S"), name, set_code, ruta_imagen]], columns=["Fecha", "Nombre", "Set", "Ruta_Imagen"])
    nuevo_registro.to_csv(CSV_LOG, mode='a', index=False, header=not os.path.exists(CSV_LOG))
    
    ESTADO["pausado"] = False
    ESTADO["cooldown_hasta"] = time.time() + 1.5
    ESTADO["blacklist"].clear() 
    return "<div class='mensaje-exito'>✅ Guardado. Retira la carta...</div>"

def rechazar_carta():
    global ESTADO
    if ESTADO["nombre"]: ESTADO["blacklist"].add(ESTADO["nombre"])
    ESTADO["pausado"] = False
    ESTADO["cooldown_hasta"] = time.time() + 0.5 
    return "<div class='mensaje-error'>❌ Descartado. Escaneando...</div>"

# --- INTERFAZ ---
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
    with gr.Row(): input_img = gr.Image(sources=["webcam"], streaming=True, type="pil", show_label=False)
    with gr.Row(): output_html = gr.HTML(value="<div class='mensaje-buscando'>Iniciando SAURON V2...</div>", elem_id="caja-resultado")
    with gr.Row():
        btn_no = gr.Button("❌ NO ES", variant="secondary")
        btn_si = gr.Button("✅ SÍ ES", variant="success")

    input_img.stream(fn=scan_realtime, inputs=[input_img], outputs=[output_html], queue=False)
    btn_si.click(fn=confirmar_carta, outputs=[output_html])
    btn_no.click(fn=rechazar_carta, outputs=[output_html])

if __name__ == "__main__":
    demo.launch(share=True)