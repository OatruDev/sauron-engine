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
import concurrent.futures

warnings.filterwarnings("ignore", category=FutureWarning)

# --- CONFIGURACIÓN DE ARCHIVOS ---
DB_PKL = 'sauron_database.pkl'
CSV_LOG = 'bulk_fast_scan.csv'
RLHF_DIR = 'dataset_rlhf'
TRANSLATOR_DB = 'sauron_translator.json'
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

print("🌐 Preparando Diccionario Global Multilingüe...")
diccionario_global = {} 
if os.path.exists(TRANSLATOR_DB):
    with open(TRANSLATOR_DB, 'r', encoding='utf-8') as f:
        rosetta_stone = json.load(f)
        for eng_name, foreign_list in rosetta_stone.items():
            diccionario_global[eng_name] = eng_name
            for f_name in foreign_list:
                diccionario_global[f_name] = eng_name
else:
    print("⚠️ No se encontró la Piedra Rosetta.")
    rosetta_stone = {}

OPCIONES_GLOBALES = list(diccionario_global.keys())

print("⚡ Procesando Vectores y Agrupando Variantes...")
processed_embeddings = []
cartas_por_nombre = {}

for c in cards_data:
    emb = c['embedding']
    if isinstance(emb, str):
        try: emb = json.loads(emb)
        except: emb = [float(x) for x in emb.strip("[]").split(",")]
    
    c['parsed_emb'] = np.array(emb, dtype='float32') 
    processed_embeddings.append(c['parsed_emb'])

    if c['name'] not in cartas_por_nombre:
        cartas_por_nombre[c['name']] = []
    cartas_por_nombre[c['name']].append(c)

embeddings_matrix = np.array(processed_embeddings).astype('float32')
faiss.normalize_L2(embeddings_matrix) 
index = faiss.IndexFlatIP(embeddings_matrix.shape[1])
index.add(embeddings_matrix)
print(f"✅ Listo. {index.ntotal} cartas en memoria.")

# --- INICIALIZAR CONTADOR PERSISTENTE ---
if os.path.exists(CSV_LOG):
    try:
        df_log = pd.read_csv(CSV_LOG)
        conteo_inicial = len(df_log)
    except:
        conteo_inicial = 0
else:
    conteo_inicial = 0

# --- ESTADO GLOBAL ---
ESTADO = {
    "pausado": False,
    "id": "",            
    "nombre": "",
    "set": "",
    "nc": "",            
    "imagen_temp": None,
    "cooldown_hasta": 0.0,
    "blacklist_nombres": set(),   
    "blacklist_variantes": set(),
    "total_guardadas": conteo_inicial  # <-- CONTADOR ACTIVO
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
    
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 80: 
        return None 

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

    if best_cnt is None: return None

    pts_work = best_cnt.reshape(4, 2).astype(np.float32)
    pts_orig = pts_work * (1.0 / scale)

    src = _order_points(pts_orig)
    dst = np.array([[0, 0], [OUT_W-1, 0], [OUT_W-1, OUT_H-1], [0, OUT_H-1]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(bgr, M, (OUT_W, OUT_H))

    return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))

# FUNCIONES PARA MULTITHREADING
def tarea_clip(card_img):
    query_vector = model.encode(card_img).astype('float32').reshape(1, -1)
    faiss.normalize_L2(query_vector) 
    D, I = index.search(query_vector, 30)
    return query_vector, D, I

def tarea_ocr(card_img):
    img_cv = np.array(card_img)
    h, w = img_cv.shape[:2]
    roi_h = max(int(h * 0.15), 20)
    roi = img_cv[0:roi_h, 0:w]
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    roi_enhanced = clahe.apply(gray)
    resultados_ocr = reader.readtext(roi_enhanced, detail=0, paragraph=True, allowlist=ALLOWLIST)
    return " ".join(resultados_ocr).strip()

# --- PIPELINE PRINCIPAL ---
def scan_realtime(imagen):
    global ESTADO
    
    if ESTADO["pausado"] or time.time() < ESTADO["cooldown_hasta"]:
        return gr.update()

    if imagen is None: return gr.update()

    try:
        t_start = time.time()
        imagen = ImageOps.mirror(imagen)

        card_img = detect_and_rectify_card(imagen)
        if card_img is None:
            ESTADO["blacklist_nombres"].clear()
            ESTADO["blacklist_variantes"].clear()
            return "<div class='mensaje-buscando' style='color:#03a9f4;'>Encuadra la carta (Evita moverla)...</div>"

        # NUEVO: EJECUCIÓN EN PARALELO (CLIP y OCR al mismo tiempo)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futuro_clip = executor.submit(tarea_clip, card_img)
            futuro_ocr = executor.submit(tarea_ocr, card_img)
            
            query_vector, D, I = futuro_clip.result()
            texto_ocr = futuro_ocr.result()
        
        if D[0][0] < 0.65:
            return "<div class='mensaje-buscando'>🔴 Escaneando arte...</div>"

        candidatos = [cards_data[idx] for idx in I[0]]
        candidatos_filtrados = []
        for c in candidatos:
            if c['name'] in ESTADO["blacklist_nombres"]: continue
            if c['id'] in ESTADO["blacklist_variantes"]: continue 
            candidatos_filtrados.append(c)

        if not candidatos_filtrados:
             return "<div class='mensaje-buscando' style='color:#f44336;'>Agotadas las versiones...</div>"

        diccionario_busqueda = {}
        for c in candidatos_filtrados:
            eng_name = c['name']
            diccionario_busqueda[eng_name] = eng_name 
            for traduccion in rosetta_stone.get(eng_name, []):
                diccionario_busqueda[traduccion] = eng_name

        carta_ganadora = None
        metodo_log = "Visual Puro (OCR Falló)"
        puntaje_fuzzy = 0

        if texto_ocr:
            mejor_coincidencia = process.extractOne(
                query=texto_ocr, 
                choices=list(diccionario_busqueda.keys()), 
                scorer=fuzz.WRatio
            )
            
            if mejor_coincidencia:
                nombre_identificado, puntaje_fuzzy, _ = mejor_coincidencia
            
            if puntaje_fuzzy >= 65:
                nombre_ingles_real = diccionario_busqueda[nombre_identificado]
                carta_ganadora = next(c for c in candidatos_filtrados if c['name'] == nombre_ingles_real)
                metodo_log = f"Local OCR: '{texto_ocr}'"
            
            else:
                mejor_global = process.extractOne(
                    query=texto_ocr, 
                    choices=OPCIONES_GLOBALES, 
                    scorer=fuzz.WRatio,
                    score_cutoff=75
                )
                
                if mejor_global:
                    nombre_global_id, puntaje_global, _ = mejor_global
                    nombre_ingles_real = diccionario_global[nombre_global_id]
                    
                    if nombre_ingles_real not in ESTADO["blacklist_nombres"]:
                        variantes = cartas_por_nombre.get(nombre_ingles_real, [])
                        variantes_validas = [v for v in variantes if v['id'] not in ESTADO["blacklist_variantes"]]
                        
                        if variantes_validas:
                            vectores_variantes = [v['parsed_emb'] for v in variantes_validas]
                            matriz_var = np.array(vectores_variantes).astype('float32')
                            similitudes = np.dot(matriz_var, query_vector.T).flatten()
                            mejor_var_idx = np.argmax(similitudes)
                            
                            carta_ganadora = variantes_validas[mejor_var_idx]
                            puntaje_fuzzy = puntaje_global
                            metodo_log = f"🚀 OVERRIDE: '{texto_ocr}'"
        
        if carta_ganadora is None:
            carta_ganadora = candidatos_filtrados[0]

        latencia_ms = (time.time() - t_start) * 1000
        nc = carta_ganadora.get('collector_number', '???')
        
        # --- NUEVA LÓGICA DE TRADUCCIÓN Y PREVIEW (HTML) ---
        nombre_ingles = carta_ganadora['name']
        
        # Buscar la traducción al español en la Piedra Rosetta
        traducciones = rosetta_stone.get(nombre_ingles, [])
        nombre_castellano = " (Sin traducción local)"
        for t in traducciones:
            if t != nombre_ingles: # Asumimos que si es distinta, es el idioma local
                nombre_castellano = f" / {t}"
                break
        
        image_uri = carta_ganadora.get('image_uri', '')

        # UI Rediseñada para el Bulk Escaneo masivo
        html_resultado = f"""
        <div class='mensaje-detectado'>
            <div style='display: flex; align-items: center; gap: 15px;'>
                <img src='{image_uri}' style='width: 45px; height: 63px; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.5);' alt='arte'>
                
                <div style='flex: 1;'>
                    <div style='font-size: 16px;'>
                        <strong style='color: #fff;'>{nombre_ingles}</strong>
                        <span style='color: #4caf50; font-weight: bold;'>{nombre_castellano}</span>
                    </div>
                    
                    <div style='font-size: 13px; color: #aaa; margin-top: 2px;'>
                        Set: {carta_ganadora['set_code'].upper()} | NC: #{nc} | Visual: {(D[0][0]*100):.1f}%
                    </div>
                    
                    <div style='font-size: 11px; color: #666; font-family: monospace;'>
                        {metodo_log} (Score: {puntaje_fuzzy:.0f}) ⚡ {latencia_ms:.0f}ms
                    </div>
                </div>
            </div>
        </div>
        """
        return html_resultado
    
    except Exception as e:
        print(f"Error procesando frame: {e}")
        return gr.update()

def confirmar_carta():
    global ESTADO
    if not ESTADO["nombre"] or ESTADO["imagen_temp"] is None: return gr.update()
    
    name, set_code, nc = ESTADO["nombre"], ESTADO["set"], ESTADO["nc"]
    fecha_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    set_dir = os.path.join(RLHF_DIR, set_code)
    os.makedirs(set_dir, exist_ok=True)
    
    ruta_imagen = os.path.join(set_dir, f"{nc}_{name.replace(' ', '_').replace('/', '-')}_{fecha_str}.jpg")
    ESTADO["imagen_temp"].save(ruta_imagen)
    
    nuevo_registro = pd.DataFrame([[datetime.now().strftime("%Y-%m-%d %H:%M:%S"), name, set_code, nc, ruta_imagen]], columns=["Fecha", "Nombre", "Set", "NC", "Ruta_Imagen"])
    nuevo_registro.to_csv(CSV_LOG, mode='a', index=False, header=not os.path.exists(CSV_LOG))
    
    # --- ACTUALIZACIÓN DEL CONTADOR PERSISTENTE ---
    ESTADO["total_guardadas"] += 1
    total = ESTADO["total_guardadas"]
    
    ESTADO["pausado"] = False
    ESTADO["cooldown_hasta"] = time.time() + 1.5
    ESTADO["blacklist_nombres"].clear()
    ESTADO["blacklist_variantes"].clear()
    return f"<div class='mensaje-exito'>✅ Guardado (Total almacenado: {total} cartas). Retira la carta...</div>"

def rechazar_carta():
    global ESTADO
    if ESTADO["nombre"]: ESTADO["blacklist_nombres"].add(ESTADO["nombre"])
    ESTADO["pausado"] = False
    ESTADO["cooldown_hasta"] = time.time() + 0.5 
    return "<div class='mensaje-error'>❌ Carta equivocada. Escaneando...</div>"

def rechazar_version():
    global ESTADO
    if ESTADO["id"]: ESTADO["blacklist_variantes"].add(ESTADO["id"]) 
    ESTADO["pausado"] = False
    ESTADO["cooldown_hasta"] = time.time() + 0.2
    return "<div class='mensaje-buscando' style='color:#2196F3;'>🔄 Buscando otra versión...</div>"

# --- INTERFAZ ---
css = """
.gradio-container { padding: 5px !important; }
.gradio-container video { transform: none !important; max-height: 40vh; object-fit: contain; }
#caja-resultado { height: 110px !important; display: flex; align-items: center; justify-content: center; margin-bottom: 5px; width: 100%; }
.mensaje-buscando { color: #ff9800; font-size: 16px; font-weight: bold; text-align: center; width: 100%; }
.mensaje-exito { color: #4caf50; font-size: 16px; font-weight: bold; text-align: center; width: 100%; }
.mensaje-error { color: #f44336; font-size: 16px; font-weight: bold; text-align: center; width: 100%; }
.mensaje-detectado { background-color: #1e1e1e; color: white; padding: 10px; border-radius: 8px; border: 2px solid #4caf50; width: 100%; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
button { min-height: 50px !important; font-size: 13px !important; font-weight: bold !important; border-radius: 8px !important; }
"""

with gr.Blocks(title="SAURON V2", css=css) as demo:
    with gr.Row(): input_img = gr.Image(sources=["webcam"], streaming=True, type="pil", show_label=False)
    # Mostramos el contador histórico en el HTML inicial
    output_html = gr.HTML(value=f"<div class='mensaje-buscando'>Iniciando... (Almacenado hasta ahora: {ESTADO['total_guardadas']} cartas)</div>", elem_id="caja-resultado")
    
    with gr.Row():
        btn_no = gr.Button("❌ NO ES", variant="secondary")
        btn_otra_version = gr.Button("🔄 OTRA VERSIÓN", variant="primary")
        btn_si = gr.Button("✅ SÍ ES", variant="success")

    input_img.stream(fn=scan_realtime, inputs=[input_img], outputs=[output_html], queue=False)
    btn_si.click(fn=confirmar_carta, outputs=[output_html])
    btn_no.click(fn=rechazar_carta, outputs=[output_html])
    btn_otra_version.click(fn=rechazar_version, outputs=[output_html])

if __name__ == "__main__":
    demo.launch(share=True)