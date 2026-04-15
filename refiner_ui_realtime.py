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
from collections import deque

warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE ARCHIVOS
# ─────────────────────────────────────────────
DB_PKL        = 'sauron_database.pkl'
CSV_LOG       = 'bulk_fast_scan.csv'
RLHF_DIR      = 'dataset_rlhf'
TRANSLATOR_DB = 'sauron_translator.json'
os.makedirs(RLHF_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# PARÁMETROS DE VISIÓN (OpenCV)
# ─────────────────────────────────────────────
WORK_WIDTH     = 640
CANNY_LOW      = 30
CANNY_HIGH     = 120
DILATE_ITER    = 2
APPROX_COEFF   = 0.02
MIN_AREA_RATIO = 0.05
ASPECT_MIN     = 1.25
ASPECT_MAX     = 1.55
OUT_W, OUT_H   = 400, 560

# ─────────────────────────────────────────────
# PARÁMETROS DE FRAME BUFFER (Confidence Lock)
# ─────────────────────────────────────────────
CONFIDENCE_LOCK_FRAMES = 2   # Ajustado a 2 para mayor velocidad de detección
FRAME_BUFFER_SIZE      = 5   


# ─────────────────────────────────────────────
# CARGA DE MODELOS
# ─────────────────────────────────────────────
print("🧠 Cargando Cerebro Visual (CLIP)...")
model = SentenceTransformer('sentence-transformers/clip-ViT-B-32')

print("📖 Cargando Lector de Texto (EasyOCR)...")
reader = easyocr.Reader(['en'], gpu=True, verbose=False)
ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 -,'"

print(f"📦 Cargando Índice FAISS ({DB_PKL})...")
with open(DB_PKL, 'rb') as f:
    cards_data = pickle.load(f)

print("🌐 Preparando Diccionario Global Multilingüe (Con soporte DFC)...")
diccionario_global = {}

# 1. Llenar diccionario con todos los nombres en inglés y dividir las caras (//)
for c in cards_data:
    eng_name = c['name']
    diccionario_global[eng_name] = eng_name
    if '//' in eng_name:
        for part in eng_name.split('//'):
            diccionario_global[part.strip()] = eng_name

# 2. Añadir traducciones
if os.path.exists(TRANSLATOR_DB):
    with open(TRANSLATOR_DB, 'r', encoding='utf-8') as f:
        rosetta_stone = json.load(f)
        for eng_name, foreign_list in rosetta_stone.items():
            for f_name in foreign_list:
                diccionario_global[f_name] = eng_name
                if '//' in f_name:
                    for part in f_name.split('//'):
                        diccionario_global[part.strip()] = eng_name
else:
    print("⚠️  No se encontró la Piedra Rosetta.")
    rosetta_stone = {}

OPCIONES_GLOBALES = list(diccionario_global.keys())

print("⚡ Procesando Vectores y Agrupando Variantes...")
processed_embeddings = []
cartas_por_nombre    = {}

for c in cards_data:
    emb = c['embedding']
    if isinstance(emb, str):
        try:
            emb = json.loads(emb)
        except Exception:
            emb = [float(x) for x in emb.strip("[]").split(",")]

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

# ─────────────────────────────────────────────
# CONTADOR PERSISTENTE MEJORADO
# ─────────────────────────────────────────────
conteo_inicial = 0
if os.path.exists(CSV_LOG):
    try:
        with open(CSV_LOG, 'r', encoding='utf-8') as f:
            lineas = f.readlines()
            if len(lineas) > 1:
                conteo_inicial = len(lineas) - 1
    except Exception as e:
        print(f"⚠️ Error leyendo CSV: {e}")


# ─────────────────────────────────────────────
# ESTADO GLOBAL 
# ─────────────────────────────────────────────
ESTADO = {
    "pausado":            False,
    "id":                 "",
    "nombre":             "",
    "set":                "",
    "nc":                 "",
    "imagen_temp":        None,
    "cooldown_hasta":     0.0,
    "blacklist_nombres":  set(),
    "blacklist_variantes":set(),
    "total_guardadas":    conteo_inicial,
    "frame_history":      deque(maxlen=FRAME_BUFFER_SIZE),
    "lock_id":            None,     
    "lock_carta":         None,     
    "lock_score":         0.0,
    "lock_metodo":        "",
    "lock_latencia":      0.0,
}


# ─────────────────────────────────────────────
# HELPERS DE IMAGEN
# ─────────────────────────────────────────────
def _extract_image_uri(carta: dict) -> str:
    flat = carta.get('image_uri', '')
    if flat: return flat

    image_uris = carta.get('image_uris')
    if isinstance(image_uris, dict):
        return image_uris.get('small') or image_uris.get('normal') or image_uris.get('large') or ''

    card_faces = carta.get('card_faces')
    if isinstance(card_faces, list) and len(card_faces) > 0:
        face_uris = card_faces[0].get('image_uris', {})
        if isinstance(face_uris, dict):
            return face_uris.get('small') or face_uris.get('normal') or face_uris.get('large') or ''

    set_code = carta.get('set_code', '').lower()
    nc = carta.get('collector_number', '')
    if set_code and nc:
        return f"https://api.scryfall.com/cards/{set_code}/{nc}?format=image&version=small"

    return ''


# ─────────────────────────────────────────────
# FUNCIONES DE VISIÓN (OpenCV)
# ─────────────────────────────────────────────
def _order_points(pts):
    pts     = pts.reshape(4, 2).astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered


def detect_and_rectify_card(frame):
    bgr    = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
    h_orig, w_orig = bgr.shape[:2]

    scale  = WORK_WIDTH / w_orig
    w_work = WORK_WIDTH
    h_work = int(h_orig * scale)
    working = cv2.resize(bgr, (w_work, h_work), interpolation=cv2.INTER_AREA)

    gray   = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)

    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 80:
        return None

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)

    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=DILATE_ITER)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = MIN_AREA_RATIO * w_work * h_work
    best_cnt  = None
    best_area = 0.0

    for cnt in contours:
        peri  = cv2.arcLength(cnt, True)
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
            best_cnt  = approx
            best_area = area

    if best_cnt is None: return None

    pts_work = best_cnt.reshape(4, 2).astype(np.float32)
    pts_orig = pts_work * (1.0 / scale)

    src = _order_points(pts_orig)
    dst = np.array([[0, 0], [OUT_W-1, 0], [OUT_W-1, OUT_H-1], [0, OUT_H-1]], dtype=np.float32)

    M      = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(bgr, M, (OUT_W, OUT_H))

    return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))


# ─────────────────────────────────────────────
# TAREAS PARALELAS
# ─────────────────────────────────────────────
def tarea_clip(card_img):
    query_vector = model.encode(card_img).astype('float32').reshape(1, -1)
    faiss.normalize_L2(query_vector)
    D, I = index.search(query_vector, 30)
    return query_vector, D, I


def tarea_ocr(card_img):
    img_cv = np.array(card_img)
    h, w   = img_cv.shape[:2]
    roi_h  = max(int(h * 0.15), 20)
    roi    = img_cv[0:roi_h, 0:w]
    gray   = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    roi_enhanced = clahe.apply(gray)
    resultados_ocr = reader.readtext(roi_enhanced, detail=0, paragraph=True, allowlist=ALLOWLIST)
    return " ".join(resultados_ocr).strip()


# ─────────────────────────────────────────────
# HELPERS DE RENDER HTML
# ─────────────────────────────────────────────
def _html_enfocando() -> str:
    return """
    <div id='sauron-sheet' class='sheet sheet--scanning'>
      <div class='sheet-focus-row'>
        <div class='focus-dot'></div>
        <span class='focus-label'>Enfocando...</span>
      </div>
    </div>
    """


def _html_buscando(msg: str = "Encuadra la carta...") -> str:
    return f"""
    <div id='sauron-sheet' class='sheet sheet--idle'>
      <p class='idle-msg'>{msg}</p>
    </div>
    """


def _html_resultado(carta: dict, score: float, metodo: str, latencia: float) -> str:
    nombre_ingles = carta['name']
    nc            = carta.get('collector_number', '???')
    set_code      = carta.get('set_code', '???').upper()
    image_uri     = _extract_image_uri(carta)          
    set_svg       = carta.get('icon_svg_uri', '')

    traducciones     = rosetta_stone.get(nombre_ingles, [])
    nombre_castellano = ""
    for t in traducciones:
        if t != nombre_ingles:
            nombre_castellano = t
            break

    precio_raw  = carta.get('prices', {}) or {}
    precio_eur  = precio_raw.get('eur') or precio_raw.get('usd') or None
    precio_html = f"<span class='price-tag'>€{precio_eur}</span>" if precio_eur else ""

    img_html = (
        f"<img src='{image_uri}' class='card-thumb' alt='carta'>"
        if image_uri
        else "<div class='card-thumb card-thumb--missing'>?</div>"
    )

    sub_nombre = f"<div class='card-sub'>{nombre_castellano}</div>" if nombre_castellano else ""

    # Inyección del Set Logo
    if set_svg:
        badge_set_html = f"<span class='badge badge--set'><img src='{set_svg}' style='height: 12px; display: inline-block; vertical-align: middle; margin-right: 4px; filter: invert(1);'>{set_code}</span>"
    else:
        badge_set_html = f"<span class='badge badge--set'>{set_code}</span>"

    return f"""
    <div id='sauron-sheet' class='sheet sheet--locked'>
      <div class='sheet-card-row'>

        {img_html}

        <div class='sheet-card-info'>
          <div class='card-name'>{nombre_ingles}</div>
          {sub_nombre}
          <div class='card-meta'>
            {badge_set_html}
            <span class='badge badge--nc'>#{nc}</span>
            {precio_html}
          </div>
          <div class='card-debug'>
            Visual {score*100:.1f}% · {metodo[:28]} · ⚡{latencia:.0f}ms
          </div>
        </div>

      </div>
    </div>
    """


def _html_confirmado(total: int) -> str:
    return f"""
    <div id='sauron-sheet' class='sheet sheet--success'>
      <p class='idle-msg'>✅ Guardada · Total: <strong>{total}</strong> cartas</p>
    </div>
    """


def _html_rechazado() -> str:
    return """
    <div id='sauron-sheet' class='sheet sheet--error'>
      <p class='idle-msg'>❌ Carta incorrecta — escaneando...</p>
    </div>
    """


def _html_otra_version() -> str:
    return """
    <div id='sauron-sheet' class='sheet sheet--scanning'>
      <p class='idle-msg'>🔄 Buscando otra versión...</p>
    </div>
    """


# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL 
# ─────────────────────────────────────────────
def scan_realtime(imagen):
    global ESTADO

    if ESTADO["pausado"] or time.time() < ESTADO["cooldown_hasta"]: return gr.update()
    if imagen is None: return gr.update()

    try:
        t_start = time.time()
        imagen  = ImageOps.mirror(imagen)

        card_img = detect_and_rectify_card(imagen)
        if card_img is None:
            ESTADO["frame_history"].clear()
            ESTADO["lock_id"]    = None
            ESTADO["lock_carta"] = None
            ESTADO["blacklist_nombres"].clear()
            ESTADO["blacklist_variantes"].clear()
            return _html_buscando("Encuadra la carta (Evita moverla)...")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futuro_clip = executor.submit(tarea_clip, card_img)
            futuro_ocr  = executor.submit(tarea_ocr, card_img)
            query_vector, D, I = futuro_clip.result()
            texto_ocr          = futuro_ocr.result()

        if D[0][0] < 0.65:
            ESTADO["frame_history"].clear()
            return _html_buscando("🔴 Escaneando arte...")

        candidatos = [cards_data[idx] for idx in I[0]]
        candidatos_filtrados = [
            c for c in candidatos
            if c['name'] not in ESTADO["blacklist_nombres"]
            and c['id'] not in ESTADO["blacklist_variantes"]
        ]

        if not candidatos_filtrados:
            return _html_buscando("Agotadas las versiones disponibles...")

        # Construir diccionario de búsqueda separando las DFCs
        diccionario_busqueda = {}
        for c in candidatos_filtrados:
            eng_name = c['name']
            diccionario_busqueda[eng_name] = eng_name
            if '//' in eng_name:
                for part in eng_name.split('//'):
                    diccionario_busqueda[part.strip()] = eng_name
                    
            for traduccion in rosetta_stone.get(eng_name, []):
                diccionario_busqueda[traduccion] = eng_name
                if '//' in traduccion:
                    for part in traduccion.split('//'):
                        diccionario_busqueda[part.strip()] = eng_name

        carta_ganadora = None
        metodo_log     = "Visual Puro"
        puntaje_fuzzy  = 0

        if texto_ocr:
            mejor_coincidencia = process.extractOne(query=texto_ocr, choices=list(diccionario_busqueda.keys()), scorer=fuzz.WRatio)
            if mejor_coincidencia:
                nombre_identificado, puntaje_fuzzy, _ = mejor_coincidencia

            if puntaje_fuzzy >= 65:
                nombre_ingles_real = diccionario_busqueda[nombre_identificado]
                carta_ganadora     = next(c for c in candidatos_filtrados if c['name'] == nombre_ingles_real)
                metodo_log         = f"OCR: '{texto_ocr}'"
            else:
                mejor_global = process.extractOne(query=texto_ocr, choices=OPCIONES_GLOBALES, scorer=fuzz.WRatio, score_cutoff=75)
                if mejor_global:
                    nombre_global_id, puntaje_global, _ = mejor_global
                    nombre_ingles_real = diccionario_global[nombre_global_id]

                    if nombre_ingles_real not in ESTADO["blacklist_nombres"]:
                        variantes       = cartas_por_nombre.get(nombre_ingles_real, [])
                        variantes_validas = [v for v in variantes if v['id'] not in ESTADO["blacklist_variantes"]]

                        if variantes_validas:
                            vectores_variantes = [v['parsed_emb'] for v in variantes_validas]
                            matriz_var         = np.array(vectores_variantes).astype('float32')
                            similitudes        = np.dot(matriz_var, query_vector.T).flatten()
                            mejor_var_idx      = np.argmax(similitudes)
                            carta_ganadora     = variantes_validas[mejor_var_idx]
                            puntaje_fuzzy      = puntaje_global
                            metodo_log         = f"OVERRIDE: '{texto_ocr}'"

        if carta_ganadora is None:
            carta_ganadora = candidatos_filtrados[0]

        latencia_ms = (time.time() - t_start) * 1000

        frame_id = carta_ganadora['id']
        ESTADO["frame_history"].append(frame_id)
        consecutive = sum(1 for fid in ESTADO["frame_history"] if fid == frame_id)

        if consecutive < CONFIDENCE_LOCK_FRAMES:
            return _html_enfocando()

        if ESTADO["lock_id"] != frame_id:
            ESTADO["lock_id"]      = frame_id
            ESTADO["lock_carta"]   = carta_ganadora
            ESTADO["lock_score"]   = D[0][0]
            ESTADO["lock_metodo"]  = metodo_log
            ESTADO["lock_latencia"] = latencia_ms

            ESTADO["imagen_temp"] = card_img
            ESTADO["nombre"]      = carta_ganadora['name']
            ESTADO["set"]         = carta_ganadora['set_code']
            ESTADO["nc"]          = carta_ganadora.get('collector_number', '???')
            ESTADO["id"]          = frame_id
            ESTADO["pausado"] = True

        return _html_resultado(carta_ganadora, D[0][0], metodo_log, latencia_ms)

    except Exception as e:
        print(f"Error procesando frame: {e}")
        return gr.update()


# ─────────────────────────────────────────────
# ACCIONES DE USUARIO
# ─────────────────────────────────────────────
def confirmar_carta():
    global ESTADO
    if not ESTADO["nombre"] or ESTADO["imagen_temp"] is None: return gr.update()

    name, set_code, nc = ESTADO["nombre"], ESTADO["set"], ESTADO["nc"]
    fecha_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    set_dir   = os.path.join(RLHF_DIR, set_code)
    os.makedirs(set_dir, exist_ok=True)

    ruta_imagen = os.path.join(set_dir, f"{nc}_{name.replace(' ', '_').replace('/', '-')}_{fecha_str}.jpg")
    ESTADO["imagen_temp"].save(ruta_imagen)

    nuevo_registro = pd.DataFrame(
        [[datetime.now().strftime("%Y-%m-%d %H:%M:%S"), name, set_code, nc, ruta_imagen]],
        columns=["Fecha", "Nombre", "Set", "NC", "Ruta_Imagen"]
    )
    nuevo_registro.to_csv(CSV_LOG, mode='a', index=False, header=not os.path.exists(CSV_LOG))

    ESTADO["total_guardadas"] += 1
    total = ESTADO["total_guardadas"]

    ESTADO["pausado"]             = False
    ESTADO["cooldown_hasta"]      = time.time() + 1.5
    ESTADO["blacklist_nombres"].clear()
    ESTADO["blacklist_variantes"].clear()
    ESTADO["frame_history"].clear()
    ESTADO["lock_id"]             = None
    ESTADO["lock_carta"]          = None
    return _html_confirmado(total)


def rechazar_carta():
    global ESTADO
    if ESTADO["nombre"]: ESTADO["blacklist_nombres"].add(ESTADO["nombre"])
    ESTADO["pausado"]        = False
    ESTADO["cooldown_hasta"] = time.time() + 0.5
    ESTADO["frame_history"].clear()
    ESTADO["lock_id"]    = None
    ESTADO["lock_carta"] = None
    return _html_rechazado()


def rechazar_version():
    global ESTADO
    if ESTADO["id"]: ESTADO["blacklist_variantes"].add(ESTADO["id"])
    ESTADO["pausado"]        = False
    ESTADO["cooldown_hasta"] = time.time() + 0.2
    ESTADO["frame_history"].clear()
    ESTADO["lock_id"]    = None
    ESTADO["lock_carta"] = None
    return _html_otra_version()


# ─────────────────────────────────────────────
# CSS  —  Fix Overflow y Alturas
# ─────────────────────────────────────────────
CSS = """
* { box-sizing: border-box; }
.gradio-container { padding: 0 !important; margin: 0 !important; background: #0d0d0d !important; max-width: 100vw !important; }

#sauron-camera { position: relative; width: 100%; background: #000; }
#sauron-camera video, #sauron-camera img { 
  width: 100% !important; 
  max-height: 45vh !important;
  object-fit: cover !important; 
  display: block; transform: none !important; border-radius: 0 !important; 
}

#sauron-sheet-wrapper { width: 100%; background: transparent; }
#sauron-sheet-wrapper > .block { padding: 0 !important; background: transparent !important; }

.sheet {
  width: 100%; background: #1a1a1a; border-radius: 20px 20px 0 0;
  padding: 10px 12px 6px; 
  border-top: 1px solid #2e2e2e;
  min-height: 75px; 
  transition: border-color .2s;
}

.sheet--idle    { border-top-color: #333; }
.sheet--scanning{ border-top-color: #2196F3; }
.sheet--locked  { border-top-color: #4caf50; }
.sheet--success { border-top-color: #4caf50; background: #0d2614; }
.sheet--error   { border-top-color: #f44336; background: #1e0a0a; }

.sheet::before { content: ''; display: block; width: 40px; height: 4px; background: #444; border-radius: 2px; margin: 0 auto 10px; }

.sheet-card-row { display: flex; align-items: center; gap: 14px; }

.card-thumb { width: 50px; height: 70px; object-fit: cover; border-radius: 5px; box-shadow: 0 3px 10px rgba(0,0,0,.6); flex-shrink: 0; }
.card-thumb--missing { background: #333; display: flex; align-items: center; justify-content: center; color: #666; font-size: 20px; }

.sheet-card-info { flex: 1; min-width: 0; }
.card-name { font-size: 15px; font-weight: 700; color: #ffffff; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.card-sub { font-size: 12px; color: #4caf50; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-top: 1px; }

.card-meta { display: flex; align-items: center; flex-wrap: wrap; gap: 6px; margin-top: 4px; }
.badge { padding: 2px 6px; border-radius: 20px; font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: .5px; }
.badge--set { background: #2e2e2e; color: #eee; display: inline-flex; align-items: center; }
.badge--nc  { background: #1e3a1e; color: #7dcf7d; }
.price-tag { background: #1a2a3a; color: #64b5f6; padding: 2px 6px; border-radius: 20px; font-size: 10px; font-weight: 700; }

.card-debug { font-size: 9px; color: #555; font-family: monospace; margin-top: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

.sheet-focus-row { display: flex; align-items: center; gap: 10px; padding: 4px 0; }
.focus-dot { width: 10px; height: 10px; background: #2196F3; border-radius: 50%; animation: pulse 1s infinite; }
@keyframes pulse { 0%,100% { opacity: 1; transform: scale(1); } 50% { opacity: .4; transform: scale(.7); } }
.focus-label { color: #2196F3; font-size: 14px; font-weight: 600; }

.idle-msg { text-align: center; font-size: 14px; font-weight: 600; color: #aaa; margin: 4px 0 0; }
.sheet--success .idle-msg { color: #4caf50; }
.sheet--error   .idle-msg { color: #f44336; }
.sheet--scanning .idle-msg{ color: #2196F3; }

#sauron-actions { width: 100%; background: #1a1a1a; padding: 8px 10px 10px; display: flex; gap: 8px; } 
#sauron-actions .block { padding: 0 !important; }

#btn-no, #btn-otra, #btn-si {
  min-height: 44px !important; 
  font-size: 14px !important;
  font-weight: 800 !important; border-radius: 12px !important; border: none !important; cursor: pointer; width: 100% !important; letter-spacing: .3px;
}
#btn-no   { background: #2a2a2a !important; color: #f44336 !important; border: 1.5px solid #f44336 !important; }
#btn-no:hover { background: #3a1010 !important; }
#btn-otra { background: #1565C0 !important; color: #fff !important; }
#btn-otra:hover { background: #1976D2 !important; }
#btn-si   { background: #2e7d32 !important; color: #fff !important; }
#btn-si:hover { background: #388e3c !important; }

.gradio-container .gap { gap: 0 !important; }
"""

# ─────────────────────────────────────────────
# INTERFAZ GRADIO
# ─────────────────────────────────────────────
_initial_html = _html_buscando(f"Iniciando... ({ESTADO['total_guardadas']} cartas almacenadas)")

with gr.Blocks(title="SAURON · MTG Scanner") as demo:

    with gr.Row(elem_id="sauron-camera"):
        input_img = gr.Image(sources=["webcam"], streaming=True, type="pil", show_label=False)

    with gr.Row(elem_id="sauron-sheet-wrapper"):
        output_html = gr.HTML(value=_initial_html, elem_id="sauron-sheet-outer")

    with gr.Row(elem_id="sauron-actions"):
        btn_no   = gr.Button("❌  NO ES", variant="secondary", elem_id="btn-no")
        btn_otra = gr.Button("🔄  OTRA VERSIÓN", variant="primary", elem_id="btn-otra")
        btn_si   = gr.Button("✅  SÍ ES", variant="success", elem_id="btn-si")

    input_img.stream(fn=scan_realtime, inputs=[input_img], outputs=[output_html], queue=False)
    btn_si.click(fn=confirmar_carta, outputs=[output_html])
    btn_no.click(fn=rechazar_carta, outputs=[output_html])
    btn_otra.click(fn=rechazar_version, outputs=[output_html])


if __name__ == "__main__":
    demo.launch(share=True, css=CSS)