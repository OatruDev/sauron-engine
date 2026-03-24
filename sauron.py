import requests
import os
import time
from io import BytesIO
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from dotenv import load_dotenv

# Cargar las variables secretas desde el archivo .env
load_dotenv()

# --- CONFIGURACIÓN ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
BATCH_SIZE = 50
IMAGE_CACHE_DIR = "cache"
PROCESSED_LOG = "processed_ids.txt"
FAILED_LOG = "failed_cards.log"

# Asegurar que los directorios y archivos existen
os.makedirs(IMAGE_CACHE_DIR, exist_ok=True)
for file in [PROCESSED_LOG, FAILED_LOG]:
    if not os.path.exists(file):
        open(file, 'w').close()

HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "resolution=merge-duplicates"
}

# --- INICIALIZACIÓN ---
print("👁️ Cargando inteligencia visual (CLIP)...")
model = SentenceTransformer('clip-ViT-B-32')

def get_processed_ids():
    with open(PROCESSED_LOG, "r") as f:
        return set(line.strip() for line in f)

def log_processed(card_id):
    with open(PROCESSED_LOG, "a") as f:
        f.write(f"{card_id}\n")

def log_failed(card_id, error):
    with open(FAILED_LOG, "a") as f:
        f.write(f"{card_id}: {error}\n")

def run_ingestion():
    processed_ids = get_processed_ids()
    
    print("📦 Obteniendo catálogo maestro de Scryfall...")
    bulk_info = requests.get("https://api.scryfall.com/bulk-data/default-cards").json()
    all_cards = requests.get(bulk_info["download_uri"]).json()
    
    # Filtramos cartas válidas que no hayamos procesado ya
    queue = [c for c in all_cards if 'image_uris' in c and 'normal' in c['image_uris'] and c['id'] not in processed_ids]
    
    print(f"🚀 Preparado para procesar {len(queue)} cartas pendientes.")
    
    batch = []
    
    for card in tqdm(queue, desc="Ingestando a SAURON", unit="carta"):
        card_id = card['id']
        img_url = card['image_uris']['normal']
        cache_path = os.path.join(IMAGE_CACHE_DIR, f"{card_id}.jpg")
        
        try:
            # 1. Obtener Imagen (Caché local o descarga)
            if os.path.exists(cache_path):
                img = Image.open(cache_path)
            else:
                img_data = requests.get(img_url, timeout=15).content
                img = Image.open(BytesIO(img_data))
                img.save(cache_path)
            
            # 2. Generar Vector CLIP
            embedding = model.encode(img, convert_to_tensor=False).tolist()
            
            # 3. Añadir al lote
            batch.append({
                "id": card_id,
                "name": card['name'],
                "set_code": card.get('set', 'unknown'),
                "image_url": img_url,
                "embedding": embedding
            })
            
            # 4. Envío masivo a Supabase
            if len(batch) >= BATCH_SIZE:
                endpoint = f"{SUPABASE_URL}/rest/v1/sauron_cards"
                res = requests.post(endpoint, headers=HEADERS, json=batch)
                
                if res.status_code in (200, 201):
                    for c in batch: 
                        log_processed(c['id'])
                    batch = []
                else:
                    print(f"\n❌ Error en batch: {res.text}")
                    time.sleep(5)
                    
        except Exception as e:
            log_failed(card_id, str(e))
            continue

    # Procesar cualquier carta residual que no haya completado un bloque de 50
    if batch:
        endpoint = f"{SUPABASE_URL}/rest/v1/sauron_cards"
        res = requests.post(endpoint, headers=HEADERS, json=batch)
        if res.status_code in (200, 201):
            for c in batch: 
                log_processed(c['id'])

if __name__ == "__main__":
    run_ingestion()
    print("\n✅ Ingesta finalizada.")