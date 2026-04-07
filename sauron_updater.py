import os
import pickle
import time
import requests
import numpy as np
from io import BytesIO
from PIL import Image, ImageFile
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client
from dotenv import load_dotenv
from tqdm import tqdm

# --- CONFIGURACIÓN PARA IMÁGENES INCOMPLETAS ---
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- CONFIGURACIÓN GENERAL ---
load_dotenv()
supabase: Client = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY"))
DB_PKL = 'sauron_database.pkl'

def update_sauron():
    print("📦 Leyendo base de datos local...")
    if not os.path.exists(DB_PKL):
        print(f"❌ Error: No se encuentra {DB_PKL}.")
        return

    with open(DB_PKL, 'rb') as f:
        cards_data = pickle.load(f)
    
    existing_ids = {c['id'] for c in cards_data if 'id' in c}
    print(f"ℹ️ Actualmente tienes {len(existing_ids)} cartas registradas.")

    print("\n🔍 Consultando Scryfall por actualizaciones...")
    bulk_info = requests.get("https://api.scryfall.com/bulk-data/default-cards").json()
    all_scryfall_cards = requests.get(bulk_info["download_uri"]).json()
    
    new_cards_raw = [
        c for c in all_scryfall_cards 
        if 'image_uris' in c and 'normal' in c['image_uris'] and c['id'] not in existing_ids
    ]
    
    if not new_cards_raw:
        print("✅ SAURON ya está actualizado. No hay cartas nuevas en Scryfall.")
        return

    print(f"🚀 Se encontraron {len(new_cards_raw)} cartas nuevas.")
    
    print("🧠 Cargando modelo CLIP...")
    model = SentenceTransformer('sentence-transformers/clip-ViT-B-32')
    
    new_processed_cards = []
    batch_size = 50
    
    for card in tqdm(new_cards_raw, desc="Vectorizando cartas", unit="carta"):
        try:
            img_url = card['image_uris']['normal']
            img_data = requests.get(img_url, timeout=15).content
            img = Image.open(BytesIO(img_data))
            
            embedding = model.encode(img).astype('float32').tolist()
            
            processed_card = {
                "id": card['id'],
                "name": card['name'],
                "set_code": card.get('set', 'unknown'),
                "image_url": img_url,
                "embedding": embedding
            }
            new_processed_cards.append(processed_card)
            
            time.sleep(0.1) 
            
        except Exception as e:
            print(f"\n❌ Error procesando {card['name']}: {e}")
            continue
            
    if not new_processed_cards:
        print("\n⚠️ No se procesó con éxito ninguna carta nueva.")
        return

    print("\n☁️ Subiendo nuevas cartas a Supabase...")
    for i in range(0, len(new_processed_cards), batch_size):
        batch = new_processed_cards[i:i + batch_size]
        try:
            supabase.table('sauron_cards').insert(batch).execute()
        except Exception as e:
            print(f"❌ Error subiendo bloque a Supabase: {e}")

    print("\n💾 Integrando datos al archivo sauron_database.pkl...")
    cards_data.extend(new_processed_cards)
    
    with open(DB_PKL, 'wb') as f:
        pickle.dump(cards_data, f)
        
    print(f"✅ Actualización completa. SAURON ahora tiene {len(cards_data)} cartas en total.")

if __name__ == "__main__":
    update_sauron()