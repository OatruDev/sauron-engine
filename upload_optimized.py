import os
import pickle
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()
supabase: Client = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY"))

def upload_optimized():
    print("Leyendo base de datos local...")
    with open('sauron_database.pkl', 'rb') as f:
        cards_data = pickle.load(f)

    total = len(cards_data)
    batch_size = 500
    print(f"Subiendo {total} cartas en bloques de {batch_size}...")

    for i in range(0, total, batch_size):
        batch = cards_data[i:i + batch_size]
        
        # Preparamos los datos para la tabla optimizada
        to_insert = [
            {
                "name": c['name'],
                "set_code": c['set_code'],
                "image_url": c['image_url'],
                "embedding": c['embedding'] # El cliente de Supabase se encarga del formato
            } for c in batch
        ]
        
        try:
            supabase.table('sauron_cards').insert(to_insert).execute()
            print(f"✅ Insertadas: {i + len(batch)} / {total}")
        except Exception as e:
            print(f"❌ Error en bloque {i}: {e}")

if __name__ == "__main__":
    upload_optimized()