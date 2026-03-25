import os
import time
from dotenv import load_dotenv
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from PIL import Image

# Cargar variables de entorno
load_dotenv()
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

print("👁️ Despertando a SAURON (Cargando modelo CLIP)...")
model = SentenceTransformer('sentence-transformers/clip-ViT-B-32')

def test_vision(image_path):
    print(f"\n🔍 Analizando la imagen: {image_path}...")
    start_time = time.time()
    
    # 1. Cargar imagen local
    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"❌ Error al abrir la imagen: {e}")
        return

    # 2. Convertir imagen a vector
    print("🧠 Extrayendo ADN visual...")
    query_vector = model.encode(img).tolist()

    # 3. Consultar a Supabase mediante la función RPC
    print("⚡ Buscando coincidencias en 108,622 cartas...")
    db_start = time.time()
    response = supabase.rpc(
        'match_cards', 
        {
            'query_embedding': query_vector,
            'match_threshold': 0.65, # Similitud mínima del 65%
            'match_count': 5         # Queremos el Top 5
        }
    ).execute()
    db_time = time.time() - db_start

    # 4. Mostrar Resultados
    total_time = time.time() - start_time
    print(f"\n✅ Búsqueda completada en {total_time:.2f} segundos (Tiempo BD: {db_time:.2f}s)")
    print("-" * 50)
    
    if response.data:
        for i, card in enumerate(response.data, 1):
            similitud_porcentaje = card['similarity'] * 100
            print(f"#{i} | {card['name']} (Set: {card['set_code'].upper()}) | Similitud: {similitud_porcentaje:.1f}%")
            print(f"    🔗 Ver imagen: {card['image_url']}\n")
    else:
        print("No se encontraron cartas que superen el umbral de similitud.")

if __name__ == "__main__":
    # Cambia 'prueba.jpg' por el nombre de la imagen que quieras probar
    IMAGEN_DE_PRUEBA = "prueba.jpg" 
    test_vision(IMAGEN_DE_PRUEBA)