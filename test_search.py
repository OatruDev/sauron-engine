import os
import time
from dotenv import load_dotenv
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from PIL import Image

# 1. Configuración de conexión
load_dotenv()
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

print("👁️ SAURON Online (Modelo CLIP cargado)...")
model = SentenceTransformer('sentence-transformers/clip-ViT-B-32')

def test_vision(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Error: No encuentro el archivo '{image_path}' en la carpeta.")
        return

    print(f"\n🔍 Analizando: {image_path}...")
    start_time = time.time()
    
    # 2. Cargar y procesar imagen
    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"❌ Error al abrir imagen: {e}")
        return

    # 3. Extraer ADN visual
    # CLIP genera vectores de 512 dimensiones
    query_vector = model.encode(img).tolist()

    # 4. Búsqueda Vectorial en Supabase
    print("⚡ Consultando índice optimizado (108,622 cartas)...")
    db_start = time.time()
    
    # Llamamos a la función RPC que creamos en el SQL Editor
    response = supabase.rpc(
        'match_cards', 
        {
            'query_embedding': query_vector,
            'match_threshold': 0.60, # Umbral del 60% para fotos reales
            'match_count': 5         # Top 5 resultados
        }
    ).execute()
    
    db_time = time.time() - db_start
    total_time = time.time() - start_time

    # 5. Reporte de Resultados
    print(f"\n✅ Escaneo completado en {total_time:.2f}s (Latencia BD: {db_time:.2f}s)")
    print("-" * 55)
    
    if response.data:
        for i, card in enumerate(response.data, 1):
            accuracy = card['similarity'] * 100
            print(f"#{i} | {card['name']} [{card['set_code'].upper()}] | Exactitud: {accuracy:.1f}%")
            print(f"   🔗 Imagen: {card['image_url']}\n")
    else:
        print("🤷 No se encontraron coincidencias claras. Prueba con mejor luz.")

if __name__ == "__main__":
    test_vision("prueba.jpg")