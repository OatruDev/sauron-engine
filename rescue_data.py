import os
import time
import pickle
from dotenv import load_dotenv
from supabase import create_client, Client

# Configuración inicial
load_dotenv()
supabase: Client = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY"))

def rescue_vectors():
    print("Iniciando operación de rescate: Descargando vectores de Supabase...")
    all_cards = []
    batch_size = 1000
    start = 0

    while True:
        print(f"Descargando bloque: {start} a {start + batch_size - 1}...")
        try:
            # Paginación mediante la API de Supabase
            response = supabase.table('sauron_cards').select('*').range(start, start + batch_size - 1).execute()
            data = response.data
            
            if not data:
                break
                
            all_cards.extend(data)
            
            # Si trajo menos del tamaño del bloque, llegamos al final
            if len(data) < batch_size:
                break
                
            start += batch_size
            time.sleep(0.2)  # Pausa ligera para no disparar alertas de límite de peticiones (Rate Limit)
            
        except Exception as e:
            print(f"\nError en el bloque {start}: {e}")
            print("Guardando progreso parcial por seguridad...")
            break

    print(f"\n¡Operación finalizada! Se han descargado {len(all_cards)} cartas.")
    
    print("Comprimiendo y guardando en disco local (sauron_database.pkl)...")
    with open('sauron_database.pkl', 'wb') as f:
        pickle.dump(all_cards, f)
        
    print("✅ Datos asegurados localmente. Ya no dependemos de Supabase.")

if __name__ == "__main__":
    rescue_vectors()