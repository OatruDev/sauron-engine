import pickle
import requests
import time

def patch_collector_numbers():
    print("🌍 Descargando datos maestros de Scryfall (~30MB)...")
    bulk_info = requests.get("https://api.scryfall.com/bulk-data/default-cards").json()
    scryfall_cards = requests.get(bulk_info["download_uri"]).json()

    print("🧠 Creando mapa de IDs a Collector Numbers...")
    id_to_cn = {c['id']: c.get('collector_number', '???') for c in scryfall_cards}

    print("📦 Cargando tu sauron_database.pkl local...")
    with open('sauron_database.pkl', 'rb') as f:
        cards_data = pickle.load(f)

    print("💉 Inyectando Collector Numbers...")
    for c in cards_data:
        c['collector_number'] = id_to_cn.get(c['id'], 'N/A')

    print("💾 Guardando nueva base de datos...")
    with open('sauron_database.pkl', 'wb') as f:
        pickle.dump(cards_data, f)

    print("✅ ¡Parche aplicado con éxito! Tu base de datos ya soporta Variantes.")

if __name__ == "__main__":
    patch_collector_numbers()