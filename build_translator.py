import os
import json
import requests
import ijson

def build_rosetta_stone():
    print("🌍 Consultando Scryfall All Cards...")
    bulk_info = requests.get("https://api.scryfall.com/bulk-data/all-cards").json()
    url = bulk_info["download_uri"]
    
    # 1. Descarga inteligente (no repite el trabajo si ya falló antes en el procesamiento)
    if not os.path.exists("temp_all.json"):
        print("⬇️ Descargando base de datos multilingüe (Esto pesa ~1.8 GB)...")
        r = requests.get(url, stream=True)
        with open("temp_all.json", "wb") as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                f.write(chunk)
    else:
        print("✅ El archivo temp_all.json ya existe. Saltando descarga.")
            
    print("🧠 Procesando idiomas por goteo (Cero impacto en RAM)...")
    rosetta = {}
    
    # 2. La magia del Streaming: ijson lee el archivo carta por carta y limpia la memoria
    with open("temp_all.json", "rb") as f:
        # 'item' le dice a ijson que extraiga cada objeto del array principal
        for c in ijson.items(f, 'item'):
            if c.get("lang") != "en" and "printed_name" in c and "name" in c:
                eng_name = c["name"]
                foreign_name = c["printed_name"]
                
                if eng_name not in rosetta:
                    rosetta[eng_name] = []
                if foreign_name not in rosetta[eng_name]:
                    rosetta[eng_name].append(foreign_name)
                
    print("💾 Guardando sauron_translator.json (~20 MB)...")
    with open("sauron_translator.json", "w", encoding="utf-8") as f:
        json.dump(rosetta, f)
        
    print("🗑️ Limpiando el archivo gigante...")
    os.remove("temp_all.json")
    
    print(f"✅ ¡Piedra Rosetta creada! Contiene traducciones para {len(rosetta)} cartas base.")

if __name__ == "__main__":
    build_rosetta_stone()