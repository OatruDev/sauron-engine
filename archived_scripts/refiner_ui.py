import os
import gradio as gr
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from PIL import Image

# Configuración
load_dotenv()
supabase: Client = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY"))
model = SentenceTransformer('sentence-transformers/clip-ViT-B-32')
CSV_FILE = "bulk_validado.csv"

def guardar_seleccion(nombre, set_code):
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    nuevo_registro = pd.DataFrame([[fecha, nombre, set_code]], columns=["Fecha", "Nombre", "Set"])
    
    # Guardar en CSV (si no existe, crea cabeceras)
    nuevo_registro.to_csv(CSV_FILE, mode='a', index=False, header=not os.path.exists(CSV_FILE))
    return f"✅ ¡Guardada! {nombre} [{set_code}] añadido al archivo."

def buscar_carta_ui(imagen):
    if imagen is None: return "Sube una foto", [], ""
    
    # Vectorización y Búsqueda
    query_vector = model.encode(imagen).tolist()
    res = supabase.rpc('match_cards', {
        'query_embedding': query_vector, 'match_threshold': 0.6, 'match_count': 3
    }).execute()
    
    if not res.data:
        return "No hay coincidencias claras.", [], ""

    # Preparar visualización
    top_card = res.data[0]
    info = f"### Top 1: {top_card['name']} [{top_card['set_code'].upper()}]\nExactitud: {top_card['similarity']*100:.1f}%"
    
    return info, top_card['name'], top_card['set_code']

# Construir Interfaz
with gr.Blocks(title="SAURON Bulk Scanner") as demo:
    gr.Markdown("# 👁️ SAURON - Escáner de Bulk")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="Cámara del móvil / Foto")
            btn_scan = gr.Button("🔍 ESCANEAR CARTA", variant="primary")
        
        with gr.Column():
            output_info = gr.Markdown("Resultados aparecerán aquí...")
            confirm_btn = gr.Button("💾 CONFIRMAR Y GUARDAR EN BULK", variant="success")
            status_msg = gr.Label(label="Estado")

    # Estados invisibles para guardar info
    card_name = gr.State("")
    card_set = gr.State("")

    # Lógica
    btn_scan.click(buscar_carta_ui, inputs=[input_img], outputs=[output_info, card_name, card_set])
    confirm_btn.click(guardar_seleccion, inputs=[card_name, card_set], outputs=[status_msg])

if __name__ == "__main__":
    # share=True genera el link público para tu celular
    demo.launch(share=True)