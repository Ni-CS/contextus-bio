import streamlit as st
import pandas as pd
import numpy as np
import faiss
from PIL import Image
import fitz  # PyMuPDF
import io
import os
import torch

# --- CONFIGURAÇÕES GLOBAIS ---
PDF_PATH = "data/Biologia-Celular.pdf"
DB_BASE_PATH = "database"

APPROACHES = {
    "1: Descrições (Especialista em Texto)": {
        "id": "approach_1_text_only",
        "model_id": "intfloat/multilingual-e5-large",
    },
    "2: Híbrida (CLIP Visual + Texto)": {
        "id": "approach_2_hybrid_clip",
        "text_model_id": "intfloat/multilingual-e5-large",
        "image_model_id": "openai/clip-vit-large-patch14",
    },
    "3: Multimodal Pura (Tudo com CLIP)": {
        "id": "approach_3_pure_clip",
        "model_id": "openai/clip-vit-large-patch14",
    },
    "4: CLIP Híbrido Unificado": { # <-- NOVA ABORDAGEM
        "id": "approach_4_unified_hybrid",
        "model_id": "openai/clip-vit-large-patch14",
    }
}

# --- FUNÇÕES DE CARREGAMENTO (EM CACHE PARA PERFORMANCE) ---
# O cache agora é feito por abordagem, garantindo que o carregamento pesado rode apenas uma vez por seleção.
@st.cache_resource
def load_selected_approach(approach_id):
    """
    Carrega todos os modelos, dados e índices para a abordagem selecionada.
    Esta função é envolvida por um spinner para feedback ao usuário.
    """
    from sentence_transformers import SentenceTransformer
    from transformers import CLIPModel, CLIPProcessor

    approach_config = next((config for name, config in APPROACHES.items() if config['id'] == approach_id), None)
    
    # Carrega os modelos necessários para a abordagem
    text_model_id = approach_config.get("text_model_id", approach_config.get("model_id"))
    image_model_id = approach_config.get("image_model_id", approach_config.get("model_id"))
    
    print(f"Carregando modelos para: {approach_id}")
    text_model_data = get_model(text_model_id)
    image_model_data = get_model(image_model_id)

    # Carrega a base de dados
    db_path = os.path.join(DB_BASE_PATH, approach_id)
    print(f"Carregando base de dados de: {db_path}")
    try:
        text_df = pd.read_csv(os.path.join(db_path, 'text_data.csv'))
        text_embeddings = np.load(os.path.join(db_path, 'text_embeddings.npy')).astype('float32')
        text_index = faiss.IndexFlatIP(text_embeddings.shape[1])
        faiss.normalize_L2(text_embeddings)
        text_index.add(text_embeddings)

        image_df = pd.read_csv(os.path.join(db_path, 'image_data.csv'))
        image_embeddings = np.load(os.path.join(db_path, 'image_embeddings.npy')).astype('float32')
        image_index = faiss.IndexFlatIP(image_embeddings.shape[1])
        faiss.normalize_L2(image_embeddings)
        image_index.add(image_embeddings)
        
        engines = {
            "text": {"df": text_df, "index": text_index},
            "image": {"df": image_df, "index": image_index}
        }
        print("Carregamento concluído.")
        return text_model_data, image_model_data, engines
        
    except FileNotFoundError:
        return None, None, None

@st.cache_resource
def get_model(model_id):
    """Função auxiliar para carregar um modelo específico."""
    from sentence_transformers import SentenceTransformer
    from transformers import CLIPModel, CLIPProcessor
    print(f"Carregando ou obtendo do cache: {model_id}...")
    if "clip" in model_id.lower():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CLIPModel.from_pretrained(model_id).to(device)
        processor = CLIPProcessor.from_pretrained(model_id)
        return {"type": "clip", "model": model, "processor": processor, "device": device}
    else:
        model = SentenceTransformer(model_id)
        return {"type": "sentence_transformer", "model": model}

@st.cache_data
def render_pdf_page(page_number, text_to_highlight=None):
    # ... (função idêntica à anterior) ...
    doc = fitz.open(PDF_PATH)
    page = doc.load_page(page_number - 1)
    if text_to_highlight:
        text_instances = page.search_for(text_to_highlight)
        for inst in text_instances:
            page.add_highlight_annot(inst)
    pix = page.get_pixmap(dpi=200)
    img_bytes = pix.tobytes("png")
    doc.close()
    return Image.open(io.BytesIO(img_bytes))

# --- SETUP DA APLICAÇÃO ---
if 'current_page' not in st.session_state: st.session_state.current_page = 1
if 'text_to_highlight' not in st.session_state: st.session_state.text_to_highlight = None
if 'search_active' not in st.session_state: st.session_state.search_active = False

def set_page_view(page_num, text=None):
    st.session_state.current_page = int(page_num)
    st.session_state.text_to_highlight = text

# --- INTERFACE GRÁFICA ---
st.set_page_config(layout="wide")
st.title("🔬 Painel Comparativo de Buscas Semânticas")

with st.sidebar:
    st.header("Configuração da Busca")
    selected_approach_name = st.radio("Escolha a abordagem:", list(APPROACHES.keys()), key="approach_selector")
    approach_id_to_load = APPROACHES[selected_approach_name]['id']

# --- NOVO: LÓGICA DE CARREGAMENTO COM SPINNER ---
with st.spinner(f"Carregando a abordagem '{selected_approach_name}'... Isso pode levar um momento."):
    text_model_data, image_model_data, engines = load_selected_approach(approach_id_to_load)

# A aplicação continua apenas se o carregamento for bem-sucedido
if not engines:
    st.error(f"Base de dados para a abordagem '{selected_approach_name}' não encontrada. Por favor, execute o script de construção correspondente.")
else:
    # O restante da aplicação é renderizado aqui
    col1, col2 = st.columns([1, 2])

    with col1:
        # ... (código da coluna de controles e resultados idêntico ao anterior) ...
        st.header("Controles")
        query = st.text_input("Busque por um conceito:", "metamorfose do girino", key="query_input")
        if st.button("Buscar") and query:
            # ... (código de busca e auto-jump idêntico ao anterior) ...
            st.session_state.search_active = True
            if text_model_data["type"] == "sentence_transformer":
                text_query_vector = text_model_data["model"].encode(["query: " + query]).astype('float32')
            else:
                with torch.no_grad():
                    inputs = text_model_data["processor"](text=query, return_tensors="pt").to(text_model_data["device"])
                    text_query_vector = text_model_data["model"].get_text_features(**inputs).cpu().numpy().astype('float32')
            faiss.normalize_L2(text_query_vector)

            with torch.no_grad():
                if image_model_data["type"] == "clip":
                    inputs = image_model_data["processor"](text=query, return_tensors="pt").to(image_model_data["device"])
                    image_query_vector = image_model_data["model"].get_text_features(**inputs).cpu().numpy().astype('float32')
                else:
                    image_query_vector = image_model_data["model"].encode(["query: " + query]).astype('float32')
            faiss.normalize_L2(image_query_vector)

            text_eng, img_eng = engines["text"], engines["image"]
            text_distances, text_indices = text_eng["index"].search(text_query_vector, 5)
            img_distances, img_indices = img_eng["index"].search(image_query_vector, 5)

            st.session_state.text_results = text_eng["df"].iloc[text_indices[0]]
            st.session_state.text_scores = text_distances[0]
            st.session_state.image_results = img_eng["df"].iloc[img_indices[0]]
            st.session_state.image_scores = img_distances[0]
            
            top_image_score = img_distances[0][0] if len(img_indices[0]) > 0 else -1
            top_text_score = text_distances[0][0] if len(text_indices[0]) > 0 else -1
            
            if top_image_score > top_text_score:
                set_page_view(st.session_state.image_results.iloc[0]['page'])
            elif top_text_score != -1:
                top_result = st.session_state.text_results.iloc[0]
                set_page_view(top_result['page'], ' '.join(top_result['original_text'].split()[:40]))
            
            st.rerun()

        if 'search_active' in st.session_state and st.session_state.search_active:
             # (código para exibir os botões de resultado)
            st.divider()
            st.subheader("Resultados da Busca")
            st.write("**🖼️ Imagens Relevantes:**")
            if 'image_results' in st.session_state:
                for i, (index, row) in enumerate(st.session_state.image_results.iterrows()):
                    score = st.session_state.image_scores[i]
                    if st.button(f"Página {int(row['page'])} (Score: {score:.4f})", key=f"img_{index}"):
                        set_page_view(row['page'])
                        st.rerun()
            
            st.write("**📝 Textos Relevantes:**")
            if 'text_results' in st.session_state:
                for i, (index, row) in enumerate(st.session_state.text_results.iterrows()):
                    score = st.session_state.text_scores[i]
                    highlight_text = ' '.join(row['original_text'].split()[:40])
                    if st.button(f"Página {int(row['page'])}: '{row['description_llm'][:40]}...' (Score: {score:.4f})", key=f"txt_{index}"):
                        set_page_view(row['page'], highlight_text)
                        st.rerun()

    with col2:
        # ... (código da coluna do visualizador idêntico ao anterior) ...
        st.header(f"Visualizador - Página {st.session_state.current_page}")
        current_highlight = st.session_state.text_to_highlight if 'search_active' in st.session_state and st.session_state.search_active else None
        try:
            page_image = render_pdf_page(page_number=st.session_state.current_page, text_to_highlight=current_highlight)
            st.image(page_image, use_container_width=True)
        except Exception as e:
            st.error(f"Não foi possível renderizar a página. Erro: {e}")