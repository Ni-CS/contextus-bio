import streamlit as st
import pandas as pd
import numpy as np
import faiss
from PIL import Image
import fitz  # PyMuPDF
import io
import os
import torch
import json
from datetime import datetime
import threading

# --- CONFIGURAÇÕES GLOBAIS ---
PDF_PATH = "data/Biologia-Celular.pdf"
DB_BASE_PATH = "database"
EVALUATION_FILE = "evaluations.json"

APPROACHES = {
    "1: Descrições (Especialista em Texto)": {"id": "approach_1_text_only", "model_id": "intfloat/multilingual-e5-large"},
    "2: Híbrida (CLIP Visual + Texto)": {"id": "approach_2_hybrid_clip", "text_model_id": "intfloat/multilingual-e5-large", "image_model_id": "openai/clip-vit-large-patch14"},
    "3: Multimodal Pura (Tudo com CLIP)": {"id": "approach_3_pure_clip", "model_id": "openai/clip-vit-large-patch14"},
    "4: CLIP Híbrido Unificado": {"id": "approach_4_unified_hybrid", "model_id": "openai/clip-vit-large-patch14"}
}

# Usa um Lock para evitar condições de corrida ao salvar o arquivo JSON
file_lock = threading.Lock()

# --- FUNÇÕES DE CARREGAMENTO (EM CACHE) ---
@st.cache_resource
def get_model(model_id):
    from sentence_transformers import SentenceTransformer
    from transformers import CLIPModel, CLIPProcessor
    print(f"Carregando ou obtendo do cache: {model_id}...")
    if "clip" in model_id.lower():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CLIPModel.from_pretrained(model_id).to(device)
        processor = CLIPProcessor.from_pretrained(model_id)
        return {"type": "clip", "model": model, "processor": processor, "device": device}
    else:
        return {"type": "sentence_transformer", "model": SentenceTransformer(model_id)}

@st.cache_data
def load_database_for_approach(approach_id):
    db_path = os.path.join(DB_BASE_PATH, approach_id)
    print(f"Carregando base de dados de: {db_path}")
    try:
        text_df = pd.read_csv(os.path.join(db_path, 'text_data.csv'))
        text_embeddings = np.load(os.path.join(db_path, 'text_embeddings.npy')).astype('float32')
        text_index = faiss.IndexFlatIP(text_embeddings.shape[1]); faiss.normalize_L2(text_embeddings); text_index.add(text_embeddings)
        image_df = pd.read_csv(os.path.join(db_path, 'image_data.csv'))
        image_embeddings = np.load(os.path.join(db_path, 'image_embeddings.npy')).astype('float32')
        image_index = faiss.IndexFlatIP(image_embeddings.shape[1]); faiss.normalize_L2(image_embeddings); image_index.add(image_embeddings)
        return {"text": {"df": text_df, "index": text_index}, "image": {"df": image_df, "index": image_index}}
    except FileNotFoundError:
        return None

@st.cache_data
def render_pdf_page(page_number, text_to_highlight=None):
    doc = fitz.open(PDF_PATH)
    page = doc.load_page(page_number - 1)
    if text_to_highlight:
        for inst in page.search_for(text_to_highlight): page.add_highlight_annot(inst)
    pix = page.get_pixmap(dpi=200)
    img_bytes = pix.tobytes("png")
    doc.close()
    return Image.open(io.BytesIO(img_bytes))

# --- FUNÇÃO PARA SALVAR AVALIAÇÕES ---
def save_evaluation(eval_data):
    with file_lock:
        evaluations = []
        if os.path.exists(EVALUATION_FILE):
            with open(EVALUATION_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    try: evaluations.append(json.loads(line))
                    except json.JSONDecodeError: pass # Ignora linhas mal formatadas ou vazias
        
        evaluations.append(eval_data)
        
        with open(EVALUATION_FILE, 'w', encoding='utf-8') as f:
            for entry in evaluations:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    st.toast(f"Avaliação '{eval_data['evaluation']}' salva!", icon="✅")

# --- FUNÇÕES DE CALLBACK E LÓGICA ---
def set_page_view(page_num, text=None):
    st.session_state.current_page = int(page_num)
    st.session_state.text_to_highlight = text

def perform_search():
    query = st.session_state.query_input
    if not query:
        st.warning("Por favor, digite um termo de busca.")
        return

    st.session_state.search_performed = True
    approach_config = APPROACHES[st.session_state.approach_selector]
    
    text_model_data = get_model(approach_config.get("text_model_id", approach_config.get("model_id")))
    image_model_data = get_model(approach_config.get("image_model_id", approach_config.get("model_id")))
    engines = load_database_for_approach(approach_config['id'])

    # Vetoriza para busca de texto
    if text_model_data["type"] == "sentence_transformer":
        text_query_vector = text_model_data["model"].encode(["query: " + query]).astype('float32')
    else: # CLIP
        with torch.no_grad():
            inputs = text_model_data["processor"](text=query, return_tensors="pt").to(text_model_data["device"])
            text_query_vector = text_model_data["model"].get_text_features(**inputs).cpu().numpy().astype('float32')
    faiss.normalize_L2(text_query_vector)

    # Vetoriza para busca de imagem
    if image_model_data["type"] == "clip":
        with torch.no_grad():
            inputs = image_model_data["processor"](text=query, return_tensors="pt").to(image_model_data["device"])
            image_query_vector = image_model_data["model"].get_text_features(**inputs).cpu().numpy().astype('float32')
    else: # SentenceTransformer
        image_query_vector = image_model_data["model"].encode(["query: " + query]).astype('float32')
    faiss.normalize_L2(image_query_vector)

    # Realiza as buscas
    text_eng, img_eng = engines["text"], engines["image"]
    text_distances, text_indices = text_eng["index"].search(text_query_vector, 5)
    img_distances, img_indices = img_eng["index"].search(image_query_vector, 5)

    st.session_state.text_results = text_eng["df"].iloc[text_indices[0]] if len(text_indices[0]) > 0 else pd.DataFrame()
    st.session_state.text_scores = text_distances[0] if len(text_indices[0]) > 0 else np.array([])
    st.session_state.image_results = img_eng["df"].iloc[img_indices[0]] if len(img_indices[0]) > 0 else pd.DataFrame()
    st.session_state.image_scores = img_distances[0] if len(img_indices[0]) > 0 else np.array([])

    # Lógica de Auto-Jump
    top_image_score = st.session_state.image_scores[0] if len(st.session_state.image_scores) > 0 else -1
    top_text_score = st.session_state.text_scores[0] if len(st.session_state.text_scores) > 0 else -1
    
    if top_image_score > top_text_score:
        set_page_view(st.session_state.image_results.iloc[0]['page'])
    elif top_text_score != -1:
        top_result = st.session_state.text_results.iloc[0]
        set_page_view(top_result['page'], ' '.join(top_result['original_text'].split()[:40]))

# --- SETUP INICIAL DA SESSÃO ---
if 'current_page' not in st.session_state: st.session_state.current_page = 1
if 'text_to_highlight' not in st.session_state: st.session_state.text_to_highlight = None
if 'search_performed' not in st.session_state: st.session_state.search_performed = False

# --- INTERFACE GRÁFICA (UI) ---
st.set_page_config(layout="wide")
st.title("🔬 Contextus Bio: Painel de Busca e Avaliação")

# Barra Lateral
with st.sidebar:
    st.header("Configuração")
    evaluator_name = st.text_input("Seu nome (Avaliador):", key="evaluator_name")
    selected_approach_name = st.radio("Escolha a abordagem de busca:", list(APPROACHES.keys()), key="approach_selector")
    approach_config = APPROACHES[selected_approach_name]
    st.info(f"**ID da Abordagem:** `{approach_config['id']}`")

# Carregamento dos dados
engines = load_database_for_approach(approach_config['id'])

# Layout principal
col1, col2 = st.columns([1, 2])

if not engines:
    st.error(f"Base de dados para a abordagem '{selected_approach_name}' não encontrada.")
else:
    with col1:
        st.header("Busca e Avaliação")
        st.text_input("Busque por um conceito:", key="query_input", on_change=perform_search)
        st.button("Buscar", on_click=perform_search)
        
        if st.session_state.get('search_performed', False):
            st.divider()
            st.subheader(f"Resultados para: '{st.session_state.query_input}'")
            
            for result_type in ["image", "text"]:
                st.write(f"**{'🖼️ Imagens' if result_type == 'image' else '📝 Textos'} Relevantes:**")
                results_key, scores_key = f"{result_type}_results", f"{result_type}_scores"
                
                if results_key in st.session_state and not st.session_state[results_key].empty:
                    for i, (index, row) in enumerate(st.session_state[results_key].iterrows()):
                        with st.container(border=True):
                            score = st.session_state[scores_key][i]
                            st.caption(f"Página {int(row['page'])} | Score: {score:.4f} | ID: {row['id']}")
                            
                            if result_type == "image":
                                st.image(Image.open(row['content_path']), use_container_width=True)
                            st.info(row['text_for_embedding'])
                            
                            # Botões de avaliação
                            btn_cols = st.columns(2)
                            eval_data = {"timestamp": datetime.now().isoformat(), "evaluator": evaluator_name, "approach_id": approach_config['id'], "query": st.session_state.query_input, "result_id": row['id'], "result_type": result_type, "result_page": int(row['page']), "result_score": float(score)}
                            
                            if btn_cols[0].button("👍 Relevante", key=f"like_{row['id']}", use_container_width=True):
                                if evaluator_name:
                                    eval_data["evaluation"] = "relevante"
                                    save_evaluation(eval_data)
                                else:
                                    st.warning("Por favor, insira seu nome de avaliador na barra lateral.")
                            
                            if btn_cols[1].button("👎 Não Relevante", key=f"dislike_{row['id']}", use_container_width=True):
                                if evaluator_name:
                                    eval_data["evaluation"] = "nao_relevante"
                                    save_evaluation(eval_data)
                                else:
                                    st.warning("Por favor, insira seu nome de avaliador na barra lateral.")
    with col2:
        st.header(f"Visualizador - Página {st.session_state.current_page}")
        current_highlight = st.session_state.text_to_highlight if st.session_state.get('search_performed', False) else None
        try:
            page_image = render_pdf_page(page_number=st.session_state.current_page, text_to_highlight=current_highlight)
            st.image(page_image, use_container_width=True)
        except Exception as e:
            st.error(f"Não foi possível renderizar a página {st.session_state.current_page}. Erro: {e}")