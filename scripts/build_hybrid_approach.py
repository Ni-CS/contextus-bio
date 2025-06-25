import pandas as pd
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
import faiss

# --- CONFIGURAÇÕES DE CAMINHOS ---
SOURCE_ANNOTATIONS_FILE = 'data/annotations.csv'
CLEAN_ANNOTATIONS_FILE = 'data/annotations_clean.csv'
OUTPUT_DB_DIR = 'database/approach_2_hybrid_clip'

# --- FUNÇÃO DE LIMPEZA E PREPARAÇÃO ---
def clean_and_prepare_data(source_file, clean_file):
    print(f"Lendo e preparando os dados de '{source_file}'...")
    df = pd.read_csv(source_file)
    df.dropna(subset=['description_llm', 'keywords_llm'], inplace=True)
    df['keywords_llm'] = df['keywords_llm'].astype(str)
    
    def create_embedding_text(row):
        description = row['description_llm']
        keywords = row['keywords_llm']
        if row['type'] == 'text':
            return f"Conteúdo sobre: {description}. Termos relacionados: {keywords}"
        elif row['type'] == 'image':
            return f"Imagem mostrando: {description}. Termos relacionados: {keywords}"
        return ""

    df['text_for_embedding'] = df.apply(create_embedding_text, axis=1)
    df.to_csv(clean_file, index=False)
    print(f"Arquivo limpo '{clean_file}' salvo com sucesso.")
    return df

# --- FUNÇÃO PARA GERAR EMBEDDINGS DE TEXTO (com e5-large) ---
def generate_text_embeddings(df):
    print("\n--- Gerando Embeddings de Texto (com e5-large) ---")
    MODEL_ID = 'intfloat/multilingual-e5-large'
    TEXT_EMBEDDINGS_FILE = os.path.join(OUTPUT_DB_DIR, 'text_embeddings.npy')
    TEXT_DATA_FILE = os.path.join(OUTPUT_DB_DIR, 'text_data.csv')

    text_df = df[df['type'] == 'text'].copy()
    if text_df.empty:
        print("Nenhum item de texto encontrado. Pulando.")
        return
        
    print(f"Carregando modelo de texto: {MODEL_ID}...")
    model = SentenceTransformer(MODEL_ID)
    texts_to_embed = ("passage: " + text_df['text_for_embedding']).tolist()
    
    print("Vetorizando textos...")
    text_embeddings = model.encode(texts_to_embed, show_progress_bar=True)
    
    np.save(TEXT_EMBEDDINGS_FILE, text_embeddings)
    text_df.to_csv(TEXT_DATA_FILE, index=False)
    print("Embeddings de texto finalizados.")

# --- FUNÇÃO PARA GERAR EMBEDDINGS HÍBRIDOS DE IMAGEM (com CLIP) ---
def generate_hybrid_image_embeddings(df):
    print("\n--- Gerando Embeddings de Imagem (HÍBRIDOS com CLIP) ---")
    MODEL_ID = "openai/clip-vit-large-patch14"
    IMAGE_EMBEDDINGS_FILE = os.path.join(OUTPUT_DB_DIR, 'image_embeddings.npy')
    IMAGE_DATA_FILE = os.path.join(OUTPUT_DB_DIR, 'image_data.csv')
    
    image_df = df[df['type'] == 'image'].copy()
    if image_df.empty:
        print("Nenhuma imagem encontrada. Pulando.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")
    
    print(f"Carregando modelo CLIP: {MODEL_ID}...")
    model = CLIPModel.from_pretrained(MODEL_ID).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)

    hybrid_embeddings = []
    print("Gerando embeddings híbridos (visual + textual)...")
    for index, row in tqdm(image_df.iterrows(), total=image_df.shape[0]):
        try:
            # Vetor da Imagem (Visual)
            image_vec_tensor = model.get_image_features(**processor(images=Image.open(row['content_path']), return_tensors="pt").to(device))
            
            # Vetor da Descrição (Textual)
            text_vec_tensor = model.get_text_features(**processor(text=row['text_for_embedding'], return_tensors="pt", padding=True, truncation=True).to(device))
            
            # Converte para NumPy e normaliza
            image_vec = image_vec_tensor.cpu().detach().numpy().astype('float32')
            text_vec = text_vec_tensor.cpu().detach().numpy().astype('float32')
            faiss.normalize_L2(image_vec)
            faiss.normalize_L2(text_vec)
            
            # Combina com pesos (40% visual, 60% textual)
            hybrid_vec = (0.4 * image_vec) + (0.6 * text_vec)
            faiss.normalize_L2(hybrid_vec) # Re-normaliza o vetor final
            
            hybrid_embeddings.append(hybrid_vec.squeeze())
        except Exception as e:
            print(f"Erro ao processar imagem {row.get('id', 'ID desconhecido')}: {e}")
            hybrid_embeddings.append(np.zeros(768)) # Dimensão do CLIP Large é 768

    np.save(IMAGE_EMBEDDINGS_FILE, np.array(hybrid_embeddings))
    image_df.to_csv(IMAGE_DATA_FILE, index=False)
    print("Embeddings híbridos de imagem finalizados.")


# --- SCRIPT PRINCIPAL ---
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DB_DIR):
        os.makedirs(OUTPUT_DB_DIR)
        
    # Reutiliza o arquivo limpo se ele já existir, senão cria um novo
    if not os.path.exists(CLEAN_ANNOTATIONS_FILE):
        cleaned_df = clean_and_prepare_data(SOURCE_ANNOTATIONS_FILE, CLEAN_ANNOTATIONS_FILE)
    else:
        print(f"Usando arquivo limpo existente: '{CLEAN_ANNOTATIONS_FILE}'")
        cleaned_df = pd.read_csv(CLEAN_ANNOTATIONS_FILE)

    generate_text_embeddings(cleaned_df)
    generate_hybrid_image_embeddings(cleaned_df)
    
    print(f"\nConstrução da base de dados para a Abordagem 2 concluída!")
    print(f"Arquivos salvos em: '{OUTPUT_DB_DIR}'")