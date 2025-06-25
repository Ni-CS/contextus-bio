import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
import faiss

# --- CONFIGURAÇÕES DE CAMINHOS PARA A NOVA ESTRUTURA ---
SOURCE_ANNOTATIONS_FILE = 'data/annotations.csv'
CLEAN_ANNOTATIONS_FILE = 'data/annotations_clean.csv' # Salvaremos uma cópia limpa na pasta de dados
OUTPUT_DB_DIR = 'database/approach_1_text_only' # Pasta de saída para a nossa melhor abordagem

# --- FUNÇÃO DE LIMPEZA E PREPARAÇÃO ---
def clean_and_prepare_data(source_file, clean_file):
    print(f"Lendo e preparando os dados de '{source_file}'...")
    if not os.path.exists(source_file):
        raise FileNotFoundError(f"Arquivo de anotações fonte '{source_file}' não encontrado!")
        
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
    print(f"Arquivo limpo '{clean_file}' salvo com sucesso. Total de {len(df)} itens.")
    return df

# --- FUNÇÃO PARA GERAR EMBEDDINGS DE TEXTO ---
def generate_text_embeddings(df):
    print("\n--- Iniciando Geração de Embeddings de Texto ---")
    MODEL_ID = 'intfloat/multilingual-e5-large'
    TEXT_EMBEDDINGS_FILE = os.path.join(OUTPUT_DB_DIR, 'text_embeddings.npy')
    TEXT_DATA_FILE = os.path.join(OUTPUT_DB_DIR, 'text_data.csv')

    text_df = df[df['type'] == 'text'].copy()
    if text_df.empty:
        print("Nenhum item de texto encontrado. Pulando.")
        return

    print(f"Carregando o modelo de texto: {MODEL_ID}...")
    model = SentenceTransformer(MODEL_ID)
    texts_to_embed = ("passage: " + text_df['text_for_embedding']).tolist()

    print("Vetorizando textos...")
    text_embeddings = model.encode(texts_to_embed, show_progress_bar=True)

    np.save(TEXT_EMBEDDINGS_FILE, text_embeddings)
    text_df.to_csv(TEXT_DATA_FILE, index=False)
    print("Embeddings de texto finalizados.")

# --- FUNÇÃO PARA GERAR EMBEDDINGS DE DESCRIÇÕES DE IMAGEM ---
def generate_image_embeddings(df):
    print("\n--- Iniciando Geração de Embeddings de Descrições de Imagem ---")
    MODEL_ID = 'intfloat/multilingual-e5-large'
    IMAGE_EMBEDDINGS_FILE = os.path.join(OUTPUT_DB_DIR, 'image_embeddings.npy')
    IMAGE_DATA_FILE = os.path.join(OUTPUT_DB_DIR, 'image_data.csv')

    image_df = df[df['type'] == 'image'].copy()
    if image_df.empty:
        print("Nenhuma imagem encontrada. Pulando.")
        return

    print(f"Carregando o modelo de texto (para descrições): {MODEL_ID}...")
    model = SentenceTransformer(MODEL_ID)
    texts_to_embed = ("passage: " + image_df['text_for_embedding']).tolist()

    print("Vetorizando descrições de imagens...")
    image_embeddings = model.encode(texts_to_embed, show_progress_bar=True)

    np.save(IMAGE_EMBEDDINGS_FILE, image_embeddings)
    image_df.to_csv(IMAGE_DATA_FILE, index=False)
    print("Embeddings de imagem finalizados.")

# --- SCRIPT PRINCIPAL ---
if __name__ == "__main__":
    # Garante que o diretório de saída exista
    if not os.path.exists(OUTPUT_DB_DIR):
        os.makedirs(OUTPUT_DB_DIR)
        
    cleaned_df = clean_and_prepare_data(SOURCE_ANNOTATIONS_FILE, CLEAN_ANNOTATIONS_FILE)
    generate_text_embeddings(cleaned_df)
    generate_image_embeddings(cleaned_df)
    
    print(f"\nConstrução da base de dados concluída com sucesso! Arquivos salvos em '{OUTPUT_DB_DIR}'")