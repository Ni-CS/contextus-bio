import pandas as pd
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import os
import faiss

# --- CONFIGURAÇÕES ---
SOURCE_ANNOTATIONS_FILE = 'data/annotations.csv'
CLEAN_ANNOTATIONS_FILE = 'data/annotations_clean.csv'
OUTPUT_DB_DIR = 'database/approach_4_unified_hybrid' # Nova pasta de saída
MODEL_ID = "openai/clip-vit-large-patch14"

# --- FUNÇÃO DE LIMPEZA (Idêntica às anteriores) ---
def clean_and_prepare_data(source_file, clean_file):
    print(f"Lendo e preparando dados de '{source_file}'...")
    df = pd.read_csv(source_file)
    df.dropna(subset=['description_llm', 'keywords_llm'], inplace=True)
    df['keywords_llm'] = df['keywords_llm'].astype(str)
    def create_embedding_text(row):
        description, keywords = row['description_llm'], row['keywords_llm']
        return f"Descrição: {description}. Palavras-chave: {keywords}"
    df['text_for_embedding'] = df.apply(create_embedding_text, axis=1)
    df.to_csv(clean_file, index=False)
    print(f"Arquivo limpo salvo em '{clean_file}'.")
    return df

# --- FUNÇÃO PRINCIPAL DE GERAÇÃO DE EMBEDDINGS ---
def generate_unified_clip_embeddings(df):
    print(f"\n--- Gerando todos os embeddings com o modelo CLIP: {MODEL_ID} ---")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    model = CLIPModel.from_pretrained(MODEL_ID).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)

    # --- Vetoriza Textos ---
    TEXT_EMBEDDINGS_FILE = os.path.join(OUTPUT_DB_DIR, 'text_embeddings.npy')
    TEXT_DATA_FILE = os.path.join(OUTPUT_DB_DIR, 'text_data.csv')
    text_df = df[df['type'] == 'text'].copy()
    
    if not text_df.empty:
        print("Vetorizando textos com CLIP...")
        inputs = processor(text=text_df['text_for_embedding'].tolist(), return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            text_embeddings = model.get_text_features(**inputs).cpu().numpy()
        np.save(TEXT_EMBEDDINGS_FILE, text_embeddings)
        text_df.to_csv(TEXT_DATA_FILE, index=False)
        print("Embeddings de texto (CLIP) finalizados.")

    # --- Vetoriza Imagens (Método Híbrido) ---
    IMAGE_EMBEDDINGS_FILE = os.path.join(OUTPUT_DB_DIR, 'image_embeddings.npy')
    IMAGE_DATA_FILE = os.path.join(OUTPUT_DB_DIR, 'image_data.csv')
    image_df = df[df['type'] == 'image'].copy()

    if not image_df.empty:
        print("Gerando embeddings HÍBRIDOS de imagem com CLIP...")
        hybrid_embeddings = []
        for _, row in tqdm(image_df.iterrows(), total=image_df.shape[0]):
            try:
                # Gera vetor visual
                image_vec_tensor = model.get_image_features(**processor(images=Image.open(row['content_path']), return_tensors="pt").to(device))
                # Gera vetor textual da descrição
                text_vec_tensor = model.get_text_features(**processor(text=row['text_for_embedding'], return_tensors="pt", padding=True, truncation=True).to(device))
                
                image_vec = image_vec_tensor.cpu().detach().numpy().astype('float32')
                text_vec = text_vec_tensor.cpu().detach().numpy().astype('float32')
                faiss.normalize_L2(image_vec)
                faiss.normalize_L2(text_vec)
                
                # Combina com pesos (40% visual, 60% textual)
                hybrid_vec = (0.4 * image_vec) + (0.6 * text_vec)
                faiss.normalize_L2(hybrid_vec)
                
                hybrid_embeddings.append(hybrid_vec.squeeze())
            except Exception as e:
                print(f"Erro ao processar imagem {row.get('id', 'ID desconhecido')}: {e}")
                hybrid_embeddings.append(np.zeros(768))

        np.save(IMAGE_EMBEDDINGS_FILE, np.array(hybrid_embeddings))
        image_df.to_csv(IMAGE_DATA_FILE, index=False)
        print("Embeddings híbridos de imagem (CLIP) finalizados.")

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DB_DIR): os.makedirs(OUTPUT_DB_DIR)
    
    cleaned_df = clean_and_prepare_data(SOURCE_ANNOTATIONS_FILE, CLEAN_ANNOTATIONS_FILE)
    generate_unified_clip_embeddings(cleaned_df)
    
    print(f"\nConstrução da base de dados para a Abordagem Híbrida Unificada concluída!")