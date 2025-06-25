import pandas as pd
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import os

# --- CONFIGURAÇÕES DE CAMINHOS ---
SOURCE_ANNOTATIONS_FILE = 'data/annotations.csv'
CLEAN_ANNOTATIONS_FILE = 'data/annotations_clean.csv'
OUTPUT_DB_DIR = 'database/approach_3_pure_clip'

# --- FUNÇÃO DE LIMPEZA E PREPARAÇÃO (Idêntica às anteriores) ---
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

# --- FUNÇÃO ÚNICA PARA GERAR TODOS OS EMBEDDINGS COM CLIP ---
def generate_all_clip_embeddings(df):
    print("\n--- Gerando Todos os Embeddings (com CLIP) ---")
    MODEL_ID = "openai/clip-vit-large-patch14"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    print(f"Carregando modelo CLIP: {MODEL_ID}...")
    model = CLIPModel.from_pretrained(MODEL_ID).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)

    # --- Processa Textos ---
    TEXT_EMBEDDINGS_FILE = os.path.join(OUTPUT_DB_DIR, 'text_embeddings.npy')
    TEXT_DATA_FILE = os.path.join(OUTPUT_DB_DIR, 'text_data.csv')
    text_df = df[df['type'] == 'text'].copy()
    
    if not text_df.empty:
        print("Vetorizando textos com o codificador de texto do CLIP...")
        text_inputs = processor(text=text_df['text_for_embedding'].tolist(), return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            text_embeddings = model.get_text_features(**text_inputs).cpu().numpy()
        
        np.save(TEXT_EMBEDDINGS_FILE, text_embeddings)
        text_df.to_csv(TEXT_DATA_FILE, index=False)
        print("Embeddings de texto (CLIP) finalizados.")
    else:
        print("Nenhum item de texto encontrado. Pulando.")

    # --- Processa Imagens ---
    IMAGE_EMBEDDINGS_FILE = os.path.join(OUTPUT_DB_DIR, 'image_embeddings.npy')
    IMAGE_DATA_FILE = os.path.join(OUTPUT_DB_DIR, 'image_data.csv')
    image_df = df[df['type'] == 'image'].copy()

    if not image_df.empty:
        print("Vetorizando imagens (pixels puros) com o codificador de imagem do CLIP...")
        image_embeddings = []
        for index, row in tqdm(image_df.iterrows(), total=image_df.shape[0]):
            try:
                image = Image.open(row['content_path'])
                inputs = processor(images=image, return_tensors="pt").to(device)
                with torch.no_grad():
                    embedding = model.get_image_features(**inputs).cpu().numpy()
                image_embeddings.append(embedding.squeeze())
            except Exception as e:
                print(f"Erro ao processar imagem {row.get('id', 'ID desconhecido')}: {e}")
                image_embeddings.append(np.zeros(768)) # Dimensão do CLIP Large é 768
        
        np.save(IMAGE_EMBEDDINGS_FILE, np.array(image_embeddings))
        image_df.to_csv(IMAGE_DATA_FILE, index=False)
        print("Embeddings de imagem (CLIP) finalizados.")
    else:
        print("Nenhuma imagem encontrada. Pulando.")


# --- SCRIPT PRINCIPAL ---
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DB_DIR):
        os.makedirs(OUTPUT_DB_DIR)
        
    if not os.path.exists(CLEAN_ANNOTATIONS_FILE):
        cleaned_df = clean_and_prepare_data(SOURCE_ANNOTATIONS_FILE, CLEAN_ANNOTATIONS_FILE)
    else:
        print(f"Usando arquivo limpo existente: '{CLEAN_ANNOTATIONS_FILE}'")
        cleaned_df = pd.read_csv(CLEAN_ANNOTATIONS_FILE)

    generate_all_clip_embeddings(cleaned_df)
    
    print(f"\nConstrução da base de dados para a Abordagem 3 concluída!")
    print(f"Arquivos salvos em: '{OUTPUT_DB_DIR}'")