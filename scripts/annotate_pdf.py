import google.generativeai as genai
import fitz  # PyMuPDF
import os
import pandas as pd
from PIL import Image
import io
import time # Importar a biblioteca time
from dotenv import load_dotenv

# Carrega a API Key do arquivo .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("API Key do Google não encontrada. Verifique seu arquivo .env")

genai.configure(api_key=GOOGLE_API_KEY)

# Cria o diretório de saída se não existirimport google.generativeai as genai
import fitz  # PyMuPDF
import os
import pandas as pd
from PIL import Image
import io
import time
import re
import numpy as np
from dotenv import load_dotenv

# Carrega a API Key do arquivo .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("API Key do Google não encontrada. Verifique seu arquivo .env")

genai.configure(api_key=GOOGLE_API_KEY)

# --- CONFIGURAÇÃO E FILTROS ---
PDF_PATH = "data/Biologia-Celular.pdf"
OUTPUT_DIR = "data"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "annotations.csv")

# Parâmetros de filtro
MIN_IMAGE_WIDTH = 100  # Ignorar imagens com largura menor que 100 pixels
MIN_IMAGE_HEIGHT = 100 # Ignorar imagens com altura menor que 100 pixels
IMAGE_VARIANCE_THRESHOLD = 20 # Limiar para detectar imagens em branco/uniformes
TEXT_STOP_KEYWORDS = [
    'copyright', 'isbn', 'cdu', 'universidade federal', 'ministério da educação',
    'reitor', 'coordenação', 'diagramação', 'ilustrações', 'sumário', 'bibliográficas'
]
TOC_REGEX_PATTERN = r'(\.|\s){10,}\s*\d+' # Padrão para detectar linhas de sumário (ex: "..... 10")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- MODELO E PROMPTS (sem alteração) ---
model = genai.GenerativeModel('gemini-1.5-flash-latest')
# ... (os prompts PROMPT_TEXT e PROMPT_IMAGE continuam os mesmos) ...
PROMPT_TEXT = """
Você é um assistente especialista em biologia celular.
Analise o seguinte bloco de texto extraído de uma apostila de biologia.
1.  Faça um resumo conciso do conteúdo principal (2-3 sentenças).
2.  Gere uma lista de 5 a 10 palavras-chave ou termos de busca relevantes.
Responda em formato:
RESUMO: [Seu resumo aqui]
PALAVRAS-CHAVE: [palavra1, palavra2, ...]

TEXTO:
---
{}
---
"""
PROMPT_IMAGE = """
Você é um assistente especialista em biologia celular.
Analise esta imagem extraída de uma apostila de biologia.
1.  Descreva detalhadamente o que a imagem representa.
2.  Gere uma lista de 5 a 10 palavras-chave ou termos de busca relevantes.
Responda em formato:
DESCRIÇÃO: [Sua descrição aqui]
PALAVRAS-CHAVE: [palavra1, palavra2, ...]
"""

# --- NOVAS FUNÇÕES DE FILTRO ---
def is_image_relevant(pil_image):
    """Verifica se uma imagem é relevante para anotação, com depuração."""
    # Filtro por dimensões mínimas
    if pil_image.width < MIN_IMAGE_WIDTH or pil_image.height < MIN_IMAGE_HEIGHT:
        print(f"  -> Imagem ignorada (dimensões pequenas: {pil_image.width}x{pil_image.height})")
        return False
        
    # Filtro por variância de cor
    gray_image = pil_image.convert('L')
    np_image = np.array(gray_image)
    std_dev = np_image.std()
    
    if std_dev < IMAGE_VARIANCE_THRESHOLD:
        # Mensagem de depuração para entendermos por que foi ignorada
        print(f"  -> Imagem ignorada (desvio padrão de cor baixo: {std_dev:.2f})")
        return False
        
    return True

def is_text_relevant(text):
    """Verifica se um bloco de texto é relevante para anotação."""
    lower_text = text.lower()
    # Filtro por palavras-chave de metadados
    if any(keyword in lower_text for keyword in TEXT_STOP_KEYWORDS):
        return False
    # Filtro por padrão de sumário (regex)
    if re.search(TOC_REGEX_PATTERN, lower_text):
        return False
    # Filtro de textos muito curtos (já existente)
    if len(lower_text) < 50:
        return False
    return True

def parse_llm_response(response_text):
    """Extrai o resumo/descrição e as palavras-chave da resposta do LLM."""
    summary = ""
    keywords = ""
    try:
        summary_part = response_text.split("RESUMO:")[1].split("PALAVRAS-CHAVE:")[0]
        summary = summary_part.strip()
    except IndexError:
        try:
            summary_part = response_text.split("DESCRIÇÃO:")[1].split("PALAVRAS-CHAVE:")[0]
            summary = summary_part.strip()
        except IndexError:
            summary = "N/A"

    try:
        keywords_part = response_text.split("PALAVRAS-CHAVE:")[1]
        keywords = keywords_part.strip()
    except IndexError:
        keywords = "N/A"

    return summary, keywords

def save_annotation(annotation_data, csv_path):
    """Salva uma única anotação no arquivo CSV em modo append."""
    df_new = pd.DataFrame([annotation_data])
    df_new.to_csv(csv_path, mode='a', header=False, index=False, encoding='utf-8-sig')


# --- FUNÇÃO PRINCIPAL ATUALIZADA ---
def annotate_pdf():
    processed_ids = set()
    csv_headers = ["id", "page", "type", "content_path", "original_text", "description_llm", "keywords_llm"]
    if os.path.exists(OUTPUT_CSV):
        print(f"Arquivo de anotações existente encontrado: {OUTPUT_CSV}. Lendo IDs...")
        df_existing = pd.read_csv(OUTPUT_CSV)
        if 'id' in df_existing.columns:
            processed_ids = set(df_existing['id'].astype(str))
        print(f"{len(processed_ids)} itens já processados. O trabalho será retomado.")
    else:
        print("Criando novo arquivo de anotações.")
        pd.DataFrame(columns=csv_headers).to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    doc = fitz.open(PDF_PATH)
    print(f"Iniciando o processamento do PDF: {PDF_PATH}")

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_number = page_num + 1
        print(f"\n--- Processando Página {page_number}/{len(doc)} ---")

        # 1. Extrair e anotar imagens relevantes
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            item_id = f"p{page_number}_img_{img_index}"
            if item_id in processed_ids:
                continue

            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            pil_image = Image.open(io.BytesIO(image_bytes))

            # *** NOVO: VERIFICAÇÃO DE RELEVÂNCIA DA IMAGEM ***
            if not is_image_relevant(pil_image):
                print(f"  Imagem {item_id} ignorada (irrelevante).")
                continue

            print(f"  Processando imagem relevante: {item_id}")
            img_path = os.path.join(OUTPUT_DIR, f"/images/{item_id}.png")
            pil_image.save(img_path)

            try:
                response = model.generate_content([PROMPT_IMAGE, pil_image])
                description, keywords = parse_llm_response(response.text)
                annotation = {"id": item_id, "page": page_number, "type": "image", "content_path": img_path, "original_text": "", "description_llm": description, "keywords_llm": keywords}
                save_annotation(annotation, OUTPUT_CSV)
                processed_ids.add(item_id)
                print(f"  Anotação de {item_id} salva com sucesso.")
                time.sleep(4.1)
            except Exception as e:
                print(f"  ERRO ao anotar a imagem {item_id}: {e}")

        # 2. Extrair e anotar textos relevantes
        text_blocks = page.get_text("blocks")
        for block_index, block in enumerate(text_blocks):
            text = block[4].strip().replace("\n", " ")
            item_id = f"p{page_number}_txt_{block_index}"
            if item_id in processed_ids:
                continue
            
            # *** NOVO: VERIFICAÇÃO DE RELEVÂNCIA DO TEXTO ***
            if not is_text_relevant(text):
                print(f"  Texto {item_id} ignorado (irrelevante).")
                continue

            print(f"  Processando texto relevante: {item_id}")
            try:
                response = model.generate_content(PROMPT_TEXT.format(text))
                summary, keywords = parse_llm_response(response.text)
                annotation = {"id": item_id, "page": page_number, "type": "text", "content_path": "", "original_text": text, "description_llm": summary, "keywords_llm": keywords}
                save_annotation(annotation, OUTPUT_CSV)
                processed_ids.add(item_id)
                print(f"  Anotação de {item_id} salva com sucesso.")
                time.sleep(4.1)
            except Exception as e:
                print(f"  ERRO ao anotar bloco de texto {item_id}: {e}")

    print(f"\nProcessamento concluído! Anotações salvas em: {OUTPUT_CSV}")

if __name__ == "__main__":
    annotate_pdf()
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- MODELO E PROMPTS ---
# Usando o modelo recomendado: Gemini 1.5 Flash
model = genai.GenerativeModel('gemini-1.5-flash-latest')

PROMPT_TEXT = """
Você é um assistente especialista em biologia celular.
Analise o seguinte bloco de texto extraído de uma apostila de biologia.
1.  Faça um resumo conciso do conteúdo principal (2-3 sentenças).
2.  Gere uma lista de 5 a 10 palavras-chave ou termos de busca relevantes.
Responda em formato:
RESUMO: [Seu resumo aqui]
PALAVRAS-CHAVE: [palavra1, palavra2, ...]

TEXTO:
---
{}
---
"""

PROMPT_IMAGE = """
Você é um assistente especialista em biologia celular.
Analise esta imagem extraída de uma apostila de biologia.
1.  Descreva detalhadamente o que a imagem representa.
2.  Gere uma lista de 5 a 10 palavras-chave ou termos de busca relevantes.
Responda em formato:
DESCRIÇÃO: [Sua descrição aqui]
PALAVRAS-CHAVE: [palavra1, palavra2, ...]
"""

def parse_llm_response(response_text):
    """Extrai o resumo/descrição e as palavras-chave da resposta do LLM."""
    summary = ""
    keywords = ""
    try:
        summary_part = response_text.split("RESUMO:")[1].split("PALAVRAS-CHAVE:")[0]
        summary = summary_part.strip()
    except IndexError:
        try:
            summary_part = response_text.split("DESCRIÇÃO:")[1].split("PALAVRAS-CHAVE:")[0]
            summary = summary_part.strip()
        except IndexError:
            summary = "N/A"

    try:
        keywords_part = response_text.split("PALAVRAS-CHAVE:")[1]
        keywords = keywords_part.strip()
    except IndexError:
        keywords = "N/A"

    return summary, keywords

def save_annotation(annotation_data, csv_path):
    """Salva uma única anotação no arquivo CSV em modo append."""
    df_new = pd.DataFrame([annotation_data])
    # Escreve no arquivo: 'a' para append, header=False para não repetir o cabeçalho
    df_new.to_csv(csv_path, mode='a', header=False, index=False, encoding='utf-8-sig')

def annotate_pdf():
    """Função principal para extrair, anotar e salvar o conteúdo do PDF de forma resiliente."""
    
    # --- LÓGICA DE RETOMADA ---
    processed_ids = set()
    csv_headers = ["id", "page", "type", "content_path", "original_text", "description_llm", "keywords_llm"]
    
    if os.path.exists(OUTPUT_CSV):
        print(f"Arquivo de anotações existente encontrado em {OUTPUT_CSV}. Lendo IDs já processados...")
        df_existing = pd.read_csv(OUTPUT_CSV)
        # Garante que a coluna 'id' exista antes de tentar acessá-la
        if 'id' in df_existing.columns:
            processed_ids = set(df_existing['id'].astype(str))
        print(f"{len(processed_ids)} itens já foram processados. O trabalho será retomado.")
    else:
        print("Nenhum arquivo de anotações encontrado. Criando um novo.")
        # Cria o arquivo com o cabeçalho
        pd.DataFrame(columns=csv_headers).to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    doc = fitz.open(PDF_PATH)
    print(f"Iniciando o processamento do PDF: {PDF_PATH}")

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_number = page_num + 1
        print(f"\n--- Processando Página {page_number}/{len(doc)} ---")

        # 1. Extrair e anotar imagens
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            item_id = f"p{page_number}_img_{img_index}"
            if item_id in processed_ids:
                print(f"  Item {item_id} já processado. Pulando.")
                continue

            print(f"  Processando imagem: {item_id}")
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            pil_image = Image.open(io.BytesIO(image_bytes))
            img_path = os.path.join(OUTPUT_DIR, f"{item_id}.png")
            pil_image.save(img_path)
            
            try:
                response = model.generate_content([PROMPT_IMAGE, pil_image])
                description, keywords = parse_llm_response(response.text)
                
                annotation = {
                    "id": item_id, "page": page_number, "type": "image", 
                    "content_path": img_path, "original_text": "",
                    "description_llm": description, "keywords_llm": keywords
                }
                save_annotation(annotation, OUTPUT_CSV)
                processed_ids.add(item_id) 
                print(f"  Anotação de {item_id} salva com sucesso.")
                time.sleep(4.1) # <-- ALTERE AQUI
            except Exception as e:
                print(f"  ERRO ao anotar a imagem {item_id}: {e}")

        # 2. Extrair e anotar blocos de texto
        text_blocks = page.get_text("blocks")
        for block_index, block in enumerate(text_blocks):
            text = block[4].strip().replace("\n", " ")
            if len(text) > 50:
                item_id = f"p{page_number}_txt_{block_index}"
                if item_id in processed_ids:
                    print(f"  Item {item_id} já processado. Pulando.")
                    continue

                print(f"  Processando texto: {item_id}")
                try:
                    response = model.generate_content(PROMPT_TEXT.format(text))
                    summary, keywords = parse_llm_response(response.text)
                    
                    annotation = {
                        "id": item_id, "page": page_number, "type": "text",
                        "content_path": "", "original_text": text,
                        "description_llm": summary, "keywords_llm": keywords
                    }
                    save_annotation(annotation, OUTPUT_CSV)
                    processed_ids.add(item_id)
                    print(f"  Anotação de {item_id} salva com sucesso.")
                    time.sleep(4.1) # <-- E AQUI
                except Exception as e:
                    print(f"  ERRO ao anotar bloco de texto {item_id}: {e}")

    print(f"\nProcessamento concluído! Anotações salvas em: {OUTPUT_CSV}")

if __name__ == "__main__":
    annotate_pdf()