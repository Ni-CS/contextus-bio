# Contextus Bio: Painel Comparativo de Buscas Semânticas

## 1\. Introdução

O **Contextus Bio** é uma aplicação web interativa projetada para explorar e comparar diferentes arquiteturas de busca semântica em documentos complexos. Utilizando a apostila "Biologia Celular" como estudo de caso, este projeto vai além da busca por palavras-chave, implementando e avaliando quatro metodologias distintas de Inteligência Artificial para entender e ranquear a relevância de textos e imagens com base em seu significado contextual.

A aplicação final permite que o usuário selecione em tempo real qual motor de busca utilizar, oferecendo uma plataforma robusta para a análise comparativa de técnicas de embedding e busca multimodal.

## 2\. Funcionalidades Principais

  - **Painel Comparativo:** Selecione entre quatro diferentes abordagens de IA na barra lateral para comparar os resultados da busca.
  - **Busca Semântica Avançada:** Encontre conteúdo relevante com base na intenção da sua busca, não apenas em palavras exatas.
  - **Resultados Multimodais:** A busca retorna tanto os trechos de texto quanto as imagens mais relevantes.
  - **Visualizador de PDF Interativo:** Navegue pelo documento, pule para a página de um resultado com um clique e veja o conteúdo em seu contexto original.
  - **Destaque de Texto Inteligente:** Resultados textuais são automaticamente destacados com um "marca-texto virtual" no visualizador.

## 3\. As Quatro Abordagens de Busca

Este projeto implementa e permite a comparação das seguintes metodologias:

1.  **Abordagem 1: Descrições (Especialista em Texto)**

      - **Conceito:** A abordagem mais robusta. Utiliza um LLM (Gemini) para gerar descrições textuais ricas para todo o conteúdo (textos e imagens). Em seguida, um modelo especialista em texto (`intfloat/multilingual-e5-large`) vetoriza essas descrições. A busca é puramente texto-vs-texto, de alta precisão semântica.

2.  **Abordagem 2: Híbrida (CLIP Visual + Texto)**

      - **Conceito:** Uma abordagem mista. A busca de texto usa o modelo especialista (`e5-large`), enquanto a busca de imagem usa vetores híbridos gerados pelo CLIP, que são uma média ponderada do conteúdo visual (pixels) e do conteúdo textual (descrição) da imagem.

3.  **Abordagem 3: Multimodal Pura (Tudo com CLIP)**

      - **Conceito:** Utiliza exclusivamente o modelo CLIP para gerar todos os vetores. Os vetores de imagem são gerados a partir dos pixels, e os de texto a partir do codificador de texto do CLIP. Garante um espaço semântico unificado.

4.  **Abordagem 4: CLIP Híbrido Unificado**

      - **Conceito:** O refinamento da Abordagem 3. Utiliza apenas o CLIP, mas enriquece os vetores de imagem criando um vetor híbrido (visual + textual), tornando-os mais contextuais e comparáveis com os vetores de texto, tudo dentro do mesmo espaço de embedding do CLIP.

## 4\. Arquitetura e Fluxo de Trabalho

O projeto segue um pipeline claro, separando a construção das bases de dados da execução da aplicação final.

```mermaid
graph TD
    A[data/annotations.csv<br>(Fonte da Verdade)] --> B(scripts/build_approach_1.py);
    A --> C(scripts/build_approach_2.py);
    A --> D(scripts/build_approach_3.py);
    A --> E(scripts/build_approach_4.py);

    B --> F[database/approach_1_text_only];
    C --> G[database/approach_2_hybrid_clip];
    D --> H[database/approach_3_pure_clip];
    E --> I[database/approach_4_unified_hybrid];

    subgraph "Fase 1: Construção das Bases de Dados"
        A; B; C; D; E;
    end

    F & G & H & I --> J{app/search_app.py<br>(com seletor)};

    subgraph "Fase 2: Execução da Aplicação"
        J;
    end

    J --> K[Interface Web Interativa];
```

## 5\. Estrutura de Arquivos

O projeto está organizado na seguinte estrutura para máxima clareza e manutenibilidade:

```
/smart-search/
├── app/
│   └── search_app.py              # Script da aplicação final com Streamlit
│
├── data/
│   ├── Biologia-Celular.pdf       # Documento original
│   ├── annotations.csv          # Dataset bruto, curado manualmente
│   └── annotations_clean.csv      # Versão limpa, gerada pelos scripts
│
├── database/
│   ├── approach_1_text_only/      # Arquivos (.npy, .csv) para a Abordagem 1
│   ├── approach_2_hybrid_clip/    # Arquivos (.npy, .csv) para a Abordagem 2
│   ├── approach_3_pure_clip/      # Arquivos (.npy, .csv) para a Abordagem 3
│   └── approach_4_unified_hybrid/ # Arquivos (.npy, .csv) para a Abordagem 4
│
├── notebooks/
│   └── exploratory_data_analysis.ipynb
│
├── scripts/
│   ├── build_approach_1.py        # Scripts para gerar as bases de dados
│   ├── build_approach_2.py
│   ├── build_approach_3.py
│   └── build_approach_4.py
│
├── requirements.txt               # Dependências do projeto
└── README.md                      # Este arquivo
```

## 6\. Instalação e Configuração

**1. Ambiente Virtual**
Recomenda-se o uso de um ambiente virtual para isolar as dependências do projeto.

```bash
python -m venv .venv
source .venv/bin/activate
```

**2. Instalar Dependências**
Instale todos os pacotes necessários de uma vez usando o arquivo `requirements.txt`.

```bash
pip install -r requirements.txt
```

## 7\. Como Executar o Projeto

#### Passo 1: Construir as Bases de Dados

Para usar o painel comparativo, você precisa primeiro gerar os arquivos de embedding para cada abordagem. Execute os scripts correspondentes a partir da pasta raiz do projeto.

```bash
# Para a Abordagem 1 (obrigatório, pois a app usa seu modelo por padrão)
python scripts/build_text_only_approach.py

# Para as outras abordagens (opcional, execute as que desejar comparar)
python scripts/build_hybrid_approach.py
python scripts/build_pure_clip_approach.py
python scripts/build_unified_hybrid.py
```

*(Nota: Os nomes dos seus scripts em `scripts/` podem ser ligeiramente diferentes. Ajuste os comandos conforme necessário).*

Cada script criará a pasta e os arquivos necessários dentro de `database/`.

#### Passo 2: Iniciar a Aplicação

Com pelo menos uma base de dados construída, inicie a aplicação Streamlit.

```bash
streamlit run app/search_app.py
```

A aplicação abrirá no seu navegador. Use a barra lateral para selecionar a metodologia que deseja testar e explore os resultados\!