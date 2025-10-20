import os
import logging
import pandas as pd
import fitz
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

logger = logging.getLogger(__name__)

PASTA_DADOS = os.getenv('DATA_FOLDER', 'data')
TAMANHO_MAX_CHUNK = int(os.getenv('CHUNK_SIZE', 1024))
SOBREPOSICAO_CHUNK = int(os.getenv('CHUNK_OVERLAP', 128))

try:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=TAMANHO_MAX_CHUNK,
        chunk_overlap=SOBREPOSICAO_CHUNK,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", " ", ""],
    )
    logger.info("RecursiveCharacterTextSplitter inicializado com sucesso.")
except Exception as e:
    logger.error(f"Erro ao inicializar RecursiveCharacterTextSplitter: {e}", exc_info=True)
    text_splitter = None

def salvar_arquivo_recebido(nome_arquivo: str, bytes_arquivo: bytes) -> str | None:
    """
    Saves the bytes of a received file to the designated data directory.
    """
    if not os.path.exists(PASTA_DADOS):
        try:
            os.makedirs(PASTA_DADOS)
            logger.info(f"Diretório de dados '{PASTA_DADOS}' criado.")
        except OSError as e:
            logger.error(f"Erro ao criar diretório de dados '{PASTA_DADOS}': {e}", exc_info=True)
            return None

    caminho_arquivo = os.path.join(PASTA_DADOS, nome_arquivo)
    try:
        with open(caminho_arquivo, "wb") as f:
            f.write(bytes_arquivo)
        logger.info(f"Arquivo '{nome_arquivo}' salvo em '{caminho_arquivo}'.")
        return caminho_arquivo
    except IOError as e:
        logger.error(f"Erro de I/O ao salvar arquivo '{nome_arquivo}': {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Erro inesperado ao salvar arquivo '{nome_arquivo}': {e}", exc_info=True)
        return None

def dividir_texto_em_partes(texto_completo: str) -> list[str]:
    """
    Splits a long text into smaller chunks using the configured RecursiveCharacterTextSplitter.
    """
    if not texto_completo:
        logger.warning("Attempted to split an empty or None text.")
        return []
    if text_splitter is None:
        logger.error("Text splitter is not initialized. Cannot split text.")
        return [texto_completo.strip()] if texto_completo.strip() else []

    try:
        partes = text_splitter.split_text(texto_completo)
        partes_limpas = [p.strip() for p in partes if p.strip()]
        logger.debug(f"Texto dividido em {len(partes_limpas)} partes.")
        return partes_limpas
    except Exception as e:
        logger.error(f"Erro ao dividir texto: {e}", exc_info=True)
        return [texto_completo.strip()] if texto_completo.strip() else []

def extrair_textos_csv(caminho_arquivo: str) -> list[str]:
    """
    Extracts text chunks from a CSV file, assuming text is in a 'texto' column.
    """
    file_basename = os.path.basename(caminho_arquivo)
    try:
        df = pd.read_csv(caminho_arquivo)
        textos = []
        if 'texto' in df.columns:
            texto_completo = " ".join(df['texto'].dropna().astype(str).tolist())
            textos = dividir_texto_em_partes(texto_completo)
            logger.info(f"Texto extraído e dividido do CSV: {file_basename}")
        else:
            logger.warning(f"Coluna 'texto' não encontrada no arquivo CSV: {file_basename}")
        return textos
    except FileNotFoundError:
        logger.error(f"Arquivo CSV não encontrado: {caminho_arquivo}")
        return []
    except pd.errors.EmptyDataError:
        logger.warning(f"Arquivo CSV vazio: {file_basename}")
        return []
    except pd.errors.ParserError as e:
        logger.error(f"Erro ao parsear CSV {file_basename}: {e}")
        return []
    except Exception as e:
        logger.error(f"Erro inesperado ao processar CSV {file_basename}: {e}", exc_info=True)
        return []

def extrair_textos_txt(caminho_arquivo: str) -> list[str]:
    """
    Extracts text chunks from a plain text file, handling potential encoding issues.
    """
    file_basename = os.path.basename(caminho_arquivo)
    try:
        with open(caminho_arquivo, 'r', encoding='utf-8') as f:
            conteudo = f.read()
        logger.info(f"Texto extraído do TXT (UTF-8): {file_basename}")
        return dividir_texto_em_partes(conteudo)
    except FileNotFoundError:
        logger.error(f"Arquivo TXT não encontrado: {caminho_arquivo}")
        return []
    except UnicodeDecodeError:
        logger.warning(f"Erro de decodificação UTF-8 no TXT {file_basename}. Tentando latin-1.")
        try:
            with open(caminho_arquivo, 'r', encoding='latin-1') as f:
                conteudo = f.read()
            logger.info(f"Texto extraído do TXT (latin-1): {file_basename}")
            return dividir_texto_em_partes(conteudo)
        except Exception as e_alt:
            logger.error(f"Falha ao ler TXT {file_basename} mesmo com latin-1: {e_alt}", exc_info=True)
            return []
    except Exception as e:
        logger.error(f"Erro inesperado ao processar TXT {file_basename}: {e}", exc_info=True)
        return []

def extrair_textos_pdf(caminho_arquivo: str) -> list[str]:
    """
    Extracts text chunks from a PDF file, page by page.
    """
    texto_completo = ""
    file_basename = os.path.basename(caminho_arquivo)
    try:
        with fitz.open(caminho_arquivo) as doc:
            for num_pagina, pagina in enumerate(doc):
                try:
                    texto_completo += pagina.get_text("text") + "\n"
                except Exception as page_e:
                    logger.warning(f"Erro ao extrair texto da página {num_pagina + 1} do PDF {file_basename}: {page_e}")
        logger.info(f"Texto extraído com sucesso do PDF: {file_basename}")
        return dividir_texto_em_partes(texto_completo.strip())
    except FileNotFoundError:
        logger.error(f"Arquivo PDF não encontrado: {caminho_arquivo}")
        return []
    except fitz.fitz.EmptyFileError:
         logger.error(f"Arquivo PDF vazio ou inválido: {file_basename}")
         return []
    except fitz.fitz.FileDataError as e:
        logger.error(f"Erro nos dados do arquivo PDF {file_basename}: {e}")
        return []
    except Exception as e:
        logger.error(f"Erro inesperado ao processar PDF {file_basename}: {e}", exc_info=True)
        return []
