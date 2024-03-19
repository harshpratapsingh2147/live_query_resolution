from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from decouple import config


BASE_TRANSCRIPT_PATH = config('BASE_TRANSCRIPT_PATH')

persistent_directory = config('PERSISTENT_DIRECTORY')

api_key = config('OPEN_AI_API_KEY')


def load_text_file(class_id):
    loader = TextLoader(f"{BASE_TRANSCRIPT_PATH}{class_id}_transcript.txt")
    pages = loader.load()
    return pages


def recursive_text_splitter(pages):
    chunk_size = 500
    chunk_overlap = 4

    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    docs = r_splitter.split_documents(pages)
    # print(docs)
    return docs


def embed_data(docs):
    embedding = OpenAIEmbeddings(api_key=api_key)
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        persist_directory=persistent_directory
    )
    return vectordb
