from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv



load_dotenv()
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")

vector_store = Chroma(
    embedding_function=embeddings ,
    collection_name="my_pdf_docs" ,
    persist_directory="./Chroma_db"
)