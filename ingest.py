import os
import glob
import logging
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from pinecone.core.client.exceptions import PineconeApiException

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

INDEX_NAME = "hse-sop-ultimate-index"
DATA_FOLDER = "data"

def validate_environment():
    if not os.getenv("PINECONE_API_KEY"):
        logger.error("❌ PINECONE_API_KEY is missing in .env file.")
        sys.exit(1)
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("❌ OPENAI_API_KEY is missing in .env file.")
        sys.exit(1)

def run_ingestion():
    validate_environment()

    if not os.path.exists(DATA_FOLDER):
        try:
            os.makedirs(DATA_FOLDER)
            logger.warning(f"⚠️ Created '{DATA_FOLDER}' directory. Please add PDFs and rerun.")
            return
        except OSError as e:
            logger.error(f"❌ Failed to create directory: {e}")
            return

    pdf_files = glob.glob(os.path.join(DATA_FOLDER, "*.pdf"))
    if not pdf_files:
        logger.warning(f"⚠️ No PDFs found in '{DATA_FOLDER}/'. Skipping ingestion.")
        return

    logger.info(f"--- Starting Ingestion for {INDEX_NAME} ---")

    try:
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        existing_indexes = [index.name for index in pc.list_indexes()]
        
        if INDEX_NAME not in existing_indexes:
            logger.info(f"Creating index: {INDEX_NAME}...")
            pc.create_index(
                name=INDEX_NAME,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        else:
            logger.info(f"Index '{INDEX_NAME}' already exists.")
            
    except PineconeApiException as e:
        logger.error(f"❌ Pinecone API Error: {e}")
        return
    except Exception as e:
        logger.error(f"❌ Unexpected error connecting to Pinecone: {e}")
        return

    try:
        logger.info("Loading PDFs...")
        loader = DirectoryLoader(DATA_FOLDER, glob="**/*.pdf", loader_cls=PyPDFLoader)
        raw_docs = loader.load()

        if not raw_docs:
            logger.warning("⚠️ PDFs loaded but no text found. They might be scanned images.")
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(raw_docs)
        logger.info(f"Processing {len(documents)} text chunks...")

    except Exception as e:
        logger.error(f"❌ Error during file processing: {e}")
        return

    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        logger.info("Upserting vectors to Pinecone...")
        PineconeVectorStore.from_documents(documents=documents, index_name=INDEX_NAME, embedding=embeddings)
        logger.info("✅ Ingestion Complete Successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to upload vectors: {e}")

if __name__ == "__main__":
    run_ingestion()