import os
import pandas as pd
from dotenv import load_dotenv

from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

DATA_PATH = "IMDB-Movie-Data.csv"
VECTOR_DB_PATH = "vectorstore"


def load_csv_data(csv_path: str):
    """Load movie dataset."""

    df = pd.read_csv(csv_path)
    df = df.fillna("")

    return df



def create_documents(df):
    """Convert dataframe rows into LangChain documents."""

    documents = []

    for _, row in df.iterrows():
        content = f"""
        Title: {row['Title']}
        Genre: {row['Genre']}
        Description: {row['Description']}
        Director: {row['Director']}
        Actors: {row['Actors']}
        Year: {row['Year']}
        Runtime: {row['Runtime (Minutes)']} minutes
        Rating: {row['Rating']}
        Votes: {row['Votes']}
        Revenue: {row['Revenue (Millions)']} million
        Metascore: {row['Metascore']}
        """

        metadata = {
            "title": row['Title'],
            "genre": row['Genre'],
            "director": row['Director'],
            "year": row['Year'],
            "rating": row['Rating']
        }

        documents.append(
            Document(
                page_content=content,
                metadata=metadata
            )
        )

    return documents



def get_embedding_model():
    """Load HuggingFace embedding model."""

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings

def build_vector_store(documents, embeddings):
    """Create FAISS vector store."""

    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )

    vectorstore.save_local(VECTOR_DB_PATH)

    print("✅ FAISS vector database created successfully")

def main():
    print("Loading dataset...")

    df = load_csv_data(DATA_PATH)

    print(f"Loaded {len(df)} movies")

    print("Creating documents...")

    documents = create_documents(df)

    print("Loading embedding model...")

    embeddings = get_embedding_model()

    print("Building FAISS index...")

    build_vector_store(documents, embeddings)

    print("✅ Ingestion completed successfully")

if __name__ == "__main__":
    main()