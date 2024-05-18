import os
import time
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class VectorDB:
    def __init__(self, persist_dir="chroma/", chunk_size=1500, chunk_overlap=500):
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.persist_dir = persist_dir
        self.embedding = OpenAIEmbeddings(show_progress_bar=True)

    def load_pdf_documents(self):
        data_dir = "data/papers"
        document_list = os.listdir(data_dir)
        docs = []
        for doc_name in document_list:
            doc_path = os.path.join(data_dir, doc_name)
            doc = PyPDFLoader(doc_path).load()
            docs.extend(doc)
        print("Number of documents loaded: ", len(document_list))
        print("Nmber of pages: ", len(docs))
        return docs
    
    def load_text_documents(self):
        data_dir = "data/texts"
        document_list = os.listdir(data_dir)
        docs = []
        for doc_name in document_list:
            doc_path = os.path.join(data_dir, doc_name)
            doc = TextLoader(doc_path).load()
            docs.extend(doc)
        print("Number of documents loaded: ", len(document_list))
        return docs

    def load_all_documents(self):
        pdf_data = self.load_pdf_documents()
        web_data = self.load_text_documents()
        return pdf_data + web_data
    
    def chunk_doc(self, docs):
        print("Chunking documents...")
        chunked_docs = self.text_splitter.split_documents(docs)
        print("Number of chunks: ", len(chunked_docs))
        return chunked_docs
    
    def embed_docs(self):
        docs = self.load_all_documents()
        chunked_docs = self.chunk_doc(docs)
        print("Embedding chunks...")
        vectordb = Chroma(
            persist_directory=self.persist_dir,
            collection_name="documents",
        )
        small_chunks = [chunked_docs[:50], chunked_docs[50:100], chunked_docs[100:150], chunked_docs[150:]]
        for small_chunk in small_chunks:
            vectordb.from_documents(
                documents=small_chunk,
                embedding=self.embedding,
                collection_name="documents",
                persist_directory=self.persist_dir
            )
            print("Iteration #vectors: ", vectordb._collection.count())
            time.sleep(60)
        print("Done embedding.")
        print("Number of vectors: ", vectordb._collection.count())
        return vectordb

if __name__ == "__main__":
    db = VectorDB()
    db.embed_docs()