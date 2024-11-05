import os
import time
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class VectorDB:
    def __init__(self, persist_dir="chroma/", chunk_size=1500, chunk_overlap=500):
        """
            chunk_size: The size of each chunk of text to embed.
            chunk_overlap: The number of characters to overlap between chunks.
        """
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.persist_dir = persist_dir
        self.embedding = OpenAIEmbeddings(show_progress_bar=True)
        self.pdf_dir = "data/papers"
        self.text_dir = "data/texts"

    def load_pdf_documents(self):
        document_list = os.listdir(self.pdf_dir)
        docs = []
        for doc_name in document_list:
            doc_path = os.path.join(self.pdf_dir, doc_name)
            doc = PyPDFLoader(doc_path).load()
            docs.extend(doc)
        print("Number of documents loaded: ", len(document_list))
        print("Nmber of pages: ", len(docs))
        return docs
    
    def load_text_documents(self):
        document_list = os.listdir(self.text_dir)
        docs = []
        for doc_name in document_list:
            doc_path = os.path.join(self.text_dir, doc_name)
            doc = TextLoader(doc_path).load()
            docs.extend(doc)
        print("Number of documents loaded: ", len(document_list))
        return docs

    def load_all_documents(self):
        # check if the papers directory exists
        if not os.path.exists(self.pdf_dir):
            print("There is no papers directory. Skip")
            pdf_data = []
        else:
            pdf_data = self.load_pdf_documents()

        # check if the texts directory exists
        if not os.path.exists(self.text_dir):
            print("There is no texts directory. Skip")
            web_data = []
        else:
            web_data = self.load_text_documents()

        return pdf_data + web_data
    
    def chunk_doc(self, docs):
        print("Chunking documents...")
        chunked_docs = self.text_splitter.split_documents(docs)
        print("Number of chunks: ", len(chunked_docs))
        return chunked_docs
    
    def embed_docs(self):
        docs = self.load_all_documents()
        if len(docs) == 0:
            print("No documents to embed.")
            return
        
        chunked_docs = self.chunk_doc(docs)
        print("Embedding chunks...")
        vectordb = Chroma(
            persist_directory=self.persist_dir,
            collection_name="documents",
        )

        batch_size = 30
        i = 0
        while i < len(chunked_docs):
            batch = chunked_docs[i:i+batch_size]
            try:
                vectordb.from_documents(
                    documents=batch,
                    embedding=self.embedding,
                    collection_name="documents",
                    persist_directory=self.persist_dir
                )
                i += batch_size
                print("Embedded ", len(batch), " documents.")
            except Exception as e:
                print("An error occurred. Sleeping for 60 seconds before retrying.")
                time.sleep(60) # sleep for 60 seconds to avoid rate limit

        print("Done embedding.")
        print("Number of vectors: ", vectordb._collection.count())
        return vectordb

if __name__ == "__main__":
    db = VectorDB()
    db.embed_docs()