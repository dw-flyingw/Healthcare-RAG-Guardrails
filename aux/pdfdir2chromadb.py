#!/usr/bin/env python3
# run in /home/l40s/dave/healthcare on l40s@10.14.56.23
# conda activate dave-nemo-env
#https://www.gettingstarted.ai/tutorial-chroma-db-best-vector-database-for-langchain-store-embeddings/
#https://blog.futuresmart.ai/using-langchain-and-open-source-vector-db-chroma-for-semantic-search-with-openais-llm

import langchain
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader, TextLoader
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from chromadb import PersistentClient

# Create a persistent ChromaDB client
pdfdir = "./pdfs/"
chromadb_path = "./chromadb"
collection_name = "bloodtests"
chroma_client = PersistentClient(path=chromadb_path)

# Get a list of all existing collections
collections = chroma_client.list_collections()
# Remove the medical_data collection if exist
if collections:
    for collection in collections:
        if collection_name == collection.name:
            print(f"removing {collection.name} collection")
            chroma_client.delete_collection(collection_name)

# create new collection
chroma_collection = chroma_client.create_collection(collection_name)

# https://python.langchain.com/docs/integrations/vectorstores/chroma
# Load documents from a directory
print ("reading PDFs")
loader = PyPDFDirectoryLoader(pdfdir)
documents = loader.load()

# https://medium.com/@kenzic/getting-started-chunking-strategy-ebd4ab81f745
# split it into chunks
print ("splitting and chunking")
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
split_docs = text_splitter.split_documents(documents)

# Choose an LLM and embedding model
#embed_model_name = "thenlper/gte-large"
embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embed_model = HuggingFaceEmbeddings(model_name=embed_model_name)

chroma_db = Chroma.from_documents(
    documents=split_docs, 
    embedding=embed_model, 
    persist_directory=chromadb_path,
    collection_name=collection_name,
)

# Save the Chroma database to disk
chroma_db.persist()

# query it just to be sure it worked
query = "Lauren Green"
response = chroma_db.similarity_search(query)
print (f"search for {query}")
print (response[0].page_content) # just show the content from that relevant pdf

#print("There are", chroma_db._collection.count(), "in the collection")

# all the data in the db
#print (chroma_db.get())


