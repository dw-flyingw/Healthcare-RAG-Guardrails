#!/usr/bin/env python3
# run in /home/l40s/dave/healthcare on l40s@10.14.56.23
# conda activate dave-nemo-env

from colorama import Fore, Style, init
init(autoreset=True) 
import pandas as pd
import tqdm
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from chromadb import PersistentClient

# Choose an LLM and embedding model
#embed_model_name = "thenlper/gte-large"
embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embed_model = HuggingFaceEmbeddings(model_name=embed_model_name)

csv_dataset = "./datasets/healthcare_dataset-100.csv"
collection_name = "healthcare_dataset"
chromadb_path = "./chromadb"

# Create a persistent ChromaDB client
chroma_client = PersistentClient(path=chromadb_path)

# Get a list of all existing collections
collections = chroma_client.list_collections()
# Remove the collection collection if exist
if collections:
    for collection in collections:
        if collection_name == collection.name:
            print(f"removing {collection.name} collection")
            chroma_client.delete_collection(collection_name)

# create new collection
chroma_collection = chroma_client.create_collection(collection_name)

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_dataset)
pd.options.display.max_colwidth = 500
print(Fore.GREEN + str(df)) # view dataset sample

# Create column for NLP use
df.loc[:, 'NLP'] = 'patient name is ' + df['Name'] + ' ' + df['Age'].apply(str) + ' years old' \
                    + ' blood type is ' + df['Blood Type'] + ' with a medical condition of ' + df['Medical Condition'] \
                    + ' their doctor is ' + df['Doctor'] + ' from the ' + df['Hospital'] + ' hospital' \
                    + ' their innsurance provider is ' + df['Insurance Provider'] \
                    + ' they are taking ' + df['Medication'] + ' medication' \
                    + ' and their test results are ' +df['Test Results']

# print sample of the new NLP string column
#print(Fore.BLUE + df.iloc[[0]]['NLP'])

# create new list from new NLP column 
docs = df['NLP'].tolist() 
# add the row number as the id to associate the concatanated record
ids = [str(x) for x in df.index.tolist()]

#chroma_collection.add( documents = docs, ids = ids )
# same as above but with a progress bar as this can take a while
num_documents = len(docs) 
with tqdm.tqdm(total=num_documents, desc="Adding documents") as pbar:
    for i in range(num_documents):
        chroma_collection.add(documents=[docs[i]], ids=[ids[i]])
        pbar.update(1)  # Update the progress bar after each document is added

# Choose an LLM and embedding model
#embed_model_name = "thenlper/gte-large"
embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embed_model = HuggingFaceEmbeddings(model_name=embed_model_name)

# Create a database with the embedings from the collection
langchain_chroma = Chroma(
    client=chroma_client,
    collection_name=collection_name,
    embedding_function=embed_model, 
    persist_directory=chromadb_path,
 
)
# Save the Chroma database to disk
langchain_chroma.persist()

# Test Query
query = "What is Patrick Parker's Blood Type?"
print (Fore.RED + f"{query}")
response = langchain_chroma.similarity_search(query)
top_response =  (response[0].page_content) # Top 3 are returned
print (Fore.BLUE + str(top_response))



