# Healthcare RAG Guardrails

> [!IMPORTANT]
> This is currently a work in progress, expect things to be broken!

Demonstrate NeMo Guardrails on a locally running Llama-2-7b model with 100 medical records in a ChromaDB vector database.

Tested on HPE DL380, AMD CPU and Nvidia L40S GPU

# Install
huggingface-cli download NousResearch/Llama-2-7b-chat-hf

huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 

edit config/config.yml to replace path to your model 

conda create nemo-env 

conda activate nemo-env

pip install -r requirements.txt 

python3 csv2chromadb.py

python3 main.py

## 

* __NeMo Guardrails:__ Safety framework for LLMs, ensuring adherence to guidelines and reducing bias/toxicity.
* __LangChain:__ Framework for building language-based applications, streamlining model retrieval and combination.
* __ChromaDB:__ Vector database for efficient text search and retrieval. 
* __Hugging Face Pipelines:__ Streamlined model loading and execution. 
* __Llama-2 7B LLM:__ Large language model employed for text generation.
* __Gradio:__ Interface library for web-based interaction with the model. 







