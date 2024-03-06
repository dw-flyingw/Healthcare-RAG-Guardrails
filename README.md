# Healthcare RAG Guardrails

> [!IMPORTANT]
> This is currently a work in progress, expect things to be broken!

Demonstrate NeMo Guardrails on a locally running Llama-27b model with 100 medical records in a ChromaDB vector embeddings database.

# Install
huggingface-cli download NousResearch/Llama-2-7b-chat-hf

huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 

edit config/config.yml to replace path to your model 

conda create nemo-env 

conda activate nemo-env

pip install -r requirements.txt 

## 
<ul>\
<li><b>NeMo Guardrails:</b> Safety framework for LLMs, ensuring adherence to guidelines and reducing bias/toxicity. <br>\
<li><b>LangChain:</b> Framework for building language-based applications, streamlining model retrieval and combination. <br>\
<li><b>ChromaDB:</b> Vector database for efficient text search and retrieval. <br>\
<li><b>Hugging Face Pipelines:</b> Streamlined model loading and execution. <br>\
<li><b>Llama-2 7B LLM:</b> Large language model employed for text generation. <br>\
<li><b>Gradio:</b> Interface library for web-based interaction with the model. <br>\
</ul>\






