# Healthcare RAG Guardrails

Demonstrate NeMo Guardrails on a locally running Llama-27b model with 100 medical records in a ChromaDB vector embeddings database.

# Install
huggingface-cli download NousResearch/Llama-2-7b-chat-hf
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2
edit config/config.yml to replace path to your model

conda create nemo-env
conda activate nemo-env

pip install -r requirements.txt






