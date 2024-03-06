#!/usr/bin/env python3
# conda activate dave-nemo-env
# Alejandro Morales & Dave Wright and lots of trial and error
# Langchain RAG Demo with NeMo Guardrails and ChromaDB

import os
import time
import sys
import logging
#logging.basicConfig(stream=sys.stdout, level=logging.INFO)
#logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
import asyncio # needed for NeMo Guardrails
import torch
import chromadb
import transformers
import gradio as gr
import setproctitle # Friendly name for nvidia-smi GPU Memory Usage
setproctitle.setproctitle('Healthcare RAG Guardrails')
#output readability
from colorama import Fore, init
init(autoreset=True) 
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from langchain.prompts import PromptTemplate
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate

# Hugging Face Optimize for NVIDIA
# from optimum.nvidia import AutoModelForCausalLM 
# https://github.com/huggingface/optimum-nvidia
# cant use just yet only in docker container or source and source can't be built on this box at the moment

# NVIDIA NeMo Guardrails
from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.llm.helpers import get_llm_instance_wrapper  
from nemoguardrails.llm.providers import register_llm_provider

# Environment Variables and Keys
#os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
os.environ['NGC_API_KEY'] = 'b2Myb3V0cGZsbnMyMnI5cDFmdW5obGJlOWc6MWYyYWNmMTMtYjNhYi00OGZlLTg0MDQtYmVlMmRhNTIzNjgy'
#os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define embedding model for the vectordb
#embed_model_name = "thenlper/gte-large"
embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embed_model = HuggingFaceEmbeddings(model_name=embed_model_name)

# Load ChromaDB vectors from disk
#collection_name = "bloodtests" # pdf bloodtests
collection_name = "healthcare_dataset" # 100 medical records

# Load ChromaDB vectors from disk
mydb = chromadb.PersistentClient(path="./chromadb")
chroma_collection = mydb.get_or_create_collection(collection_name)
langchain_chroma = Chroma(
    client=mydb,
    collection_name=collection_name,
    embedding_function=embed_model
)

###### Select Models
#https://huggingface.co/NousResearch/Llama-2-7b-chat-hf
model_name = "NousResearch/Llama-2-7b-chat-hf" # Jordan Nanos likes this one

#waiting for chat-hf version of this model
#https://huggingface.co/BioMistral/BioMistral-7B
#model_name = "BioMistral/BioMistral-7B" 

# What is the difference between LlamaForCausalLM and AutoModelForCausalLM?
#model = LlamaForCausalLM.from_pretrained(model_name) # this works too
#tokenizer = LlamaTokenizer.from_pretrained(model_name) # this works too
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# GPU settings
#n_gpu_layers = 0 # for CPU
n_gpu_layers = -1  # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU. 0 means all in CPU
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

# LLM Hyperparameters
temperature = 0.25 # might need to be lower for guardrails
max_tokens = 150 # this can slow down the response time
top_p = 0.95
top_k = 2 # is this not the same as "k": 2 in rag_chain?
context_window = 4096  # max is 4096
repeat_penalty = 1.1 # why is this not like maybe 10
seed = 22 # For reproducibility

# put the model into a pipeline
pipeline = transformers.pipeline("text-generation", 
                                 model = model,
                                 tokenizer = tokenizer,                                 
                                 torch_dtype = torch.float16, 
                                 #device = "cpu", 
                                 device = torch.device('cuda'), 
                                 max_new_tokens=max_tokens,
                                 temperature=temperature,
                                 do_sample=True,
                                 return_full_text=True,
                                 top_k=top_k,
                                 top_p=top_p,
                                 #use_fp8=True, # Hugging Face Optimize for NVIDIA
                                 )

# Wrap the pipline with Langchain Huggingface Pipeline class
hfpipeline = HuggingFacePipeline(pipeline=pipeline, verbose=True)

# Wrap the HuggingFacePipeline with Nemo’s get_llm_instance_wrapper function and register it using register_llm_provider.
# Create NeMo Rails
NeMoPipeline = get_llm_instance_wrapper(llm_instance=hfpipeline, llm_type="hf_pipeline_llama2")
register_llm_provider("hf_pipeline_llama2", NeMoPipeline) 
rails_config = RailsConfig.from_path("./config") # config.yml and topics.co

# Do Voodoo here!
initial_prompt_template = """
[INST] <<SYS>>
Instruction:  You are a helpful clinical AI assistant that can answer patient's questions about their medical records. 
              Use non-technical medical terms that can be understood by everyone. 
              Avoid using acronyms and any other medical terms that are technical. 
              If you do not know the answer just say you do not know the answer.
              Be concise and to the point when answering the question. Below is an example:

Context: patient name is Tiffany Ramirez 81 years old blood type is O- with a medical condition of Diabetes their doctor is Patrick Parker from the Wallace-Hamilton hospital their innsurance provider is Medicare they are taking Aspirin medication and their test results are Inconclusive
            
Question: What is Tiffany Ramirez's Blood Type?

Answer: Based on our medical records Tiffany Ramirez's blood type is O-. O- negative is known as the "universal donor" because its red blood cells can be safely transfused to anyone, regardless of their blood type.  O- negative is a relatively rare blood type, found in only about 7% of the population. 

<</SYS>>
Context: {context}

Question: {question} 

Answer:
[/INST]
 """

# Create prompt from prompt template 
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=initial_prompt_template,
)

# Ask Alejandro why do we need to put newlines around each thing?
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


# Create llm rag chain
# need to understand RnnablePassthrough and StrOutParser really works and what is the | format
# "k": 2 selects the top 2 closest vectors
# ensures the question remains unchanged
rag_chain = (
    {"context": langchain_chroma.as_retriever(search_kwargs={"k": 1}) | format_docs, "question": RunnablePassthrough()}
    | prompt
    | hfpipeline
    | StrOutputParser() 
)


async def generate_guarded_response(query):
    response = rag_chain.invoke(query) 
    print (Fore.GREEN + 'generate_guarded_response query ' + Fore.BLUE + str(query))
    print (Fore.GREEN + 'generate_guarded_response response ' + Fore.BLUE + str(response))
    return response

# Register async functions for use in rails
rag_rails = LLMRails(config=rails_config, llm=hfpipeline)
#rag_rails.register_action(action=similarity_search, name="similarity_search")
rag_rails.register_action(action=generate_guarded_response, name="generate_guarded_response")

async def generate_text(prompt,temperature):
    # Use temperature value from gradio slider
    hfpipeline.pipeline.temperature = temperature 
    print (Fore.RED + 'prompt ' + Fore.BLUE + str(prompt))

    guarded = await rag_rails.generate_async(prompt) # pass it through  NeMo Guardrails
    generated = rag_chain.invoke(prompt) # unguarded generated response 
   
    print (Fore.RED + 'hfpipeline temperature ' + Fore.BLUE + str(hfpipeline.pipeline.temperature))
    print (Fore.RED + 'guarded response ' + Fore.BLUE + str(guarded)) 
    print (Fore.RED + 'generated response ' + Fore.BLUE + str(generated))
    # the return order must match with the gradio interface output order
    return guarded, generated


#############################
# Create a Gradio interface #
#############################

title = "Retrieval Augmented Generation with Nvidia NeMo Guardrails"
description = f"embedings = {embed_model_name} <br>  \
               model = {model_name} <br> \
               nemo guardrail engine = {model_name} <br> \
               chromadb = 100 record {collection_name}"       
article = " <p>\
<ul>\
<li><b>NeMo Guardrails:</b> Safety framework for LLMs, ensuring adherence to guidelines and reducing bias/toxicity. <br>\
<li><b>LangChain:</b> Framework for building language-based applications, streamlining model retrieval and combination. <br>\
<li><b>ChromaDB:</b> Vector database for efficient text search and retrieval. <br>\
<li><b>Hugging Face Pipelines:</b> Streamlined model loading and execution. <br>\
<li><b>Llama-2 7B LLM:</b> Large language model employed for text generation. <br>\
<li><b>Gradio:</b> Interface library for web-based interaction with the model. <br>\
</ul>\
"
demo = gr.Interface(
                   fn=generate_text, 
                   inputs=[
                            gr.Textbox(label="Prompt", placeholder="select an Example to submit"),
                            gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, value=temperature),
                            ],
                   outputs=[
                            gr.Textbox(label="Guarded Response"),
                            gr.Textbox(label="Generated Response", elem_id="warning"), 
                            ],
                   title=title, 
                   description=description, 
                   allow_flagging="never", 
                   theme='upsatwal/mlsc_tiet', # Dark theme large fonts  huggingface hosted
                   examples=[
                              ["I am Sally Shaw, what is my blood type?"],
                              ["I am Angela Brown, what were my latest test results?"], 
                              ["I am Michael Bradshaw, where do I have healthcare insurance?"], 
                              ["I am Amanda Ortiz, who is my doctor?"],
                              ["I am Amy Roberts, do I have cancer? "],
                              ["What are your political beliefs?"],
                              ["How do I cook spaghetti?"],
                              ["How can I make a ghost gun?"],
                            ],
                    article=article, # HTML to display under the Example prompt buttons
                    # removes default gradio footer
                    css="""
                        footer{display:none !important} 
                        """
                   )

# this binds to all interfaces which is needed for proxy forwarding
demo.launch(server_name="0.0.0.0", server_port=7865)