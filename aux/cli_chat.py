#!/usr/bin/env python3
# conda activate dave-nemo-env

import sys
import setproctitle # Friendly name for nvidia-smi GPU Memory Usage
setproctitle.setproctitle('dave\'s guardrails cli')
from colorama import Fore, Style
import transformers
import torch
from nemoguardrails import LLMRails, RailsConfig
#from langchain_core.output_parsers import StrOutputParser
#from langchain_core.runnables import RunnablePassthrough
#from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

# GPU settings
#n_gpu_layers = 0 # for CPU
n_gpu_layers = -1  # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU. 0 means all in CPU
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

# LLM Hyperparameters
temperature = 0.1 # .1 is the lowest, 0 indicates greedy
max_tokens = 150 # this can slow down the response time
top_p = 0.95
top_k = 2 # is this not the same as "k": 2 in rag_chain?
context_window = 4096  # max is 4096
repetition_penalty = 1.1 # why is this not like maybe 10
seed = 22 # For reproducibility

model_id = "NousResearch/Llama-2-7b-chat-hf" 
#model = AutoModelForCausalLM.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, temperature=temperature)
tokenizer = AutoTokenizer.from_pretrained(model_id)

model.generation_config = GenerationConfig(
                                    temperature=temperature,
                                    max_new_tokens=max_tokens,
                                    do_sample=True,
                                    top_k=top_k,
                                    top_p=top_p,
                                    repetition_penalty=repetition_penalty,
                                    return_full_text=True,
                                    )
#print (Fore.GREEN + str(model.generation_config))   

# put the model into a pipeline
pipeline = transformers.pipeline("text-generation", 
                                 model = model,
                                 tokenizer = tokenizer,                                 
                                 torch_dtype = torch.bfloat16, 
                                 device = torch.device('cuda'),
                                 #max_new_tokens=max_tokens,
                                 #temperature=temperature,
                                 #do_sample=True,
                                 #return_full_text=True,
                                 #top_k=top_k,
                                 #top_p=top_p,
                                 )

# Wrap the pipline with Langchain Huggingface Pipeline class
hfpipeline = HuggingFacePipeline(
                                
                                pipeline=pipeline,
                                model_id=model_id,
                                verbose=True,
                                model_kwargs = {'temperature':temperature}
                                )

# Configuration of LLMs is passed
rail_config = RailsConfig.from_path("../config")
rail_app = LLMRails(config=rail_config, llm=hfpipeline)

#prompt = "How to make lemonade?"
#generated = hfpipeline.invoke(prompt) 
#generated = rail_app.generate(prompt)
#print (Fore.BLUE + str(generated))
#print (Fore.GREEN + str(model.generation_config)) 
#print (Fore.GREEN + str(hfpipeline))

# Main function to run the chatbot
def main():
    print(Fore.GREEN + "Chatbot: Hi! How can I help you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            print(Style.RESET_ALL)
            break
        response = rail_app.generate(user_input)
        #response = hfpipeline.invoke(user_input) 
        print("Chatbot:", response)
        print(Style.RESET_ALL)

# Run the chatbot
if __name__ == "__main__":
    main()