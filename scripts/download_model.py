import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set the cache directory
os.environ["TRANSFORMERS_CACHE"] = "/vol/csedu-nobackup/course/I00041_informationretrieval/users/aditya/RAG-Uncertainty-Estimator/models/huggingface_cache"

# Define model path
model_name = "Qwen/Qwen2.5-7B-Instruct"
local_model_path = "./models/qwen2.5-7B-instruct"

# Download and load model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Save locally
model.save_pretrained(local_model_path)
tokenizer.save_pretrained(local_model_path)
