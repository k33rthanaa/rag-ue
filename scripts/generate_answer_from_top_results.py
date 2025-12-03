import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

def load_model(model_path, device):
    """Load the model and tokenizer for answer generation."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model = model.to(device)
    return tokenizer, model

def generate_answer(query, top_results, tokenizer, model, device, max_length=512):
    """Generate an answer from the top results."""
    # Combine the query and the top 5 results as the context
    context = "\n".join([f"Document {i+1}: {result['snippet']}" for i, result in enumerate(top_results)])
    
    prompt = f"Question: {query}\nContext:\n{context}\nAnswer:"
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=max_length, truncation=True).to(device)
    
    # Generate the answer using the model
    outputs = model.generate(inputs.input_ids, max_new_tokens=150, num_return_sequences=1, no_repeat_ngram_size=2)
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

def main(query, top_results, model_path, device="cuda"):
    """Main function to load model and generate answer."""
    # Load the model and tokenizer
    tokenizer, model = load_model(model_path, device)
    
    # Generate the answer from the top results
    answer = generate_answer(query, top_results, tokenizer, model, device)
    
    # Print the answer
    print(f"Answer: {answer}")

if __name__ == "__main__":
    # Example query and top results (replace with actual results)
    query = "What is the theory of relativity?"
    
    # Assuming you have the top 5 results in this format (replace with your actual top 5 results)
    top_results = [
        {"rank": 1, "score": 0.7151, "doc_id": "1110693", "title": "Theory of relativity", "snippet": "Theory of relativity The theory of relativity usually encompasses two interrelated theories by Albert Einstein: special relativity and general relativity..."},
        {"rank": 2, "score": 0.6987, "doc_id": "1110694", "title": "Theory of relativity", "snippet": "and gravitational time dilation, and length contraction. In the field of physics, relativity improved the science of elementary particles..."},
        {"rank": 3, "score": 0.6902, "doc_id": "8524661", "title": "General relativity", "snippet": "black holes, respectively. The bending of light by gravity can lead to the phenomenon of gravitational lensing..."},
        {"rank": 4, "score": 0.6835, "doc_id": "2926889", "title": "Principle of relativity", "snippet": "gravitation as an effect of the geometry of spacetime. Einstein based this new theory on the general principle of relativity..."},
        {"rank": 5, "score": 0.6804, "doc_id": "10671645", "title": "Gravity", "snippet": "the curvature of spacetime and are named after him. The Einstein field equations are a set of 10 simultaneous, non-linear, differential equations..."}
    ]
    
    # Define the model path for Qwen2.5-7B-Instruct
    model_path = "scripts/models/qwen2.5-7B-instruct"  # Adjust if needed
    
    # Device to use (cuda or cpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Run the script
    main(query, top_results, model_path, device)
