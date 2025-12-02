from transformers import AutoModel, AutoTokenizer

def download_contriever_model():
    model_name = "facebook/contriever"  # You can also use "facebook/contriever-msmarco"
    
    # Download the model and tokenizer
    print(f"Downloading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Save the model locally
    model.save_pretrained("./models/contriever")
    tokenizer.save_pretrained("./models/contriever")

    print(f"Model and tokenizer have been saved to './models/contriever'")

if __name__ == "__main__":
    download_contriever_model()
