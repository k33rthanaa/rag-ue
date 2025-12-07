#!/usr/bin/env python3
"""Test Qwen 7B with INT4 quantization on 11GB GPU"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

print("=" * 80)
print("🧪 Testing Qwen 7B on 11GB GPU with INT4 Quantization")
print("=" * 80)

# Step 1: Check GPU
print("\n1️⃣ Checking GPU availability...")
if not torch.cuda.is_available():
    print("❌ No GPU available!")
    exit(1)

print(f"✅ GPU found: {torch.cuda.get_device_name(0)}")
print(f"   Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Step 2: Load tokenizer
print("\n2️⃣ Loading tokenizer...")
model_path = "scripts/models/qwen2.5-7B-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
print("✅ Tokenizer loaded")

# Step 3: Configure INT4 quantization
print("\n3️⃣ Configuring INT4 quantization...")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # Use 4-bit quantization
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,       # Extra compression
    bnb_4bit_quant_type="nf4"             # Optimal for LLMs
)
print("✅ Quantization config ready")

# Step 4: Load model with max_memory to control allocation
print("\n4️⃣ Loading Qwen 7B in INT4 (this takes ~2 minutes)...")
print("   Expected memory usage: ~4GB")

max_memory = {0: "10GB", "cpu": "30GB"}  # Force CPU offloading if needed

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quantization_config,
    device_map="auto",
    max_memory=max_memory,
    low_cpu_mem_usage=True
)

print("✅ Model loaded successfully!")

# Step 5: Check memory usage
memory_used = torch.cuda.memory_allocated(0) / 1e9
print(f"\n5️⃣ GPU Memory Usage: {memory_used:.2f} GB / 11.55 GB")

if memory_used > 10:
    print("⚠️  Warning: Using >10GB, close to limit!")
else:
    print("✅ Memory usage looks good!")

# Step 6: Test generation
print("\n6️⃣ Testing generation...")
test_prompt = "Tell me a brief bio of Albert Einstein in 2 sentences."

inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")

print("   Generating answer...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n" + "=" * 80)
print("📝 GENERATED ANSWER:")
print("=" * 80)
print(answer)
print("=" * 80)

# Final memory check
final_memory = torch.cuda.memory_allocated(0) / 1e9
print(f"\n✅ Test complete! Final GPU memory: {final_memory:.2f} GB")

if final_memory < 11:
    print("✅ SUCCESS: Model fits in 11GB GPU!")
else:
    print("❌ FAILED: Model exceeds 11GB limit")
