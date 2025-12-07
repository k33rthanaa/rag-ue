#!/usr/bin/env python3
"""Test Qwen 7B with INT8 quantization on 11GB GPU"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("=" * 80)
print("🧪 Testing Qwen 7B on 11GB GPU with INT8 Quantization")
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
model_path = "scripts/models/qwen2.5-7B-instruct"  # Your local path
tokenizer = AutoTokenizer.from_pretrained(model_path)
print("✅ Tokenizer loaded")

# Step 3: Load model in INT8
print("\n3️⃣ Loading Qwen 7B in INT8 (this takes ~2 minutes)...")
print("   Expected memory usage: ~7GB")

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    load_in_8bit=True,         # INT8 quantization
    device_map="auto",          # Automatic device placement
    torch_dtype=torch.float16
)

print("✅ Model loaded successfully!")

# Step 4: Check memory usage
memory_used = torch.cuda.memory_allocated(0) / 1e9
print(f"\n4️⃣ GPU Memory Usage: {memory_used:.2f} GB / 11.26 GB")

if memory_used > 10:
    print("⚠️  Warning: Using >10GB, close to limit!")
else:
    print("✅ Memory usage looks good!")

# Step 5: Test generation
print("\n5️⃣ Testing generation...")
test_prompt = "Tell me a brief bio of Albert Einstein in 2 sentences."

inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")

print("   Generating answer...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=250,
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
