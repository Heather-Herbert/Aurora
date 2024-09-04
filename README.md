# Aurora

from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model_name = "EleutherAI/gpt-neo-1.3B"  # or another model of your choice
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Encode input text
input_text = "Hello, how are you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# Generate response
output = model.generate(input_ids, max_length=50)
response = tokenizer.decode(output[0], skip_special_tokens=True)

print(response)