from modeling_nini import NiniForCausalLM, NiniConfig, get_nini_tokenizer
import torch

tokenizer = get_nini_tokenizer()
config = NiniConfig(vocab_size=len(tokenizer), hidden_size=768, num_hidden_layers=16, intermediate_size=4096, num_attention_heads=8)
model = NiniForCausalLM(config)

total_params = sum(p.numel() for p in model.parameters())
print(total_params)
for p, q in model.named_parameters():
    print(p, q.shape)
text = "The quick brown fox jumps over the lazy dog."
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs, labels=inputs["input_ids"])
print("loss:", outputs["loss"])
