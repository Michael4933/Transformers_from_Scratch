import torch
from transformers import AutoTokenizer
from modeling_nini import NiniForCausalLM, NiniConfig, get_nini_tokenizer

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 加载 tokenizer 和模型配置
    tokenizer = get_nini_tokenizer()
    config = NiniConfig(
        vocab_size=len(tokenizer),
        hidden_size=768,
        num_hidden_layers=16,
        num_attention_heads=8,
        intermediate_size=4096,
        dropout=0.1,
    )

    # 2. 加载模型
    model = NiniForCausalLM(config).to(device)
    ckpt_path = "/user/liyou/results/checkpoint_epoch3.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # 3. 输入 prompt
    prompt = "人工智能是"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # 4. 生成文本
    output_text = model.generate(input_ids, tokenizer, max_new_tokens=50, temperature=10.0, top_k=20)
    print("=== Generated Text ===")
    print(output_text)


if __name__ == "__main__":
    main()
