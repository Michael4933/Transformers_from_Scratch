# filename: mini_qwen.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
import math

# ========== 1. Load Qwen3 Tokenizer ==========
def get_nini_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("/user/liyou/opensource_models/Qwen2.5-3B")  # 或 Qwen/Qwen2.5
    return tokenizer

# ========== 2. Basic Config ==========
class NiniConfig:
    def __init__(
        self,
        vocab_size,
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=2048,
        max_position_embeddings=1024,
        dropout=0.1,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.dropout = dropout


# ========== 3. Core Components ==========

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, V)
        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.d_k = hidden_size // num_heads
        self.num_heads = num_heads

        self.W_Q = nn.Linear(hidden_size, hidden_size)
        self.W_K = nn.Linear(hidden_size, hidden_size)
        self.W_V = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, hidden_size)

        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x, mask=None):
        residual = x
        B, L, H = x.shape
        q = self.W_Q(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_K(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_V(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)

        out, attn = self.attention(q, k, v, mask)
        out = out.transpose(1, 2).contiguous().view(B, L, H)
        out = self.fc(out)
        out = self.dropout(out)
        return self.norm(out + residual)


class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.linear2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        residual = x
        x = self.linear2(F.gelu(self.linear1(x)))
        x = self.dropout(x)
        return self.norm(x + residual)


# ========== 4. Positional Encoding ==========
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


# ========== 5. Decoder Block ==========
class MiniDecoderLayer(nn.Module):
    def __init__(self, config: NiniConfig):
        super().__init__()
        self.self_attn = MultiHeadAttention(config.hidden_size, config.num_attention_heads, config.dropout)
        self.ffn = FeedForward(config.hidden_size, config.intermediate_size, config.dropout)

    def forward(self, x, mask=None):
        x = self.self_attn(x, mask)
        x = self.ffn(x)
        return x


# ========== 6. Nini Model ==========
class NiniModel(nn.Module):
    def __init__(self, config: NiniConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embed = PositionalEncoding(config.hidden_size, config.max_position_embeddings)
        self.layers = nn.ModuleList([MiniDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(self, input_ids, attention_mask=None):
        x = self.embed_tokens(input_ids)
        x = self.pos_embed(x)
        for layer in self.layers:
            x = layer(x, attention_mask)
        return self.norm(x)


# ========== 7. Causal LM Head ==========
class NiniForCausalLM(nn.Module):
    def __init__(self, config: NiniConfig):
        super().__init__()
        self.model = NiniModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, labels=None, attention_mask=None):
        hidden_states = self.model(input_ids, attention_mask)
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return {"loss": loss, "logits": logits}
    
    @torch.no_grad()
    def generate(self, input_ids, tokenizer, max_new_tokens=50, temperature=1.0, top_k=20):
        """
        简单自回归文本生成 (greedy / top-k sampling)
        """
        self.eval()
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)

        for _ in range(max_new_tokens):
            outputs = self(input_ids)
            logits = outputs["logits"][:, -1, :] / temperature

            # Top-k 采样
            if top_k > 0:
                topk_logits, topk_indices = torch.topk(logits, top_k)
                probs = F.softmax(topk_logits, dim=-1)
                next_token = topk_indices[0, torch.multinomial(probs, 1)]
            else:
                next_token = torch.argmax(logits, dim=-1)[0]

            input_ids = torch.cat([input_ids, next_token], dim=-1)

            if next_token.item() == tokenizer.eos_token_id:
                break

        return tokenizer.decode(input_ids[0], skip_special_tokens=True)

