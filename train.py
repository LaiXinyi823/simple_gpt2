import torch
from src.gpt2 import GPT2Embedding, MaskedSelfAttention
from src.config import GPT2Config

batch_size = 4

# 定义GPT2配置
config = GPT2Config(vocab_size=1000, embedding_size=768, max_seq_len=128, hidden_size=768)

# 自定义测试数据集
data = torch.randint(0, 1000, (batch_size, config.max_seq_len))

# 测试embedding
embedding = GPT2Embedding(config)
embedded_data = embedding(data)

# 测试多头自注意力机制
multi_head_attention = MaskedSelfAttention(config)
attentioned_data = multi_head_attention(embedded_data)

print(data.shape, multi_head_attention.shape)