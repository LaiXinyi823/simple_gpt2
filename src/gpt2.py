import math
import torch
import torch.nn as nn

class GPT2Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.embedding_size)
        self.position_embeddings = nn.Embedding(config.max_seq_len, config.embedding_size)
        self.max_seq_len = config.max_seq_len
    
    def forward(self, input_ids):
        position_ids = torch.arange(self.max_seq_len)
        position_ids = position_ids.expand_as(input_ids)
        return self.token_embeddings(input_ids) + self.position_embeddings(position_ids)

class MaskedSelfAttention(nn.Module):
    def __init__(self, config):
        self.W_q = nn.Linear(config.embedding_size, config.hidden_size)
        self.W_k = nn.Linear(config.embedding_size, config.hidden_size)
        self.W_v = nn.Linear(config.embedding_size, config.hidden_size)

    def forward(self, x):
        q = self.transpose_qkv(self.W_q(x))
        k = self.transpose_qkv(self.W_k(x))
        v = self.transpose_qkv(self.W_v(x))
        output = self.transpose_output(self.DotProductAttention(q, k, v))
        return output
    
    def DotProductAttention(self, q, k, v):
        # q, k, v :(b_s, seq_len, hidden_size)
        d_k = k.shape[-1]
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(d_k)
        attention_weights = nn.Softmax(scores)
        return torch.bmm(attention_weights, v)
    
    def transpose_qkv(self, x, num_heads):
        x = x.reshape(x.shape[0], x.shape[1], num_heads, -1)
        x = x.permute(0, 2, 1, 3)
        return x.reshape(-1, x.shape[2], x.shape[3])
    
    def transpose_output(self, x, num_heads):
        x = x.reshape(-1, num_heads, x.shape[1], x.shape[2])
        x = x.permute(0, 2, 1, 3)
        return x.reshape(x.shape[0], x.shape[1], -1)


# class FFN(nn.Module):

# class GPT2Layer(nn.Module):

# class GPT2Model(nn.Module):
