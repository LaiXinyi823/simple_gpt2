class GPT2Config():
    def __init__(self,
                 vocab_size,
                 embedding_size,
                 max_seq_len,
                 hidden_size):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size