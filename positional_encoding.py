import numpy as np

class PositionalEncoding:
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len


    def forward(self):
        i = np.arange(0, self.d_model, 2)
        demonimatoe = np.power(10000, i / self.d_model)
        position = np.arange(self.max_seq_len)[:, np.newaxis]
        even_pos_encoding = np.sin(position / demonimatoe)
        odd_pos_encoding = np.cos(position / demonimatoe)
        stacked = np.stack((even_pos_encoding, odd_pos_encoding), axis = 2)
        pos_encoding = stacked.reshape(self.max_seq_len, self.d_model)
        return pos_encoding


pe = PositionalEncoding(d_model=6, max_seq_len=10)





