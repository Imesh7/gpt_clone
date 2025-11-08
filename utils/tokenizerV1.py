import re

'''Tis tokenizer just created for learninig purposes.
We will us tiktoken as our toknier

More info
https://colab.research.google.com/drive/1gyRyRDrxg8EhSWSypApGVWNVdVMpdzIJ?usp=sharing'''

class TokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for i,s in vocab.items()}

    def encode(self, text):
        proc = re.split(r'\s', text)
        set_text = set(proc)
        ids = [self.str_to_int[t] for t in set_text]
        return ids
    
    def decode(self, ids):
        text = "".join([self.int_to_str[i] for i in ids])
        return text
    

# BPE tokenizer
