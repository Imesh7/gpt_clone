import torch.nn as nn
import tiktoken

from dataloader import create_dataloder


class GPTv1(nn.Module):
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("gpt2")
        create_dataloder()


    def forward():
        pass