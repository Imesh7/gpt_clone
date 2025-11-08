import torch.nn as nn
from config.config import GptConfig
from data.dataloader import create_dataloder, load_text
from data.tokenizer import Tokenizer
from model import GPTModel, generateToken


class GPTv1(nn.Module):
    def __init__(self):
        self.tokenizer = Tokenizer(encoding_name="gpt2")
        text = load_text()
        # encode will don inside the data loader
        dataloder = create_dataloder(
            tokenizer=self.tokenizer,
            text=text,
            context_length=GptConfig["context_length"],
            stride=GptConfig["stride"],
            batch_size=GptConfig["batch_size"],
        )
        model = GPTModel(cfg=GptConfig)
        idx = generateToken(
            model=model,
            idx=dataloder.dataset,
            context_size=GptConfig["context_size"],
            max_new_token=10,
        )

    def forward():
        pass
