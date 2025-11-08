import torch
import torch.nn as nn
import tiktoken
from config.config import GptConfig
from transformer.transformer import MultiHeadAttention

class GPTModel:
    def __init__(self, cfg):
        self.token_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_em = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["dropout_rate"])
        self.transformer = nn.Sequential(
            [GptTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = GptLayerNorm(cfg["emb_size"])
        self.out_token = nn.Linear(cfg["emb_dim"], cfg["vocab_size"])

    def forward(self, x):
        batch, seq_length = x.shape
        tok_embs = self.token_emb(x)
        pos_embs = self.pos_em(torch.arange(seq_length, device=x.device))

        text_emb = tok_embs + pos_embs
        droped_emb = self.dropout(text_emb)
        tb = self.transformer(droped_emb)
        norm_token = self.final_norm(tb)
        logits = self.out_token(norm_token)
        return logits


"""Gpt Transformer Block"""

class GptTransformerBlock:
    def __init__(self, cfg):
        self.multi_head = MultiHeadAttention(
            d_in=cfg["emb_dim"], d_out=cfg["emb_dim"], num_heads=cfg["num_heads"]
        )

        self.feedforward = FeedForward(cfg)
        self.norm1 = GptLayerNorm(cfg["emb_dim"])
        self.norm2 = GptLayerNorm(cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["dropout_rate"])

    def forward(self, x):
        norm1 = self.norm1(x)
        atten = self.multi_head(norm1)
        x = self.dropout(atten)
        # If need we can do some shortcut

        norm2 = self.norm2(x)
        ff = self.feedforward(norm2)
        x = self.dropout(ff)
        # If need we can do some shortcut

        return x


"""Layer normalize"""
class GptLayerNorm(nn.Module):
    def __init__(self, emb_size):
        super.__init__()
        self.eps = 1e-5
        self.scale = nn.Embedding(emb_size)
        self.shift = nn.Embedding(emb_size)

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)

        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return norm_x * self.scale + self.shift


# tokernizer = tiktoken.get_encoding("gpt-2")
# tokernizer.encode()

# GPTModel(GptConfig)

"""GELU activation"""
class Gelu:
    def __init__(self):
        super.__init__()

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(2 / torch.pi) * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


"""Feedforward neural network"""
class FeedForward:
    def __init__(self, cfg):
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            Gelu(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


""" Example neural network with shortcut handling , 
Feedforward neural network with shortcut"""
class ExampleNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, shortcutStatus):
        self.shortcutStatus = shortcutStatus
        self.layers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), Gelu()),
                nn.Sequential(
                    nn.Linear(layer_sizes[1], layer_sizes[2]),
                    Gelu(),
                ),
                nn.Sequential(
                    nn.Linear(layer_sizes[2], layer_sizes[3]),
                    Gelu(),
                ),
                nn.Sequential(
                    nn.Linear(layer_sizes[3], layer_sizes[4]),
                    Gelu(),
                ),
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            out = layer[x]

            if self.shortcutStatus & x.shape == out.shape:
                x += out
            else:
                x
        return x


# layers = [3, 3, 3, 3, 1]
# x = torch.randn(2, 5, 768)
# model_without_shortcut = ExampleNeuralNetwork(layers, False)
# out = model_without_shortcut(x)


"""Generate Token"""

def generateToken(idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_con_token = idx[:, :context_size]
        model = GPTModel()
        logits = model(idx_con_token)
        prob = torch.softmax(logits, dim=-1)
        new_idx = torch.argmax(prob, dim=-1, keepdim=True)
        idx = torch.cat([idx, new_idx], dim=1)

    return idx


""" Temparure sampling """
""" This means we can set a value over 1+ to sharpen the response & if you need to 
diverge some more we can use less 1 (0.1) """

"""In underlaying what is happening there is assume we have probability 0.8 & 0.2 item.

- if we divide 0.2 by 0.1 if will increase more than 0.2+
- if we divide 0.2 by 2 then it will reduced to 0.1 , after that 
if we do the softmax it will become more less value.
"""


def temperature_scalling(logits, temperature):
    x = logits / temperature
    return torch.softmax(x, dim=0)


def top_k_sampling(logits, top_k):
    return torch.topk(logits, top_k)



"""If you do top-k , after that you need to make `-enf` the other logits"""

def top_k_logits_rep(logits, min_value):
    new_logits = torch.where(
        condition=logits > min_value, input=torch.tensor(float("-inf")), other=logits
    )

    return new_logits


def generateToken(model, idx,context_size, max_new_token, temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_token):
        idx_con = idx[:, -context_size:]
        logits = model(idx_con)
        if top_k is not None:
            x = top_k_sampling(logits, top_k)
            logits = top_k_logits_rep(logits, x)

        if temperature > 0.0:
            logits = temperature_scalling(logits, temperature)
            idx_next = torch.multinomial(logits, num_samples=1)

        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if idx_next == eos_id:
            break
        else:
            idx = torch.cat((idx, idx_next), dim=1)

    return idx