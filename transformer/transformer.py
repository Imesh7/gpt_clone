import torch.nn as nn
import torch


class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        self.queries = nn.Linear(d_in, d_out)
        self.keys = nn.Linear(d_in, d_out)
        self.values = nn.Linear(d_in, d_out)
        self.dropOut = nn.Dropout(dropout)

    def forward(self, x):
        batch, num_tokens, d_in = x
        query = self.queries(x)
        key = self.keys(x)
        value = self.values(x)

        attention_score = query @ key.T  # `T` means Transpose

        # atten_w = nn.Softmax(attention_score / self.key.shape[-1] ** 0.5, dim=-1)
        mask = torch.tril(torch.ones(num_tokens, num_tokens))
        masked_atten_we = attention_score.masked_fill(mask.bool(), -torch.inf)

        masked_norm = torch.softmax(masked_atten_we / key.shape[-1] ** 0.5, dim=1)
        atten_wei = self.dropOut(masked_norm)
        context_vec = atten_wei @ value
        return context_vec


batch = torch.stack((input))  # we can use just the torch.randn to generate input


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads):
        context_length = batch.shape[1]  # num of token it can process
        self.heads = nn.ModuleList(
            [
                CausalAttention(d_in, d_out, context_length, 0.0, False)
                for _ in range(num_heads)
            ]
        )

    def forward(self, x):
       # loop through the each head , concat the each head result
       # dim =-1 means get last dimention.
       # In here last dim is 2. beacuse d_out is 2.
       # so if we concat it num_head * d_out
       return torch.cat([head(x) for head in self.heads], dim=-1)