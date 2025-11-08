import torch
from torch.utils.data import DataSet, DataLoader
import tiktoken


class GPTDataSet(DataSet):
    def __init__(
        self, tokenizer, text, context_length, stride
    ):  #  In here `stride` gap between 2 window
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text)

        for i in range(0, len(token_ids) - context_length, stride):
            input_chunk = token_ids[i : i + context_length]
            target_chunk = token_ids[i : i + context_length + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def _len_(self):
        return len(self.input_ids)

    def _get_item(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloder(tokenizer, text, context_length, stride, batch_size):

    dataset = GPTDataSet(
        tokenizer=tokenizer, text=text, context_length=context_length, stride=stride
    )
    dataLoader = DataLoader(
        dataset=dataset, batch_size=batch_size, drop_last=True, shuffle=True
    )

    return dataLoader

def load_text():
    with open('../the-verdict.txt', encoding='utf-8') as f:
       text = f.read()
    return text