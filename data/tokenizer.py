import tiktoken

class Tokenizer:
    def __init__(self, encoding_name):
        self.tokenizer = tiktoken.get_encoding(encoding_name=encoding_name)

    def encode(self, text):
        return self.tokenizer.encode(text=text)

    def decode(self, input):
        return self.tokenizer.decode(input)