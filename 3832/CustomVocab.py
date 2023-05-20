class CustomVocab:
    def __init__(self, counter):
        self.itos = [token for token, _ in counter.most_common()]
        self.stoi = {token: i for i, token in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)