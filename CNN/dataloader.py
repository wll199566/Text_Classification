import torch
from torchtext import data
from torchtext import datasets

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Vocab(datasets):
    def __init__(self, root="./data/yelp.cleaned.datasets", glove_size=50, max_size=3000):
        self.TEXT = data.Field(tokenize='spacy')
        self.LABEL = data.Field(sequential=False, use_vocab=False, dtype=torch.float)
        tv_dataFields = [("label", self.LABEL), ("text", self.TEXT)]
        self.trn, self.val = data.TabularDataset.splits(path=root, train='train.csv',
                                                        validation="valid.csv", format='csv',
                                                        skip_header=True, fields=tv_dataFields)

        # Print to test
        print("init Vocab success!")
        print("val[0].__dict__: ", self.val[0].__dict__['label'])
        print("val[0].['text']: ", self.val[0].__dict__['text'][:10])

        glove = "glove.6B." + glove_size + "d"
        self.TEXT.build_vocab(self.trn, max_size=max_size, vectors=glove)

    def __getitem__(self, index):
        return self.trn[index], self.val[index]

    def __len__(self):
        return len(self.trn) + len(self.val)


def data_loader(root="./data/yelp.cleaned.datasets", glove_size=50, max_size=3000):
    vocab = Vocab(root, glove_size, max_size)
    # train_iterator, valid_iterator = data.BucketIterator.splits(
    #     (vocab.trn, vocab.val),
    #     batch_size=batch_size,
    #     device=device,
    #     sort_key=lambda x: len(x.text),
    #     # BucketIterator 依据什么对数据分组
    #     sort_within_batch=True)

    return vocab.TEXT.vocab, vocab.trn, vocab.val
