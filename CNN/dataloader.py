import torch
from torchtext import data
import spacy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Vocab(torch.utils.data.Dataset):
    def __init__(self, root="./data/yelp.cleaned.datasets", max_size=3000, load_val=True):
        self.spacy_en = spacy.load('en')
        self.TEXT = data.Field(tokenize=self.tokenizer)
        self.LABEL = data.Field(sequential=False)
        tv_dataFields = [("label", self.LABEL), ("text", self.TEXT)]

        self.load_val = load_val
        if self.load_val:
            self.trn, self.val = data.TabularDataset.splits(path=root, train='train.csv',
                                                            validation="valid.csv", format='csv',
                                                            skip_header=True, fields=tv_dataFields)
        else:
            self.trn, _ = data.TabularDataset.splits(path=root, train='train.csv',
                                                     format='csv',
                                                     skip_header=True, fields=tv_dataFields)
            self.val = None

        # Print to test
        # print("init Vocab success!")
        # print("val[0].__dict__: ", self.val[0].__dict__['label'])
        # print("val[0].['text']: ", self.val[0].__dict__['text'][:10])

        self.TEXT.build_vocab(self.trn, max_size=max_size, vectors="glove.6B.50d")
        self.LABEL.build_vocab(self.trn, max_size=5)

    def __getitem__(self, index):
        return self.trn[index]  # , self.val[index]

    def __len__(self):
        return len(self.trn)  # + len(self.val)

    def tokenizer(self, text):
        token = [t.text for t in self.spacy_en.tokenizer(text)]
        if len(token) < 5:
            for i in range(0, 5 - len(token)):
                token.append('<PAD>')
        return token


def data_loader(root="./data/yelp.cleaned.datasets", max_size=3000, load_val=True, load_train=True):
    vocab = Vocab(root, max_size, load_val=True)
    # train_iterator, valid_iterator = data.BucketIterator.splits(
    #     (vocab.trn, vocab.val),
    #     batch_size=batch_size,
    #     device=device,
    #     sort_key=lambda x: len(x.text),
    #     # BucketIterator 依据什么对数据分组
    #     sort_within_batch=True)

    if load_val and load_train:
        return vocab.TEXT.vocab, vocab.trn, vocab.val
    elif load_train:
        return vocab.TEXT.vocab, vocab.trn, None
    else:
        return vocab.TEXT.vocab, None, None
