from torchtext.data import Iterator
from torchtext.data import TabularDataset
import argparse
import torch
from torchtext import data
import spacy
from model import CNN

N_FILTERS = 100
FILTER_SIZES = [3, 4, 5]
OUTPUT_DIM = 1
DROPOUT = 0.5
BATCH_SIZE = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_DIM = 3002
EMBEDDING_DIM = 50
spacy_en = spacy.load('en')


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = rounded_preds == y  # convert into float for division
    acc = correct.sum().float() / len(correct)
    return acc


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            #             _, res = model(batch.text)
            batch = batch.to(device)
            res = model(batch.text)
            predictions = res.squeeze(1)

            loss = criterion(predictions, batch.label)

            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def tokenizer(text):
    token = [t.text for t in spacy_en.tokenizer(text)]
    if len(token) < 5:
        for i in range(0, 5 - len(token)):
            token.append('<PAD>')
    return token


def main(args):
    if args.is_yelp:
        root = '../data/yelp.cleaned.datasets'
    else:
        root = '../data/amazon.cleaned.datasets'

    model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    criterion = torch.nn.BCEWithLogitsLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    TEXT = data.Field(tokenize=tokenizer)
    LABEL = data.Field(sequential=False, use_vocab=False, dtype=torch.float)
    tst_datafields = [("label", LABEL), ("text", TEXT)]
    tst = TabularDataset(
        path=root + "/test.csv",  # 文件路径
        format='csv',
        skip_header=True,  # 如果你的csv有表头, 确保这个表头不会作为数据处理
        fields=tst_datafields)
    test_iter = Iterator(tst, batch_size=BATCH_SIZE,
                         device=device,
                         sort=False,
                         sort_within_batch=False,
                         repeat=False)
    test_loss, test_acc = evaluate(model, test_iter, criterion)
    print("Yelp? ", args.is_yelp)
    print(f'Test. Loss: {test_loss:.3f} | Test. Acc: {test_acc * 100:.2f}% ')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default="./Yelp-10.ckpt",
                        help='Test model')
    parser.add_argument('--is_yelp', type=bool,
                        default=True,
                        # default='../data/yelp.cleaned.datasets',
                        help='directory of dataset')

    args = parser.parse_args()
    main(args)
