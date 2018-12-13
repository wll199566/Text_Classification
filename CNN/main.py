import argparse
import torch
from torchtext import data

from model import CNN
from dataloader import data_loader

N_FILTERS = 100
FILTER_SIZES = [3, 4, 5]
OUTPUT_DIM = 1
DROPOUT = 0.5
BATCH_SIZE = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        #         _, res = model(batch.text)
        res = model(batch.text)
        predictions = res.squeeze(1)
        #         print(predictions, type(predictions))
        #         print(batch.label, type(batch.label))
        #         break

        loss = criterion(predictions, batch.label)
        # loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            #             _, res = model(batch.text)
            res = model(batch.text)
            predictions = res.squeeze(1)

            loss = criterion(predictions, batch.label)

            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = rounded_preds == y  # convert into float for division
    acc = correct.sum().float() / len(correct)
    return acc


def main(args):
    print("Begin to load data!")
    EMBEDDING_DIM = args.glove_size
    vocab, train_loader, val_loader = data_loader(args.root, args.glove_size, args.max_size)
    print("Loading Finished")
    INPUT_DIM = len(vocab)
    pretrained_embeddings = vocab.vectors

    model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)
    model.embedding.weight.data.copy_(pretrained_embeddings)

    train_iterator, valid_iterator = data.BucketIterator.splits(
        (vocab.trn, vocab.val),
        batch_size=BATCH_SIZE,
        device=device,
        sort_key=lambda x: len(x.text),
        # BucketIterator 依据什么对数据分组
        sort_within_batch=True)

    N_EPOCHS = 10
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.BCEWithLogitsLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    print("Start training!")
    for epoch in range(N_EPOCHS):
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
        print(
            f'| Epoch: {epoch + 1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc * 100:.2f}% |')
        filepath = 'Yelp-{}.ckpt'.format(epoch + 1)
        torch.save(model.state_dict(), filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--glove_size', type=int,
                        default='50',
                        help='glove vector size')
    parser.add_argument('--root', type=str,
                        default='../data/yelp.cleaned.datasets',
                        help='directory of dataset')
    parser.add_argument('--max_size', type=int,
                        default='3000',
                        help='maximum size of sentences')
    args = parser.parse_args()
    main(args)
