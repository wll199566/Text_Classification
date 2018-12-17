import argparse
import torch

from model import CNN
from dataloader import data_loader
import spacy
import csv
import json

N_FILTERS = 100
FILTER_SIZES = [3, 4, 5]
OUTPUT_DIM = 1
DROPOUT = 0.5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_DIM = 3002
EMBEDDING_DIM = 50
spacy_en = spacy.load('en')
amazon_root = '../data/amazon.cleaned.datasets'


def main(args):
    print("Start!")
    model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, is_train=False)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    print("Load model success!")
    vocab, _, _ = data_loader(amazon_root, load_val=False, load_train=False)
    print("Load vocab success!")
    label_list = []
    context_list = []

    model.eval()

    yelp_size = ['5', '05', '005', '0005']
    input_yelp_files = []
    output_json_files = []
    for s in yelp_size:
        input_yelp_files.append(args.input_yelp_dir + s + '.csv')
        output_json_files.append('./yelp' + s + 'vec.json')

    def get_vec(sentence, min_len=5):
        tokenized = [tok.text for tok in spacy_en.tokenizer(sentence)]
        if len(tokenized) < min_len:
            tokenized += ['<pad>'] * (min_len - len(tokenized))
        indexed = [vocab.stoi[t] for t in tokenized]
        tensor = torch.LongTensor(indexed).to(device)
        tensor = tensor.unsqueeze(1)
        with torch.no_grad():
            res = model(tensor)
        return res

    for idx in range(4):
        print("Yelp-0." + yelp_size[idx] + '%')
        print("Start transfer to vector!")
        with open(input_yelp_files[idx], 'rt', encoding='utf-8') as fin:
            csv_header = csv.reader(fin, delimiter=',')
            for i, row in enumerate(csv_header):
                label_list.append(row[0])
                context_list.append(get_vec(row[1]).cpu())

        label_list = label_list[1:]
        context_list = context_list[1:]

        "Start to write to json!"
        with open(output_json_files[idx], 'wt') as fout:
            for i, context in enumerate(context_list):
                average_vec_dict = {}
                average_vec_dict['label'] = str(label_list[i])
                average_vec_dict['avg_vec'] = context_list[i].squeeze(0).numpy().tolist()
                json.dump(average_vec_dict, fout)
                fout.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default="./Amazon10.ckpt",
                        help='Load model')
    parser.add_argument('--input_yelp_dir', type=str,
                        default="../data/yelp.cleaned.datasets/train",
                        help='')
    # parser.add_argument('--output_json_file', type=str,
    #                     default="./yelp5vec.json",
    #                     help='')
    args = parser.parse_args()
    main(args)
