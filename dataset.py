from glob import glob
import torch
from torch.utils.data import Dataset
from utils import read_txt, save_txt


def split_crime_n_punishment_file(data_path="data/crime-and-punishment.txt",
                                  save_path="data/docs"):
    doc = ['\n']
    doc_idx = 0
    line_breaker_count = 0
    with open(data_path) as f:
        for line in f:
            if line == doc[-1] == "\n":
                line_breaker_count += 1
            else:
                line_breaker_count = 0
                if line != "\n":
                    doc.append(line.strip())
                else:
                    doc.append("\n")

            if line_breaker_count == 3:
                doc = "".join(doc[1:])
                if doc.startswith("PART"):
                    save_txt(f"{save_path}/{doc_idx}.txt", doc)
                    doc_idx += 1
                doc = ['\n']


class CustomDataset(Dataset):
    def __init__(self, hp, root, tokenizer):
        self.max_len = hp.max_position_embeddings
        self.file_list = sorted(glob(f"{root}/*.txt"))
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        text = read_txt(file_path).replace('\n', "<n>")
        x = self.tokenizer.encode(text)

        doc_len = len(x)
        pad_len = (self.max_len//2) - doc_len % (self.max_len // 2)

        x += [self.tokenizer.pad_token_id]*pad_len + [self.tokenizer.pad_token_id]*(self.max_len//2)
        x = torch.LongTensor(x)
        x = x.view(-1, self.max_len // 2)
        x = torch.cat([x[:-1], x[1:]], dim=1)
        return x


if __name__ == "__main__":
    from en_tokenizer import CharTokenizer
    from utils import read_json, make_dot_dict

    # split_crime_n_punishment_file()
    hp = read_json('config.json')
    hp = make_dot_dict(hp)

    tokenizer = CharTokenizer(model_path='sentencepiece_models/cp-char.model')
    dataset = CustomDataset(hp=hp, root='data/docs', tokenizer=tokenizer)
    for d in dataset:
        print(d)
        print(d.size())

        tmp = d.tolist()
        for line in tmp[:10]:
            print(tokenizer.decode(line))

        break

