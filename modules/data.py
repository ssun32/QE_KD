import torch
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def collate_fn(batches, tokenizer):
    batch_text = []
    mts = []
    wps = []
    batch_z_scores = []
    batch_da_scores = []

    for batch in batches:
        batch_text.append((batch["source"], batch["target"]))
        batch_z_scores.append(batch["z_score"])
        batch_da_scores.append(batch["da_score"])

    tokenized = [tokenizer(batch_text, 
                           truncation="longest_first", 
                           max_length=128, 
                           padding=True,
                           pad_to_max_length=False, 
                           return_special_tokens_mask=True,
                           return_tensors = "pt")]

    input_ids = tokenized[0]["input_ids"]
    meta = {"text":[], 
            "src_mask":torch.zeros(input_ids.size()),
            "tgt_mask":torch.zeros(input_ids.size())}
    for i, t in enumerate(input_ids):
        meta["text"].append(tokenizer.convert_ids_to_tokens(t.tolist()))

        end_token = 0
        for j, id in enumerate(t):
            #change to tgt if encounter the first </s>
            if id == 2:
                end_token += 1

            if end_token == 3:
                meta["tgt_mask"][i, j] = 1
                break
            elif end_token == 2:
                meta["tgt_mask"][i, j] = 1
            else:
                meta["src_mask"][i, j] = 1


    return tokenized, batch_z_scores, batch_da_scores, meta

class QEDataset(Dataset):
    def __init__(self, filepath):
        if type(filepath) == type("str"):
            filepath = [filepath]

        self.dataset = []
        for fp in filepath:
            rows = []

            for i, l in enumerate(open(fp)):
                if i == 0:
                    header = {h:j for j,h in enumerate(l.strip().split("\t"))}
                else:
                    items = l.strip().split("\t")
                    mean_score = None if "mean" not in header else float(items[header["mean"]])/100
                    zmean_score = None if "z_mean" not in header else float(items[header["z_mean"]])

                    rows.append({
                           "source": items[header["original"]],
                           "target": items[header["translation"]],
                           "da_score": mean_score,
                           "z_score": zmean_score})

            self.dataset += rows

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

class TransferDataset(Dataset):
    def __init__(self, filepath):

        self.rows = []
        for i, l in enumerate(open(filepath)):
            items = l.strip().split("\t")

            self.rows.append({
                   "source": items[0],
                   "target": items[0],
                   "da_score": 0.0,
                   "z_score": 0.0})

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]
