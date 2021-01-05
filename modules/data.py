import torch
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from .utils import convert_to_class
from tqdm import tqdm

def collate_fn(batches, tokenizer):
    batch_text = []
    batch_z_scores = []
    batch_da_scores = []
    batch_class = []
    batch_bi_class = []

    for batch in batches:
        if batch["target"] is not None:
            batch_text.append((batch["source"], batch["target"]))
        else:
            batch_text.append(batch["source"])

        batch_z_scores.append(batch["z_score"])
        batch_da_scores.append(batch["da_score"])

        batch_class.append(batch["da_class"])
        batch_bi_class.append(batch["bi_da_class"])

    tokenized = tokenizer(batch_text, 
                           truncation="longest_first", 
                           max_length=128, 
                           padding=True,
                           pad_to_max_length=False, 
                           return_special_tokens_mask=True,
                           return_tensors = "pt")

    input_ids = tokenized["input_ids"]
    meta = {"text":[], 
            "src_mask":torch.zeros(input_ids.size()),
            "tgt_mask":torch.zeros(input_ids.size())}
    for i, t in enumerate(input_ids):
        meta["text"].append(tokenizer.convert_ids_to_tokens(t.tolist()))

    labels = {"z_scores": batch_z_scores, 
              "da_scores": batch_da_scores,
              "da_classes": batch_class,
              "bi_da_classes": batch_bi_class}

    return tokenized, labels, meta

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
                           "da_class": convert_to_class(mean_score, "qe"),
                           "bi_da_class": convert_to_class(mean_score, "qe", binary=True),
                           "z_score": zmean_score,
                           "task": "qe"})

            self.dataset += rows

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def get_class_weights(self):
        counts = [0] * 6
        for row in self.dataset:
            counts[row["da_class"]] += 1

        class_weights = [1/count for count in counts]
        return class_weights

    def get_bi_class_weights(self):
        counts = [0] * 2
        for row in self.dataset:
            counts[row["bi_da_class"]] += 1

        class_weights = [1/count for count in counts]
        return class_weights


class TransferDataset(Dataset):
    def __init__(self, filepath):

        self.rows = []
        for i, l in enumerate(open(filepath)):
            items = l.strip().split("\t")

            self.rows.append({
                   "source": items[0],
                   "target": items[0],
                   "da_score": 0,
                   "da_class": 0,
                   "z_score": 0,
                   "task": "qe"})

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

class BlogDataset(Dataset):
    def __init__(self, filepath):
        if type(filepath) == type("str"):
            filepath = [filepath]

        self.dataset = []
        for fp in filepath:
            rows = []

            for i, l in enumerate(open(fp)):
                age, source = l.strip().split("\t")
                age = float(age)/100

                rows.append({
                       "source": source,
                       "target": None,
                       "da_score": age,
                       "da_class": convert_to_class(age, "blog"),
                       "bi_da_class": convert_to_class(age, "blog", binary=True),
                       "z_score": 0.0,
                       "task": "blog"})

            self.dataset += rows

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def get_class_weights(self):
        counts = [0] * 3
        for row in self.dataset:
            counts[row["da_class"]] += 1

        class_weights = [1/count for count in counts]
        return class_weights

    def get_bi_class_weights(self):
        counts = [0] * 2
        for row in self.dataset:
            counts[row["bi_da_class"]] += 1

        class_weights = [1/count for count in counts]
        return class_weights

