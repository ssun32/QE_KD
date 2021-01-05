import sys
import time
import os, socket, signal, time
import json
import torch
import math
import logging
import numpy as np
from modules.data import QEDataset, collate_fn
from modules.mini_model import QEMini
from modules.qe_transformer import QETransformer
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from functools import partial
from scipy.stats import pearsonr
from glob import glob
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', required=True)
parser.add_argument('--num_gpus', type=int, default=1)
args = parser.parse_args()
print(args)

with open(args.config_file) as fjson:
    config = json.load(fjson)

model_name = config["model_name"]
learning_rate= config["learning_rate"]
epochs = config["epochs"]
batch_size = config["batch_size_per_gpu"] * args.num_gpus
accum_grad = config["accum_grad"]
eval_interval = config["eval_interval"]
loss_fn = "mse" if "loss_fn" not in config else config["loss_fn"]

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = QETransformer(hidden_size=768, intermediate_size=768*4,n_heads=12,n_layers=12, model_name="xlm-roberta-base")

test_file = config["dev"][0]["tsv_file"]
test_dataset = QEDataset(test_file)

test_dataloader = DataLoader(test_dataset, 
                         batch_size=1, 
                         collate_fn=partial(collate_fn, tokenizer=tokenizer), 
                         shuffle=False)

total_init_time = 0
total_embed_time = 0
total_encoder_time = 0
total_mlp_time = 0
total = 0
for batch, _, _ in test_dataloader:
    with torch.no_grad():

        #repeat 10 times
        init_time = 0
        embed_time = 0
        encoder_time = 0
        mlp_time = 0

        for _ in range(10):
            start = time.time()
            #embedding layer
            embed = model.embeddings(input_ids=batch["input_ids"])
            
            embed_time += time.time() - start

            start = time.time()
            #encoder layer
            hs = embed
            att_mask = batch["attention_mask"]
            for encoder_layer in model.encoder_layers:
                encoder_output = encoder_layer(hs, att_mask)
            encoder_time += time.time() - start

            start = time.time()
            #mlp layer

            final_cls_token = hs[:, 0, :]
            model.reg_head(model.mlp(final_cls_token))
            mlp_time += time.time() - start
        total_init_time += init_time/10
        total_embed_time += embed_time/10
        total_encoder_time += encoder_time/10
        total_mlp_time += mlp_time/10
        total += 1

print(total)
print(total_init_time/total)
print(total_embed_time/total)
print(total_encoder_time/total)
print(total_mlp_time/total)
