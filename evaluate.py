import sys
import os, socket, signal, time
import json
import torch
import math
import logging
import numpy as np
from modules.data import QEDataset, collate_fn
from modules.qe_transformer import QETransformer
from modules.evaluator import Evaluator
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from functools import partial
from scipy.stats import pearsonr
from glob import glob
from tqdm import tqdm
import argparse

#arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config_file', required=True)
parser.add_argument('--num_gpus', type=int, default=1)
parser.add_argument('--prune_layer', type=int, default=-1)
parser.add_argument('--prune_head', type=int, default=-1)

args = parser.parse_args()
print(args)

with open(args.config_file) as fjson:
    config = json.load(fjson)

def get_evaluator(config):
    #get parameters from config file
    model_name = config["model_name"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    #checkpoint_path = os.path.join(config["output_dir"], "model.pt")
    checkpoint_path = os.path.join(config["output_dir"], "mini_best.pt")

    powerbert_conf = os.path.join(config["output_dir"], "powerbert.conf")
    if os.path.exists(powerbert_conf):
        powerbert_cuts = [int(l) for l in open(powerbert_conf)]
    else:
        powerbert_cuts = None

    model = QETransformer(config=config, 
                          intermediate_size = 4096,
                          checkpoint_path=checkpoint_path,
                          powerbert_cuts = powerbert_cuts)

    evaluator = Evaluator(model)
    return evaluator, tokenizer

evaluator, tokenizer = get_evaluator(config)


dataloaders = []
for ld in ["en-de", "en-zh", "ro-en", "et-en", "si-en", "ne-en", "ru-en"]:
    test_file = "data/%s/test20.%s.df.short.tsv" % (ld, ld.replace("-",""))
    test_dataset = QEDataset(test_file)
    test_dataloader = DataLoader(test_dataset, 
                         batch_size=config["batch_size_per_gpu"] * args.num_gpus,
                         collate_fn=partial(collate_fn, tokenizer=tokenizer), 
                         shuffle=False)
    dataloaders.append((ld, test_dataloader))

#if args.prune_layer != -1:
#    prune_dict = {args.prune_layer:[args.prune_head]}
#else:
#    prune_dict = None

avg_pc = 0
avg_rmse = 0
for ld, dataloader in dataloaders:
    pc, rmse = evaluator.eval(dataloader)
    print("%s\t%.4f\t%.4f"%(ld, pc, rmse))
    avg_pc += pc
    avg_rmse += rmse
print("Avg\t%.4f\t%.4f\n"%(avg_pc/7, avg_rmse/7))
