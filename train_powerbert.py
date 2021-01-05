import sys
import os, socket, signal, time
import json
import torch
import math
import logging
import numpy as np
from modules.data import QEDataset, BlogDataset, collate_fn
from modules.qe_transformer import QETransformer
from modules.trainer import QETrainer
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from functools import partial
from scipy.stats import pearsonr
from glob import glob
from tqdm import tqdm
import argparse

# define the handler function
# note that this is not executed here, but rather
# when the associated signal is sent
def sig_handler(signum, frame):
    print("caught signal", signum)
    print(socket.gethostname(), "USR1 signal caught.")
    # do other stuff to cleanup here
    print('requeuing job ' + os.environ['SLURM_JOB_ID'])
    os.system('scontrol requeue ' + os.environ['SLURM_JOB_ID'])
    sys.exit(-1)

def term_handler(signum, frame):
    print("bypassing sigterm", flush=True)

signal.signal(signal.SIGUSR1, sig_handler)
signal.signal(signal.SIGTERM, term_handler)
print('signal installed', flush=True)

#arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config_file', required=True)
parser.add_argument('--num_gpus', type=int, default=1)
args = parser.parse_args()
print(args)

with open(args.config_file) as fjson:
    config = json.load(fjson)
print(config)

def get_trainer(config, find_powerbert_conf=False, load_powerbert_cut=False):
    #get parameters from config file
    model_name = config["model_name"]
    learning_rate= config["learning_rate"]
    epochs = config["epochs"]
    batch_size = config["batch_size_per_gpu"] * args.num_gpus
    eval_interval = config["eval_interval"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    use_layers = config.get("use_layers", None)
    if config["task"] == "bi_classification":
        n_classes = 2
    else:
        n_classes = 6 if config["dataset"] == "qe" else 3

    if load_powerbert_cut:
        powerbert_conf = os.path.join(config["output_dir"], "best.powerbert.conf")
        powerbert_cuts = [int(l) for l in open(powerbert_conf)]
        print(powerbert_cuts)
    else:
        powerbert_cuts = None

    model = QETransformer(
                hidden_size = config["hidden_size"],
                intermediate_size = config["intermediate_size"],
                n_heads = config["n_heads"],
                n_layers = config["n_layers"],
                max_layer = config["max_layer"],
                checkpoint_path = config["checkpoint_path"],
                model_name = config["model_name"],
                n_classes = n_classes,
                find_powerbert_conf = find_powerbert_conf,
                powerbert_cuts=powerbert_cuts)

    Dataset = QEDataset if config["dataset"] == "qe" else BlogDataset
                        
    train_file = config["train"][0]["tsv_file"]
    train_dataset = Dataset(train_file)
    dev_file = config["dev"][0]["tsv_file"]
    dev_dataset = Dataset(dev_file)
    test_file = config["test"][0]["tsv_file"]
    test_dataset = Dataset(test_file)

    task = config["task"] if "task" in config else "regression"

    train_dataloader = DataLoader(train_dataset, 
                             batch_size=batch_size, 
                             collate_fn=partial(collate_fn, tokenizer=tokenizer), 
                             shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, 
                             batch_size=batch_size, 
                             collate_fn=partial(collate_fn, tokenizer=tokenizer), 
                             shuffle=False)
    test_dataloader = DataLoader(test_dataset, 
                             batch_size=batch_size, 
                             collate_fn=partial(collate_fn, tokenizer=tokenizer), 
                             shuffle=False)

    if task == 'bi_classification':
        class_weights = train_dataset.get_bi_class_weights()
    else:      
        class_weights = train_dataset.get_class_weights()
    print("class weights:", class_weights)

    trainer = QETrainer(model, config["output_dir"], 
                        train_dataloader, 
                        dev_dataloader, 
                        test_dataloader,
                        learning_rate=learning_rate,
                        epochs=epochs,
                        eval_interval=eval_interval,
                        task=task,
                        find_powerbert_conf=find_powerbert_conf,
                        n_classes = n_classes,
                        class_weights = class_weights,
                        config=config)
    return trainer

#initialize trainer
trainer = get_trainer(config, find_powerbert_conf=True)
trainer.train()
trainer = get_trainer(config, load_powerbert_cut=True)
trainer.train()
