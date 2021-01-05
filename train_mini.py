import sys
import os, socket, signal, time
import json
import torch
import math
import logging
import numpy as np
from modules.data import QEDataset, collate_fn
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

def get_trainer(config):
    #get parameters from config file
    model_name = config["model_name"]
    learning_rate= config["learning_rate"]
    epochs = config["epochs"]
    batch_size = config["batch_size_per_gpu"] * args.num_gpus
    accum_grad = config["accum_grad"]
    eval_interval = config["eval_interval"]
    loss_fn = "mse" if "loss_fn" not in config else config["loss_fn"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    powerbert_conf = os.path.join(config["output_dir"], "powerbert.conf")
    if os.path.exists(powerbert_conf):
        powerbert_cuts = [int(l) for l in open(powerbert_conf)]
    else:
        powerbert_cuts = None
    checkpoint_path = config["checkpoint_path"]
    model = QETransformer(checkpoint_path=checkpoint_path, powerbert_cuts = powerbert_cuts)
    model.train()

    #for name, param in model.named_parameters():
    #    if "token_mask" not in name:
    #        param.requires_grad = False
    #    print(name, param.shape, param.requires_grad)


    train_file = config["train"][0]["tsv_file"]
    train_dataset = QEDataset(train_file)
    dev_file = config["dev"][0]["tsv_file"]
    dev_dataset = QEDataset(dev_file)
    test_file = config["test"][0]["tsv_file"]
    test_dataset = QEDataset(test_file)

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

    trainer = QETrainer(model, 
                        config["output_dir"], 
                        train_dataloader, 
                        dev_dataloader, 
                        test_dataloader = test_dataloader,
                        checkpoint_prefix = "mini_",
                        learning_rate=learning_rate,
                        epochs=epochs,
                        eval_interval=eval_interval)
    return trainer

#initialize trainer
checkpoint = os.path.join(config["output_dir"], "mini_checkpoint.pt")
"""
if os.path.exists(checkpoint):
    try:
        print("checkpoint found, restarting from checkpoint...")
        trainer = torch.load(checkpoint)
        trainer.init_logging()
    except:
        print("Failed to load checkpoint...")
        trainer = get_trainer(config)
else:
    trainer = get_trainer(config)
"""
trainer = get_trainer(config)
trainer.train()
