import sys
import os, socket, signal, time
import json
import torch
import math
import logging
import numpy as np
from modules.data import QEDataset, collate_fn
from modules.model import QETransformer
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

    use_layers = config.get("use_layers", None)
    model = QETransformer(model_name, use_layers = use_layers)

    print(model.transformer.config.hidden_act)
    for name, params in model.transformer.encoder.layer[0].named_parameters():
        print(name, params.shape)
    sys.exit(0)
    if "checkpoint_path" in config and config["checkpoint_path"] is not None:
        model.load_state_dict(torch.load(config["checkpoint_path"]))

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
    trainer = QETrainer(model, config["output_dir"], train_dataloader, dev_dataloader, 
                        learning_rate=learning_rate,
                        epochs=epochs,
                        eval_interval=eval_interval)
    return trainer

#initialize trainer
checkpoint = os.path.join(config["output_dir"], "checkpoint.pt")
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

trainer.train()
