import sys
import os, socket, signal, time
import json
import torch
import math
import logging
import numpy as np
from modules.data import QEDataset, collate_fn
from modules.mini_model import QEMini
from modules.model import QETransformer
from modules.minievaluator import MiniEvaluator
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

def get_evaluator(config, checkpoint_path = None):
    #get parameters from config file
    model_name = config["model_name"]
    learning_rate= config["learning_rate"]
    epochs = config["epochs"]
    batch_size = config["batch_size_per_gpu"] * args.num_gpus
    accum_grad = config["accum_grad"]
    eval_interval = config["eval_interval"]
    loss_fn = "mse" if "loss_fn" not in config else config["loss_fn"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
  
    teacher_model = QETransformer(model_name)
    student_model = QEMini(250002, 1024)

    #teacher_checkpoint_path = os.path.join(config["output_dir"], "best.pt")
    #teacher_model.load_state_dict(torch.load(teacher_checkpoint_path))
    checkpoint_path = os.path.join(config["output_dir"], "kdmini_layer.0.best.pt")
    student_model.load_state_dict(torch.load(checkpoint_path))

    test_file = config["dev"][0]["tsv_file"]
    test_dataset = QEDataset(test_file)

    test_dataloader = DataLoader(test_dataset, 
                             batch_size=batch_size, 
                             collate_fn=partial(collate_fn, tokenizer=tokenizer), 
                             shuffle=False)

    evaluator = MiniEvaluator(teacher_model, student_model, test_dataloader)
    return evaluator

evaluator = get_evaluator(config)

print(evaluator.eval())
