import sys
import os, socket, signal, time
import json
import torch
import math
import logging
import numpy as np
from modules.data import QEDataset, collate_fn
from modules.model import QETransformer
from modules.evaluator import QEEvaluator
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

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    use_layers = config.get("use_layers", None)
    model = QETransformer(model_name, use_layers=use_layers)

    if checkpoint_path is None:
        checkpoint_path = os.path.join(config["output_dir"], "best.pt")

    model.load_state_dict(torch.load(checkpoint_path))

    evaluator = QEEvaluator(model, checkpoint_path)
    return evaluator, tokenizer

evaluator, tokenizer = get_evaluator(config)

avg_pc = 0
avg_rmse = 0
for ld in ["en-de", "en-zh", "ro-en", "et-en", "si-en", "ne-en", "ru-en"]:
    test_file = "data/%s/test20.%s.df.short.tsv" % (ld, ld.replace("-",""))
    test_dataset = QEDataset(test_file)
    test_dataloader = DataLoader(test_dataset, 
                             batch_size=config["batch_size_per_gpu"] * args.num_gpus,
                             collate_fn=partial(collate_fn, tokenizer=tokenizer), 
                             shuffle=False)
    pc, rmse = evaluator.eval(test_dataloader)
    print("%s\t%.4f\t%.4f"%(ld, pc, rmse))
    avg_pc += pc
    avg_rmse += rmse
print("Avg\t%.4f\t%.4f"%(avg_pc/7, avg_rmse/7))
