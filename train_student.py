import sys
import os, socket, signal, time
import json
import torch
import math
import logging
import numpy as np
from modules.data import QEDataset, TransferDataset, collate_fn
from modules.model import QETransformer
from modules.mini_model import QEMini
from modules.kd_trainer import KDTrainer
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

    teacher_model = QETransformer(model_name)
    checkpoint_path = os.path.join(config["output_dir"], "best.pt")
    teacher_model.load_state_dict(torch.load(checkpoint_path))

    vocab_size = teacher_model.transformer.get_input_embeddings().weight.shape[0]

    #student_model = QETransformer("xlm-roberta-base")
    print(config["n_layers"])
    student_model = QEMini(vocab_size, layers=config["n_layers"])

    train_id = config["train"][0]["id"]
    train_file = config["train"][0]["tsv_file"]
    train_dataset = QEDataset(train_file)
    dev_file = config["dev"][0]["tsv_file"]
    dev_dataset = QEDataset(dev_file)
    test_file = config["test"][0]["tsv_file"]
    test_dataset = QEDataset(test_file)

    transfer_dataset = train_file[0].split("/")
    transfer_dataset[-1] = "transfer_set.tsv"
    if train_id == "all":
        transfer_dataset[-2] = "all"

    transfer_dataset = TransferDataset("/".join(transfer_dataset))

    train_dataloader = DataLoader(train_dataset, 
                             batch_size=batch_size, 
                             collate_fn=partial(collate_fn, tokenizer=tokenizer), 
                             shuffle=True)

    dev_dataloader = DataLoader(dev_dataset, 
                             batch_size=batch_size, 
                             collate_fn=partial(collate_fn, tokenizer=tokenizer), 
                             shuffle=False)

    test_dataloader = DataLoader(dev_dataset, 
                             batch_size=batch_size, 
                             collate_fn=partial(collate_fn, tokenizer=tokenizer), 
                             shuffle=False)

    transfer_dataloader = DataLoader(transfer_dataset, 
                             batch_size=batch_size, 
                             collate_fn=partial(collate_fn, tokenizer=tokenizer), 
                             shuffle=True)

    trainer = KDTrainer(teacher_model, 
                        student_model,
                        config["output_dir"], 
                        train_dataloader, 
                        dev_dataloader, 
                        test_dataloader,
                        transfer_dataloader,
                        copy_embeddings=config["copy_embeddings"],
                        encoder_copy_mapping=config["encoder_copy_mapping"],
                        encoder_sup_mapping=config["encoder_sup_mapping"],
                        copy_mlp=config["copy_mlp"],
                        use_transferset=config["use_transferset"],
                        checkpoint_prefix=config["checkpoint_prefix"],
                        learning_rate=learning_rate,
                        epochs=epochs,
                        eval_interval=eval_interval)
    return trainer

#initialize trainer
checkpoint = os.path.join(config["output_dir"], "%scheckpoint.pt"%(config["checkpoint_prefix"]))
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

#trainer.multi_stage_train()
#trainer.unfreeze_parameters("embeddings")
#trainer.unfreeze_parameters("mlp")
trainer.train("layer.%s"%(int(config["train_layer"])))
with open(os.path.join(config["output_dir"], "%soutput.txt"%config["checkpoint_prefix"]), "w") as fout:
    pc, mse = trainer.test()
    print("%.6f %.6f"%(pc, mse), file=fout)
