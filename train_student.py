import json
import torch
import math
import logging
import numpy as np
import os, socket, signal, time
from modules.data import QEDataset, TransferDataset, collate_fn
from modules.qe_transformer import QETransformer
from modules.kd_model import KDModel
from modules.kd_trainer import KDTrainer
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from transformers import AutoTokenizer, AutoModel
from functools import partial
from scipy.stats import pearsonr
from glob import glob
from tqdm import tqdm
import argparse

def sig_handler(signum, frame):
    print("caught signal", signum)
    print(socket.gethostname(), "USR1 signal caught.")
    print('requeuing job ' + os.environ['SLURM_JOB_ID'])
    os.system('scontrol requeue ' + os.environ['SLURM_JOB_ID'])
    sys.exit(-1)

def term_handler(signum, frame):
    print("bypassing sigterm", flush=True)

def train(rank, world_size, kd_model, tokenizer, args, config):
    learning_rate= config["learning_rate"]
    epochs = config["epochs"]
    batch_size = config["batch_size_per_gpu"]

    eval_interval = config["eval_interval"]

    train_id = config["train"][0]["id"]

    datasets = {}
    dataloaders = {}
    samplers = {}
    for split in ["train", "dev", "test"]:
        datasets[split] = QEDataset(config[split][0]["tsv_file"])

        samplers[split] = torch.utils.data.distributed.DistributedSampler(
                            datasets[split],
                            num_replicas=world_size,
                            shuffle=split in ["train", "transfer"],
                            rank=rank)

        dataloaders[split] = DataLoader(datasets[split],
                                        batch_size = batch_size,
                                        collate_fn=partial(collate_fn, tokenizer=tokenizer),
                                        shuffle=False,
                                        pin_memory=True,
                                        sampler=samplers[split])

    trainer = KDTrainer(kd_model, 
                        config["output_dir"], 
                        dataloaders,
                        samplers,
                        use_transferset=False,
                        checkpoint_prefix=config["checkpoint_prefix"],
                        learning_rate=learning_rate,
                        epochs=epochs,
                        eval_interval=eval_interval,
                        rank=rank,
                        world_size=world_size)

    trainer.train()

if __name__ == '__main__':

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

    model_name = config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    #teacher model
    checkpoint_path = config["checkpoint_path"]
    teacher_model = QETransformer(checkpoint_path=checkpoint_path)

    #student model
    student_model = QETransformer(intermediate_size = 1024)

    kd_model = KDModel(teacher_model, 
                       student_model,
                       encoder_sup_mapping=config["encoder_sup_mapping"],
                       encoder_copy_mapping=config["encoder_copy_mapping"])

    world_size = torch.cuda.device_count()
    print(world_size)
    mp.spawn(train, nprocs=world_size, args=(world_size, kd_model, tokenizer, args, config), join=True)
