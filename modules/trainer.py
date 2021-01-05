import os, sys
import json
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr
from sklearn.metrics import f1_score
from tqdm import tqdm
from .utils import convert_to_class
import logging
import math

class QETrainer(object):
    def __init__(self, 
                 model, 
                 output_dir,
                 train_dataloader,
                 dev_dataloader,
                 test_dataloader = None,
                 checkpoint_prefix = '',
                 learning_rate=1e-6,
                 epochs=20,
                 eval_interval = 500,
                 task="regression",
                 find_powerbert_conf=False,
                 n_classes = 6,
                 class_weights = None,
                 config=None):

        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.test_dataloader = test_dataloader

        #intialize model
        self.gpu=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model

        #task
        self.task = task
        self.find_powerbert_conf = find_powerbert_conf
        if self.task in ["ordinal_regression", "classification"]:
            self.model.set_weight(weight=torch.tensor(class_weights))

        #l1loss weight
        self.l1loss_weight = 1e-5 if self.task == "regression" else 1e-4

        self.n_classes = n_classes

        self.model = torch.nn.DataParallel(model)
        self.model = self.model.to(self.gpu)
        self.model.train()

        #initialize optimizer
        param1 = [param for name, param in model.named_parameters() if "token_mask" not in name]
        param2 = [param for name, param in model.named_parameters() if "token_mask" in name]
        param_group = [{"params":param1},
                        {"params":param2, "lr": 1e-3}]
        self.optimizer = torch.optim.AdamW(param_group, lr=learning_rate)

        self.epochs = epochs
        self.eval_interval = eval_interval

        self.log_file = os.path.join(output_dir, "%slog"%checkpoint_prefix)
        self.best_model_path = os.path.join(output_dir, "%sbest.pt"%checkpoint_prefix)
        self.best_test_outputs = os.path.join(output_dir, "%sbest.test.output"%checkpoint_prefix)
        self.model_checkpoint_path = os.path.join(output_dir, "%scheckpoint.pt"%checkpoint_prefix)
        if self.find_powerbert_conf:
            self.best_powerbert_conf = os.path.join(output_dir, "%sbest.powerbert.conf"%checkpoint_prefix)

        #initialize training parameters
        self.cur_epoch = 0
        self.global_steps = 0
        self.best_eval_result = -10000000
        self.early_stop_count = 0

        self.init_logging()

    def init_logging(self):
        logging.basicConfig(filename=self.log_file,
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)

    def forward(self, batch, labels=None):
        batch = {k:v.to(self.gpu) for k,v in batch.items()}
        batch["labels"] = labels
        batch["task"] = self.task
        return self.model(**batch)

    #evaluate on dev dataset, return pearson_correlation and mse
    def eval(self, dataloader):
        self.model.eval()
        all_predicted_labels, all_actual_labels = [], []

        l1loss = 0
        loss = 0
        features = defaultdict(float)

        eval_meta = {}

        with torch.no_grad():
            eval_meta["mass"] = [el.token_mask.clamp(0,1).squeeze().sum().item() for el in self.model.module.encoder_layers]

            for batch, labels, meta in tqdm(dataloader):

                if self.task == "regression":
                    labels = torch.tensor(labels["da_scores"]).to(self.gpu)
                elif self.task in ["ordinal_regression", "classification"]:
                    labels = torch.tensor(labels["da_classes"]).to(self.gpu)
                else:
                    labels = torch.tensor(labels["bi_da_classes"]).to(self.gpu)

                att_mask = batch["attention_mask"].to(self.gpu)
                src_mask = meta["src_mask"].to(self.gpu)
                tgt_mask = meta["tgt_mask"].to(self.gpu)

                outputs = self.forward(batch, labels)

                predictions = outputs["qe_scores"] if self.task == "regression" else outputs["qe_cls"]
                print(predictions)

                loss += outputs["loss"].mean().item() * labels.size(0)

                predictions[torch.isnan(predictions)] = 0
                all_predicted_labels += predictions.flatten().tolist()
                all_actual_labels += labels.flatten().tolist()

                if self.find_powerbert_conf:
                    for i, encoder_output in enumerate(outputs["encoder_outputs"]):
                        token_mask = encoder_output["token_mask"]
                        l1loss += self.l1loss_weight*(i+1)*token_mask.clamp(0,1).squeeze().sum()

        all_predicted_labels = np.array(all_predicted_labels)
        all_actual_labels = np.array(all_actual_labels)

        if self.task == "regression":
            pearson_correlation = pearsonr(all_predicted_labels, all_actual_labels)[0]
            mse = np.square(np.subtract(all_predicted_labels, all_actual_labels)).mean()
            metrics = {"pc": pearson_correlation, "loss": mse}
        elif self.task in ["ordinal_regression", "classification"]:
            f1 = f1_score(all_actual_labels, all_predicted_labels, average="macro", labels=range(self.n_classes))
            metrics = {"f1": f1, "loss": loss/len(all_predicted_labels)}
        else:
            f1 = f1_score(all_actual_labels, all_predicted_labels)
            metrics = {"f1": f1, "loss": loss/len(all_predicted_labels)}

        self.model.train()

        eval_meta["l1loss"] = l1loss / all_predicted_labels.shape[0]
        eval_meta["predictions"] = all_predicted_labels

        return metrics, eval_meta

    #main train loop
    def train(self):
        for epoch in range(self.cur_epoch, self.epochs):
            print("Epoch ", epoch)
            total_loss = 0
            total_batches = 0
            for batch, labels, meta in tqdm(self.train_dataloader):
                
                if self.task == "regression":
                    labels = torch.tensor(labels["da_scores"]).to(self.gpu)
                elif self.task in ["ordinal_regression", "classification"]:
                    labels = torch.tensor(labels["da_classes"]).to(self.gpu)
                else:
                    labels = torch.tensor(labels["bi_da_classes"]).to(self.gpu)

                outputs = self.forward(batch, labels)
                predictions = outputs["qe_scores"] if self.task == "regression" else outputs["qe_cls"]
                cur_batch_size = predictions.size(0)

                loss = outputs["loss"].mean()

                if torch.isnan(predictions).any():
                    continue

                if self.find_powerbert_conf:
                    l1loss = 0
                    for i, encoder_output in enumerate(outputs["encoder_outputs"]):
                        token_mask = encoder_output["token_mask"]
                        l1loss += self.l1loss_weight*(i+1)*token_mask.clamp(0,1).squeeze().sum()
                    loss += l1loss/cur_batch_size

                total_batches += cur_batch_size
                total_loss += loss.item() * cur_batch_size

                loss.backward()
                self.optimizer.step()
                self.model.zero_grad()

                avg_loss = total_loss/total_batches
                log = "Epoch %s Global steps: %s Train loss: %.4f\n" %(epoch, self.global_steps, avg_loss)

                self.global_steps += 1
                if (self.global_steps) % self.eval_interval == 0:

                    if self.test_dataloader is not None:
                        metrics, eval_meta_test = self.eval(self.test_dataloader)
                        m = metrics["pc"] if self.task == "regression" else metrics["f1"]
                        
                        log += "test loss: %.4f l1loss: %.15f, r:%.4f\n" % (metrics["loss"], eval_meta_test["l1loss"], m)
                  
                    metrics, eval_meta_dev = self.eval(self.dev_dataloader)
                    m = metrics["pc"] if self.task == "regression" else metrics["f1"]

                    if self.find_powerbert_conf:
                        m = -(metrics["loss"] + eval_meta_dev["l1loss"])

                    log += "Dev loss: %.4f l1loss: %.15f, r:%.4f\n" % (metrics["loss"], eval_meta_dev["l1loss"], m)
                    log += "Token_masks: " + ",".join(["%s"%math.ceil(mass) for mass in eval_meta_dev["mass"]]) + "\n"

                    if m > self.best_eval_result:
                        self.best_eval_result = m
                        #torch.save(self.model.module.state_dict(), self.best_model_path)

                        if self.find_powerbert_conf:
                            with open(self.best_powerbert_conf, "w") as fout:
                                prev = 1000
                                for mass in eval_meta_dev["mass"]:
                                    mass = math.ceil(mass)
                                    mass = min(mass, prev)
                                    prev = mass
                                    print(mass, file=fout)
                        else:
                            with open(self.best_test_outputs, "w") as fout:
                                for pred in eval_meta_test["predictions"]:
                                    print(pred, file=fout)

                        self.early_stop_count = 0
                    else:
                        self.early_stop_count += 1

                    logging.info(log)

                    if self.early_stop_count > 50:
                        return 

            #save checkpoint
            #torch.save(self, self.model_checkpoint_path)
