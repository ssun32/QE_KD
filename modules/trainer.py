import os, sys
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr
from tqdm import tqdm
import logging

class QETrainer(object):
    def __init__(self, 
                 model, 
                 output_dir,
                 train_dataloader,
                 dev_dataloader,
                 test_loader = None,
                 checkpoint_prefix = '',
                 learning_rate=1e-6,
                 epochs=20,
                 eval_interval = 500):

        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader

        #intialize model
        self.gpu=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.model = torch.nn.DataParallel(model)
        self.model = self.model.to(self.gpu)
        self.model.train()

        #initialize optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        self.epochs = epochs
        self.eval_interval = eval_interval

        self.log_file = os.path.join(output_dir, "%slog"%checkpoint_prefix)
        self.best_model_path = os.path.join(output_dir, "%sbest.pt"%checkpoint_prefix)
        self.model_checkpoint_path = os.path.join(output_dir, "%scheckpoint.pt"%checkpoint_prefix)

        #initialize training parameters
        self.cur_epoch = 0
        self.global_steps = 0
        self.best_eval_result = 0
        self.early_stop_count = 0

        self.init_logging()

    def init_logging(self):
        logging.basicConfig(filename=self.log_file,
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)

    def forward(self, batch, actual_scores=None):
        batch = [{k:v.to(self.gpu) for k,v in b.items()} for b in batch]

        predicted_scores, _ = self.model(batch)

        if torch.isnan(predicted_scores).any():
            return None, None
        else:
            return predicted_scores, _

    #evaluate on dev dataset, return pearson_correlation and mse
    def eval(self):
        self.model.eval()
        all_predicted_scores, all_actual_scores = [], []
        with torch.no_grad():
            for batch, z_scores, da_scores in tqdm(self.dev_dataloader):
                predicted_scores,_ = self.forward(batch)
                predicted_scores[torch.isnan(predicted_scores)] = 0
                all_predicted_scores += predicted_scores.flatten().tolist()
                all_actual_scores += z_scores

        all_predicted_scores = np.array(all_predicted_scores)
        all_actual_scores = np.array(all_actual_scores)

        pearson_correlation = pearsonr(all_predicted_scores, all_actual_scores)[0]
        mse = np.square(np.subtract(all_predicted_scores, all_actual_scores)).mean()
        self.model.train()
        return pearson_correlation, mse

    #main train loop
    def train(self):
        for epoch in range(self.cur_epoch, self.epochs):
            print("Epoch ", epoch)
            total_loss = 0
            total_batches = 0
            for batch, z_scores, da_scores in tqdm(self.train_dataloader):
                z_scores = torch.tensor(z_scores).to(self.gpu)
                predicted_scores, transformer_outputs = self.forward(batch)

                att_mask = batch[0]["attention_mask"].unsqueeze(1)

                if predicted_scores is None:
                    continue

                loss = torch.nn.MSELoss()(predicted_scores.squeeze(), z_scores)

                cur_batch_size = predicted_scores.size(0)
                total_batches += cur_batch_size
                total_loss += loss.item() * cur_batch_size

                loss.backward()
                self.optimizer.step()
                self.model.zero_grad()

                avg_loss = total_loss/total_batches
                log = "Epoch %s Global steps: %s Train loss: %.4f\n" %(epoch, self.global_steps, avg_loss)

                self.global_steps += 1
                if self.global_steps % self.eval_interval == 0:
                   
                    pearson_correlation, mse = self.eval()
                    log += "Dev loss: %.4f r:%.4f\n" % (mse, pearson_correlation)
                    if pearson_correlation > self.best_eval_result:
                        self.best_eval_result = pearson_correlation
                        torch.save(self.model.module.state_dict(), self.best_model_path)

                        self.early_stop_count = 0
                    else:
                        self.early_stop_count += 1

                    logging.info(log)

                    if self.early_stop_count > 100:
                        return 

            #save checkpoint
            torch.save(self, self.model_checkpoint_path)

