import os, sys
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from scipy.stats import pearsonr
from tqdm import tqdm
import logging

class KDTrainer(object):
    def __init__(self, 
                 kd_model, 
                 output_dir,
                 dataloaders,
                 samplers,
                 checkpoint_prefix = "kd_",
                 use_transferset=False,
                 rank=0,
                 world_size=1,
                 learning_rate=1e-6,
                 epochs=20,
                 eval_interval = 500):

        self.dataloaders = dataloaders
        self.samplers = samplers

        self.use_transferset=use_transferset
        self.rank = rank
        self.world_size = world_size

        self.kd_model = kd_model

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.eval_interval = eval_interval

        self.log_file = os.path.join(output_dir, "%slog"%checkpoint_prefix)
        self.best_model_path = os.path.join(output_dir, "%sbest.pt"%checkpoint_prefix)
        self.model_checkpoint_path = os.path.join(output_dir, "%scheckpoint.pt"%checkpoint_prefix)

        #initialize training parameters
        self.cur_epoch = 0
        self.global_steps = 0
        self.best_eval_result = -100000
        self.early_stop_count = 0

        self.init_logging()

    def init_logging(self):
        logging.basicConfig(filename=self.log_file,
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)


    #evaluate on dev dataset, return pearson_correlation and mse
    def eval(self):

        self.kd_model.student_model.eval()
        total_loss = 0
        total_batches = 0
        all_predicted_scores, all_actual_scores = [], []
        with torch.no_grad():

            dataloader = self.dataloaders["dev"]
            if self.rank == 0:
                dataloader = tqdm(dataloader)

            for batch, z_scores, da_scores, meta in dataloader:
                batch = {k: v.to(self.rank) for k,v in batch.items()}
                z_scores = torch.tensor(z_scores).to(self.rank)

                batch_size = batch["input_ids"].size(0)

                kd_outputs = self.kd_model(**batch)
                s_predicted_outputs = kd_outputs["student_output"]
                s_predicted_outputs[torch.isnan(s_predicted_outputs)] = 0

                loss = kd_outputs["loss"].mean()
                total_loss += loss.item() * s_predicted_outputs.size(0)
                total_batches += s_predicted_outputs.size(0)

                all_predicted_scores.append(s_predicted_outputs)
                all_actual_scores.append(z_scores)

            all_predicted_scores = torch.cat(all_predicted_scores)
            all_actual_scores = torch.cat(all_actual_scores)

        all_predicted_scores_list = [torch.ones_like(all_predicted_scores) for _ in range(self.world_size)]
        all_actual_scores_list = [torch.ones_like(all_predicted_scores) for _ in range(self.world_size)]
        dist.all_gather(all_predicted_scores_list, all_predicted_scores)
        dist.all_gather(all_actual_scores_list, all_actual_scores)

        predicted_scores = torch.cat(all_predicted_scores_list)
        actual_scores = torch.cat(all_actual_scores_list)

        all_predicted_scores = predicted_scores.tolist()
        all_actual_scores = actual_scores.tolist()

        pearson_correlation = pearsonr(all_predicted_scores, all_actual_scores)[0]
        mse = np.square(np.subtract(all_predicted_scores, all_actual_scores)).mean()
        self.kd_model.student_model.train()

        return pearson_correlation, mse, total_loss/total_batches


    #main train loop
    def train(self):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(
            backend='nccl',
	    init_method='env://',
	    world_size=self.world_size,
	    rank=self.rank)  

        torch.cuda.set_device(self.rank)
        kd_model = self.kd_model.to(self.rank)
        kd_model = torch.nn.parallel.DistributedDataParallel(kd_model, device_ids=[self.rank])
        optimizer = torch.optim.AdamW(kd_model.parameters(), lr=self.learning_rate)

        #init
        if self.use_transferset:
            dataloader = self.dataloaders["transfer"]
        else:
            dataloader = self.dataloaders["train"]

        for epoch in range(self.cur_epoch, self.epochs):
            self.samplers["train"].set_epoch(epoch)

            if self.rank == 0:
                print("Epoch ", epoch)

            total_loss = 0
            total_output_loss = 0
            total_batches = 0

            if self.rank == 0:
                dataloader = tqdm(dataloader)

            for batch, z_scores, da_scores, meta in dataloader:
                batch_size = batch["input_ids"].size(0)
                batch = {k: v.to(self.rank) for k,v in batch.items()}
                batch["scores"] = torch.tensor(z_scores).to(self.rank)

                kd_outputs = self.kd_model(**batch)

                #if torch.isnan(kd_outputs["student_output"]).any():
                #    continue

                loss = kd_outputs["loss"].mean()
                loss.backward()
                optimizer.step()
                kd_model.zero_grad()

                total_batches += batch_size
                total_loss += loss.item() * batch_size

                del kd_outputs, loss
                
                self.global_steps += 1
                if self.global_steps % self.eval_interval == 0:
                    avg_loss = total_loss/total_batches
                    total_loss = 0 
                    total_batches = 0
                    log = "Epoch %s Global steps: %s Train loss: %.4f\n" %(epoch, self.global_steps, avg_loss)
                   
                    pearson_correlation, mse, dev_loss = self.eval()
                    dist.barrier()

                    log += "Dev loss: %.4f mse: %.4f r:%.4f\n" % (dev_loss, mse, pearson_correlation)

                    if self.rank == 0:
                        logging.info(log)

                    if pearson_correlation > self.best_eval_result:
                        self.best_eval_result = pearson_correlation
                        if self.rank == 0:
                            torch.save(self.kd_model.student_model.state_dict(), self.best_model_path)
                        self.early_stop_count = 0
                    else:
                        self.early_stop_count += 1

                    if self.early_stop_count > 2000:
                        return 

            #save checkpoint
            self.cur_epoch += 1
            #torch.save(self, self.model_checkpoint_path)
