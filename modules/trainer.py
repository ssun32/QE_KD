import os, sys
import json
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

        l1loss = 0

        features = defaultdict(float)

        with torch.no_grad():
            for batch, z_scores, da_scores, meta in tqdm(self.dev_dataloader):

                att_mask = batch[0]["attention_mask"].to(self.gpu)
                src_mask = meta["src_mask"].to(self.gpu)
                tgt_mask = meta["tgt_mask"].to(self.gpu)

                predicted_scores, transformer_outputs = self.forward(batch)
                predicted_scores[torch.isnan(predicted_scores)] = 0
                all_predicted_scores += predicted_scores.flatten().tolist()
                all_actual_scores += z_scores

                """
                for l in range(24):
                    for h in range(16):

                        k = (l, h)
                        #get attention from layer l and head h
                        att = transformer_outputs["attentions"][l][:,h,:]

                        #avg diagonal prob
                        avg_prob_diag = ((att.diagonal(dim1=-2, dim2=-1) * att_mask).sum(dim=-1)) / att_mask.sum(dim=-1)
                        offset=-1
                        avg_prob_diag_lower1 = ((att.diagonal(offset=offset, dim1=-2, dim2=-1) * att_mask[:, -offset:]).sum(dim=-1)) / att_mask[:, -offset:].sum(dim=-1)
                        offset=1
                        avg_prob_diag_upper1 = ((att.diagonal(offset=offset, dim1=-2, dim2=-1) * att_mask[:, :-offset]).sum(dim=-1)) / att_mask[:, :-offset].sum(dim=-1)
                        avg_prob = ((att*att_mask.unsqueeze(-1)).sum(dim=-2)) / att_mask.sum(dim=-1, keepdim=True)
                        avg_prob_entropy = ((-avg_prob*(avg_prob+1e-9).log())*att_mask).sum(dim=-1)

                        #entropy
                        entropy = (-att*(att+1e-9).log()*att_mask.unsqueeze(1)).sum(dim=-1)
                        avg_entropy = ((entropy*att_mask).sum(dim=-1))/att_mask.sum(dim=-1)

                        #(src, src)
                        src_src_att = ((att * src_mask.unsqueeze(1)).sum(dim=-1) * src_mask).sum(dim=-1)/src_mask.sum(dim=-1)
                        src_tgt_att = ((att * tgt_mask.unsqueeze(1)).sum(dim=-1) * src_mask).sum(dim=-1)/src_mask.sum(dim=-1)
                        tgt_src_att = ((att * src_mask.unsqueeze(1)).sum(dim=-1) * tgt_mask).sum(dim=-1)/tgt_mask.sum(dim=-1)
                        tgt_tgt_att = ((att * tgt_mask.unsqueeze(1)).sum(dim=-1) * tgt_mask).sum(dim=-1)/tgt_mask.sum(dim=-1)

                        src_prob = src_src_att/src_tgt_att
                        tgt_prob = tgt_tgt_att/tgt_src_att

                        features[(l, h, "avg_prob_diag")] += avg_prob_diag.sum().item()
                        features[(l, h, "avg_prob_diag_lower1")] += avg_prob_diag_lower1.sum().item()
                        features[(l, h, "avg_prob_diag_upper1")] += avg_prob_diag_upper1.sum().item()
                        features[(l, h, "avg_prob_entropy")] += avg_prob_entropy.sum().item()
                        features[(l, h, "avg_entropy")] += avg_entropy.sum().item()
                        features[(l, h, "src_prob")] += src_prob.sum().item()
                        features[(l, h, "tgt_prob")] += tgt_prob.sum().item()

                if False and "token_masks" in transformer_outputs:
                    l1loss += (1/(24*128))*torch.stack([torch.norm(token_mask.squeeze(), p=1, dim=-1).mean() for token_mask in transformer_outputs["token_masks"]]).sum().item() * predicted_scores.shape[0]

        dout = defaultdict(list)
        for l in range(24):
            for h in range(16):
                print(l, h)

                print("Avg_prob_diag %.4f"%(features[l, h, "avg_prob_diag"]/1000))
                print("Avg_prob_diag_lower1 %.4f"%(features[l, h, "avg_prob_diag_lower1"]/1000))
                print("Avg_prob_diag_upper1 %.4f"%(features[l, h, "avg_prob_diag_upper1"]/1000))
                print("Avg_prob_entropy %.4f"%(features[l, h, "avg_prob_entropy"]/1000))
                print("Avg_entropy %.4f"%(features[l, h, "avg_entropy"]/1000))
                print("src_probs %.4f"%(features[l, h, "src_prob"]/1000))
                print("tgt_probs %.4f"%(features[l, h, "tgt_prob"]/1000))

        sys.exit(0)
        """


        #with open("prunelist.json", "w") as fout:
        #    print(json.dumps(dout), file=fout)

        all_predicted_scores = np.array(all_predicted_scores)
        all_actual_scores = np.array(all_actual_scores)

        pearson_correlation = pearsonr(all_predicted_scores, all_actual_scores)[0]
        mse = np.square(np.subtract(all_predicted_scores, all_actual_scores)).mean()
        self.model.train()
        return pearson_correlation, mse, l1loss / all_predicted_scores.shape[0]

    #main train loop
    def train(self):
        for epoch in range(self.cur_epoch, self.epochs):
            print("Epoch ", epoch)
            total_loss = 0
            total_l1loss = 0
            total_batches = 0
            for batch, z_scores, da_scores, meta in tqdm(self.train_dataloader):
                z_scores = torch.tensor(z_scores).to(self.gpu)
                predicted_scores, transformer_outputs = self.forward(batch)

                att_mask = batch[0]["attention_mask"].unsqueeze(1)

                if predicted_scores is None:
                    continue

                loss = torch.nn.MSELoss()(predicted_scores.squeeze(), z_scores)

                if "token_masks" in transformer_outputs:
                    l1loss = (1/(24*128))*torch.stack([torch.norm(token_mask.squeeze(), p=1, dim=-1).mean() for token_mask in transformer_outputs["token_masks"]]).sum()

                #loss += l1loss

                cur_batch_size = predicted_scores.size(0)
                total_batches += cur_batch_size
                total_loss += loss.item() * cur_batch_size

                loss.backward()
                self.optimizer.step()
                self.model.zero_grad()

                avg_loss = total_loss/total_batches
                log = "Epoch %s Global steps: %s Train loss: %.4f\n" %(epoch, self.global_steps, avg_loss)

                self.global_steps += 1
                if (self.global_steps-1) % self.eval_interval == 0:
                   
                    pearson_correlation, mse, l1loss = self.eval()
                    log += "Dev loss: %.4f l1loss: %.15f, r:%.4f\n" % (mse, l1loss, pearson_correlation)
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

