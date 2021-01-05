import os
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr
from tqdm import tqdm
import logging

class Evaluator(object):
    def __init__(self, model):

        #intialize model
        self.gpu=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model

        self.model = torch.nn.DataParallel(model)
        self.model = self.model.to(self.gpu)
        self.model.eval()

    def forward(self, batch, prune_dict={}):
        batch = [{k:v.to(self.gpu) for k,v in b.items()} for b in batch]

        predicted_scores, transformer_outputs = self.model(batch, prune_dict=prune_dict)

        if torch.isnan(predicted_scores).any():
            return None, None
        else:
            return predicted_scores, transformer_outputs

    #evaluate on dev dataset, return pearson_correlation and mse
    def eval(self, test_dataloader, prune_dict={}):

        all_predicted_scores, all_actual_scores = [], []
        with torch.no_grad():
            for batch, z_scores, da_scores, meta in tqdm(test_dataloader):
                predicted_scores,_ = self.forward(batch, prune_dict)
                predicted_scores[torch.isnan(predicted_scores)] = 0
                all_predicted_scores += predicted_scores.flatten().tolist()
                all_actual_scores += z_scores

        all_predicted_scores = np.array(all_predicted_scores)
        all_actual_scores = np.array(all_actual_scores)

        pearson_correlation = pearsonr(all_predicted_scores, all_actual_scores)[0]
        mse = np.square(np.subtract(all_predicted_scores, all_actual_scores)).mean()
        return pearson_correlation, mse
