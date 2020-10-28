import os
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr
from tqdm import tqdm
import logging

class QEEvaluator(object):
    def __init__(self, 
                 model, 
                 checkpoint_path=None):

        #intialize model
        self.gpu=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model

        if checkpoint_path is not None:
            self.model.load_state_dict(torch.load(checkpoint_path))

        self.model = torch.nn.DataParallel(model)
        self.model = self.model.to(self.gpu)
        self.model.eval()

    def forward(self, batch):
        batch = [{k:v.to(self.gpu) for k,v in b.items()} for b in batch]

        predicted_scores, _ = self.model(batch)

        if torch.isnan(predicted_scores).any():
            return None
        else:
            return predicted_scores

    #evaluate on dev dataset, return pearson_correlation and mse
    def eval(self, test_dataloader):
        self.model.eval()
        all_predicted_scores, all_actual_scores = [], []
        with torch.no_grad():
            for batch, z_scores, da_scores in tqdm(test_dataloader):
                predicted_scores = self.forward(batch)
                predicted_scores[torch.isnan(predicted_scores)] = 0
                all_predicted_scores += predicted_scores.flatten().tolist()
                all_actual_scores += z_scores

        all_predicted_scores = np.array(all_predicted_scores)
        all_actual_scores = np.array(all_actual_scores)

        pearson_correlation = pearsonr(all_predicted_scores, all_actual_scores)[0]
        mse = np.square(np.subtract(all_predicted_scores, all_actual_scores)).mean()
        self.model.train()
        return pearson_correlation, mse
