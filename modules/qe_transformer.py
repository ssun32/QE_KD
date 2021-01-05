import sys
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import XLMRobertaConfig, XLMRobertaModel, AutoModel
from modules.transformer import EncoderLayer
from modules.model import QETransformer
from .spacecutter import OrdinalLogisticModel, CumulativeLinkLoss

class QETransformer(nn.Module):

    def __init__(
            self, 
            vocab_size = 250002, 
            hidden_size = 1024, 
            intermediate_size = 4096,
            n_heads = 16,
            n_layers = 24,
            max_layer = 23,
            n_classes = 6,
            allowed_layer = None,
            checkpoint_path = None,
            powerbert_cuts = None,
            find_powerbert_conf = False,
            model_name = "xlm-roberta-large",
            config = None):
        super().__init__()

        self.head_dim = int(hidden_size/n_heads)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_layer = max_layer
        self.config = config
        self.powerbert_cuts = powerbert_cuts
        self.find_powerbert_conf = find_powerbert_conf

        self.embeddings = AutoModel.from_pretrained(model_name).embeddings

        #init encoder layers
        self.encoder_layers = nn.ModuleList(
                                [EncoderLayer(
                                    self.n_heads, 
                                    self.hidden_size, 
                                    self.intermediate_size,
                                    self.head_dim, 
                                    self.head_dim) for _ in range(self.n_layers)])

        self.mlp = nn.Sequential(
                        nn.Linear(hidden_size, 4*hidden_size),
                        nn.ReLU(),
                        nn.Dropout(0.1))

        self.reg_head = nn.Linear(4*hidden_size, 1)
        self.cls_head = nn.Sequential(
                            nn.Linear(4*hidden_size, n_classes))
        self.ordinal_reg_head = OrdinalLogisticModel(n_classes = n_classes)

        self.cls_softmax = nn.Softmax(dim=-1)

        self.reg_loss = torch.nn.MSELoss()
        self.ordinal_reg_loss = CumulativeLinkLoss()
        self.cls_loss = torch.nn.CrossEntropyLoss()

        #load from checkpoint
        if checkpoint_path is not None:
            print("Loading from checkpoint...")
            checkpoint_dict = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
            model_dict = self.state_dict()

            new_state_dict = {k:v for k, v in checkpoint_dict.items() if v.shape == model_dict[k].shape}
            
            self.load_state_dict(new_state_dict, strict=False)

    def set_weight(self, weight):
        weight /= weight.sum()
        self.cls_loss = torch.nn.CrossEntropyLoss(weight=weight)
        self.ordinal_reg_loss = CumulativeLinkLoss(class_weights=weight)

    def forward(self, input_ids, 
                attention_mask, 
                special_tokens_mask, 
                labels=None, 
                task="regression"):
        if task == "ordinal_regression":
            self.cutpoints_clip()

        outputs = {"encoder_outputs": []}

        att_mask = attention_mask

        #(BS, SEQ_LEN, HS)
        embed_output = self.embeddings(input_ids=input_ids)
        outputs["embeddings"] = embed_output

        hs = embed_output

        for layer_n, encoder_layer in enumerate(self.encoder_layers):
            if layer_n > self.max_layer:
                break

            if self.powerbert_cuts is not None:
                powerbert_cut = self.powerbert_cuts[layer_n]
            else:
                powerbert_cut = None

            encoder_output = encoder_layer(hs, 
                                           att_mask = att_mask, 
                                           layer_n = layer_n,
                                           prune_heads = None,
                                           powerbert_cut = powerbert_cut,
                                           find_powerbert_conf = self.find_powerbert_conf)
            att_mask = encoder_output["att_mask"]

            outputs["encoder_outputs"].append(encoder_output)

            hs = encoder_output["output"]
            att_mask = encoder_output["att_mask"]

        if self.max_layer == -1:
            final_cls_token = hs.sum(dim=1)
        else:
            final_cls_token = hs[:,0,:]

        mlp_output = self.mlp(final_cls_token)

        reg_output = self.reg_head(mlp_output)
        outputs["qe_scores"] = reg_output.squeeze(dim=-1)

        if task == "ordinal_regression":
            qe_logits = self.ordinal_reg_head(reg_output)
            outputs["qe_cls"] = torch.argmax(qe_logits, dim=-1)

        else:
            qe_logits = self.cls_head(mlp_output)
            outputs["qe_cls"] = torch.argmax(self.cls_softmax(qe_logits), dim=-1)

        if labels is not None:
            if task == "regression":
                outputs["loss"] = self.reg_loss(outputs["qe_scores"], labels)
            elif task == "ordinal_regression":
                outputs["loss"] = self.ordinal_reg_loss(qe_logits, labels.unsqueeze(-1))
            else:
                outputs["loss"] = self.cls_loss(qe_logits, labels)

        return outputs

    def cutpoints_clip(self, margin = 0.0, min_val = -1e6):
        cutpoints = self.ordinal_reg_head.link.cutpoints.data
        for i in range(cutpoints.shape[0] - 1):
            cutpoints[i].clamp_(min_val, cutpoints[i + 1] - margin)
