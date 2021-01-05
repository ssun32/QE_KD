import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import XLMRobertaConfig, XLMRobertaModel
from modules.transformer import EncoderLayer

class QEMini(nn.Module):

    def __init__(
            self, 
            vocab_size = 250002, 
            hidden_size = 1024, 
            n_heads = 16,
            layers = 24):
        super(QEMini, self).__init__()

        head_dim = int(hidden_size/n_heads)

        config = XLMRobertaConfig(max_position_embeddings=514,hidden_size=1024,num_hidden_layers=1, type_vocab_size=1, num_attention_heads=16, vocab_size=vocab_size)

        xlmr = XLMRobertaModel(config)
        self.embeddings = xlmr.embeddings

        self.encoder_layers = nn.ModuleList([EncoderLayer(n_heads, hidden_size, head_dim, head_dim) for _ in range(layers)])

        self.mlp = nn.Sequential(
                    nn.Linear(hidden_size, 4*hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(4*hidden_size, 1))

    def forward(self, batch):
        transformer_outputs = {
                "hidden_states": [],
                "attentions": [],
                "log_attentions": []}

        #(BS, SEQ_LEN, SEQ_LEN)
        att_mask = batch[0]["attention_mask"]
        a3d_mask = att_mask.unsqueeze(1).repeat(1, att_mask.size(-1), 1) * att_mask.unsqueeze(-1)

        #(BS, SEQ_LEN, HS)
        embed_output = self.embeddings(input_ids=batch[0]["input_ids"])
        transformer_outputs["embeddings"] = embed_output

        hidden_state = embed_output
        for encoder_layer in self.encoder_layers:
            hidden_state, att, log_att = encoder_layer(hidden_state, mask = a3d_mask)
            transformer_outputs["hidden_states"].append(hidden_state)
            transformer_outputs["attentions"].append(att)
            transformer_outputs["log_attentions"].append(log_att)

        cls_token = hidden_state[:,0,:]
        #cls_token = (att_output * att_mask.unsqueeze(-1)).sum(dim=1) / att_mask.sum(dim=-1, keepdim=True)
        return self.mlp(cls_token).squeeze(), transformer_outputs
