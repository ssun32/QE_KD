import sys
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.model import QETransformer


class ScaledDotProductAttention(nn.Module):

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        log_attn = F.log_softmax(attn, dim=-1)

        output = torch.matmul(attn, v)

        return {"input": q,
                "output": output, 
                "attn": attn, 
                "log_attn": log_attn}


class FeedForward(nn.Module):

    def __init__(self, hidden_size, intermediate_size, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Sequential(
                        nn.Linear(hidden_size, 4*hidden_size, bias=True))
        self.gelu = nn.GELU()
    
        self.fc2 = nn.Sequential(
                    nn.Linear(4*hidden_size, hidden_size, bias=True),
                    nn.Dropout(dropout))

        self.ln = nn.LayerNorm(hidden_size, eps=1e-5)

        if intermediate_size != 4*hidden_size:
            self.fc1_proj = nn.Sequential(
                                nn.Linear(4*hidden_size, intermediate_size),
                                nn.Dropout(dropout))
            self.fc2_proj = nn.Sequential(
                                nn.Linear(intermediate_size, 4*hidden_size),
                                nn.Dropout(dropout))
        else:
            self.fc1_proj, self.fc2_proj = None, None

    def forward(self, hs):
        fc1_output = self.fc1(hs)
        if self.fc1_proj is not None:
            fc1_output = self.fc1_proj(fc1_output)

        fc1_output = self.gelu(fc1_output)

        if self.fc2_proj is not None:
            fc1_output = self.fc2_proj(fc1_output)

        fc2_output = self.fc2(fc1_output)
        ln_output = self.ln(fc2_output + hs)

        return {"input": hs,
                "output": ln_output,
                "fc1_output": fc1_output,
                "fc2_output": fc2_output}


class MultiHeadAttention(nn.Module):

    def __init__(self, n_head, d_model, d_k, d_v, lf_d = 128, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.query = nn.Linear(d_model, n_head * d_k, bias=True)
        self.key = nn.Linear(d_model, n_head * d_k, bias=True)
        self.value = nn.Linear(d_model, n_head * d_v, bias=True)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=True)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model, eps=1e-5)

    def forward(self, q, k, v, mask=None, layer_n=None, prune_heads = None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        #reshape to batch_size x seq_len x 16 x 64
        q = self.query(q).view(sz_b, len_q, n_head, d_k)
        k = self.key(k).view(sz_b, len_k, n_head, d_k)
        v = self.value(v).view(sz_b, len_v, n_head, d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(-1) * mask.unsqueeze(1)
            mask = mask.unsqueeze(1)

        sa_output = self.attention(q, k, v, mask = mask)

        hs = sa_output["output"]

        hs = hs.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        fc_output = self.dropout(self.fc(hs))

        ln_output = self.ln(fc_output + residual)

        return {"input": residual,
                "output": ln_output,
                "sa_output": sa_output,
                "fc_output": fc_output}

class EncoderLayer(nn.Module):
    def __init__(self, n_head, d_model, intermediate_size, d_k, d_v, dropout=0.1):
        super().__init__()
        self.MHattention = MultiHeadAttention(n_head, d_model, d_k, d_v)
        self.fc = FeedForward(d_model, intermediate_size, dropout)
        
        #make a soft token mask with max_seq = 128
        self.token_mask = nn.Parameter(torch.ones([1,128,1], requires_grad=True))

    def forward(self, 
                hs, 
                att_mask, 
                layer_n = None, 
                prune_heads = None,
                powerbert_cut = None,
                find_powerbert_conf=False):

        #multihead self_attention
        mhsa_outputs = self.MHattention(
                                    hs, 
                                    hs, 
                                    hs, 
                                    att_mask, 
                                    layer_n = layer_n, 
                                    prune_heads = prune_heads)

        att_output = mhsa_outputs["output"]

        if find_powerbert_conf or powerbert_cut is not None:
            attn = mhsa_outputs["sa_output"]["attn"]

            #powerbert - calculate attention-based scoring
            att_score = attn.sum(dim=-2).sum(dim=1)[:,1:]
            sorted_att_score, indices = att_score.sort(descending=True)
            indices = indices + 1
            rows = torch.arange(0, att_score.size(0), dtype=torch.long).unsqueeze(-1)
            cols = indices[:, :powerbert_cut]
            cols = torch.cat([torch.zeros(cols.size(0), 1, dtype=torch.long).to(cols.device), cols], dim=-1)
            att_output = att_output[rows, cols, :]
            att_mask = att_mask[rows, cols]

            #clamp to [0, 1]
            token_mask = torch.clamp(
                            self.token_mask[:,:att_mask.size(1),:] * att_mask.unsqueeze(-1),
                            min=0.0,
                            max=1.0)

            if find_powerbert_conf:
                att_output *= token_mask
        else:
            token_mask = None

        fc_outputs = self.fc(att_output)

        return {"input": hs,
                "output": fc_outputs["output"],
                "mhsa_outputs": mhsa_outputs,
                "fc_outputs": fc_outputs,
                "att_mask": att_mask,
                "token_mask": token_mask}
