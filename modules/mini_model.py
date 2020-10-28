import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import XLMRobertaConfig, XLMRobertaModel

class QEMini(nn.Module):

    def __init__(
            self, 
            vocab_size, 
            hidden_size = 1024, 
            n_heads = 16,
            layers = 12):
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

        return output, attn, log_attn


class MultiHeadAttention(nn.Module):

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.query = nn.Linear(d_model, n_head * d_k, bias=True)
        self.key = nn.Linear(d_model, n_head * d_k, bias=True)
        self.value = nn.Linear(d_model, n_head * d_v, bias=True)
        self.dense = nn.Linear(n_head * d_v, d_model, bias=True)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-5)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        q = self.query(q).view(sz_b, len_q, n_head, d_k)
        k = self.key(k).view(sz_b, len_k, n_head, d_k)
        v = self.value(v).view(sz_b, len_v, n_head, d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        q, attn, log_attn = self.attention(q, k, v, mask=mask)

        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.dense(q))
        q += residual

        q = self.layer_norm(q)
        return q, attn, log_attn


class EncoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.MHattention = MultiHeadAttention(n_head, d_model, d_k, d_v)
        self.fc = nn.Sequential(
                    nn.Linear(d_model, d_model*4, bias=True),
                    nn.GELU(),
                    nn.Linear(d_model*4, d_model, bias=True),
                    nn.Dropout(dropout))
        
        self.ln = nn.LayerNorm(d_model, eps=1e-5)

    def forward(self, hidden_state, mask):
        att_output, attn, log_attn = self.MHattention(hidden_state, hidden_state, hidden_state, mask)
        return self.ln(self.fc(att_output)+att_output), attn, log_attn
