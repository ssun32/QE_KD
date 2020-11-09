import sys
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import XLMRobertaConfig, XLMRobertaModel
from modules.model import QETransformer

class QEMiniLinformer(nn.Module):

    def __init__(
            self, 
            vocab_size = 250002, 
            hidden_size = 1024, 
            n_heads = 16,
            n_layers = 24,
            config = None):
        super(QEMiniLinformer, self).__init__()

        self.head_dim = int(hidden_size/n_heads)
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.config = config
        self.copy()

    def copy(self):
        xlmr = QETransformer("xlm-roberta-large")
        checkpoint_path = os.path.join(self.config["output_dir"].replace("_linformer",''), "best.pt")
        xlmr.load_state_dict(torch.load(checkpoint_path))
        print("Copying embedding...")
        self.embeddings = xlmr.transformer.embeddings
        print("Copying mlp...")
        self.mlp = xlmr.mlp

        self.encoder_layers = nn.ModuleList([EncoderLayer(self.n_heads, self.hidden_size, self.head_dim, self.head_dim) for _ in range(self.n_layers)])

        def copy_encoder(s_encoder_layer, t_encoder_layer):
            #copy multiheadattention
            s_encoder_layer.MHattention.query.load_state_dict(
                    t_encoder_layer.attention.self.query.state_dict())
            s_encoder_layer.MHattention.key.load_state_dict(
                    t_encoder_layer.attention.self.key.state_dict())
            s_encoder_layer.MHattention.value.load_state_dict(
                    t_encoder_layer.attention.self.value.state_dict())
            s_encoder_layer.MHattention.dense.load_state_dict(
                    t_encoder_layer.attention.output.dense.state_dict())
            s_encoder_layer.MHattention.layer_norm.load_state_dict(
                    t_encoder_layer.attention.output.LayerNorm.state_dict())

            #copy feedforward
            s_encoder_layer.fc[0].load_state_dict(
                    t_encoder_layer.intermediate.dense.state_dict())
            s_encoder_layer.fc[2].load_state_dict(
                    t_encoder_layer.output.dense.state_dict())
            s_encoder_layer.ln.load_state_dict(
                    t_encoder_layer.output.LayerNorm.state_dict())

        for tgt, src in zip(self.encoder_layers, xlmr.transformer.encoder.layer):
            copy_encoder(tgt, src)

    def forward(self, batch, prune_dict = {}):
        transformer_outputs = {
                "hidden_states": [],
                "attentions": [],
                "log_attentions": [],
                "token_masks": []}

        #(BS, SEQ_LEN, SEQ_LEN)
        att_mask = batch[0]["attention_mask"]
        token_mask = att_mask.unsqueeze(-1)
        a3d_mask = token_mask.repeat(1, 1, att_mask.size(-1)) * token_mask.transpose(1,2)

        #(BS, SEQ_LEN, HS)
        embed_output = self.embeddings(input_ids=batch[0]["input_ids"])
        transformer_outputs["embeddings"] = embed_output

        hidden_state = embed_output
        for layer_n, encoder_layer in enumerate(self.encoder_layers):
            #if layer_n in [0]:
            #    continue
            #a3d_mask = token_mask.repeat(1, 1, att_mask.size(-1)) * token_mask.transpose(1,2)
            prune_heads = prune_dict[layer_n] if layer_n in prune_dict else None
            hidden_state, att, log_att, token_mask = encoder_layer(hidden_state, 
                                                                   mask = a3d_mask, 
                                                                   orig_mask = att_mask, 
                                                                   layer_n = layer_n,
                                                                   prune_heads = prune_heads)

            transformer_outputs["hidden_states"].append(hidden_state)
            transformer_outputs["attentions"].append(att)
            transformer_outputs["log_attentions"].append(log_att)
            transformer_outputs["token_masks"].append(token_mask)


        cls_token = hidden_state[:,0,:]
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

    def __init__(self, n_head, d_model, d_k, d_v, lf_d = 128, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.query = nn.Linear(d_model, n_head * d_k, bias=True)
        self.key = nn.Linear(d_model, n_head * d_k, bias=True)
        self.value = nn.Linear(d_model, n_head * d_v, bias=True)
        self.dense = nn.Linear(n_head * d_v, d_model, bias=True)

        #linformer params
        self.lf_key = nn.Linear(128, lf_d, bias=True)
        self.lf_value = nn.Linear(128, lf_d, bias=True)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-5)

        #with open("prunelist.json") as f:
        #    self.prunelist = json.load(f)

    def forward(self, q, k, v, mask=None, layer_n=None, prune_heads = None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
       
        #batch_size x seq_len x 16 x 64
        q = self.query(q).view(sz_b, len_q, n_head, d_k)
        k = self.key(k).view(sz_b, len_k, n_head, d_k)
        v = self.value(v).view(sz_b, len_v, n_head, d_v)

        k_sl = k.shape[1]
        v_sl = v.shape[1]

        #k = (k.transpose(1,3).matmul(self.lf_key.weight[:,:k_sl].t()) + self.lf_key.bias).transpose(1,3)
        #v = (v.transpose(1,3).matmul(self.lf_value.weight[:,:v_sl].t()) + self.lf_value.bias).transpose(1,3)


        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        q, attn, log_attn = self.attention(q, k, v, mask=mask)
        #if str(layer_n) in self.prunelist:
        #    i = self.prunelist[str(layer_n)]
        #    q[:,i,:,:] = 0
        #q[0,:,:,:] = 0
        #head_mask = torch.ones(q.size()).to(q.get_device())

        if prune_heads is not None:
            head_mask[:, prune_heads, :, :] = 0
        #if layer_n == 0:
        #    head_mask[:,:,:,:] = 0
        #elif layer_n == 1:
        #    head_mask[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],:,:] = 0
        #elif layer_n == 2:
        #    head_mask[:, [3, 4, 5], :, :] = 0
            #head_mask[:,:,:,:] = 0
        #elif layer_n == 2:
            #head_mask[:, [0, 1, 2, 3, 4],:,:] = 0
       #     head_mask[:,:,:,:] = 0
        #elif layer_n == 5:
        #    head_mask[:,[0, 1, 2],:,:] = 0
        #q=q*head_mask 

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

        #self.token_mask = nn.Sequential(
        #                    nn.Linear(d_model, 1),
        #                    nn.Dropout(dropout),
        #                    nn.Sigmoid())

        #def init_weight(layer):
        #    if type(layer) == nn.Linear:
        #        layer.bias.data.fill_(10)

        #self.token_mask.apply(init_weight)

    def forward(self, hidden_state, mask, orig_mask, layer_n = None, prune_heads=None):
        att_output, attn, log_attn = self.MHattention(hidden_state, hidden_state, hidden_state, mask, layer_n = layer_n, prune_heads = prune_heads)
        output = self.ln(self.fc(att_output) + att_output)
        
        #token_mask = self.token_mask(output) * orig_mask.unsqueeze(-1)
        token_mask = None

        return output, attn, log_attn, token_mask
