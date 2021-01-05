import torch
import time
from modules.transformer import EncoderLayer
from modules.qe_transformer import QETransformer

seq_len = 128
hs = 1024
intermediate = hs * 4
heads = 16
layers = 24
input = torch.ones((1, seq_len, hs))
mask = torch.ones((1, seq_len))

model = EncoderLayer(heads, hs, intermediate, 64, 64)
model = EncoderLayer(12, 768, 768*4, 64, 64)
for params in model.parameters():
    print(params.shape)


#model = QETransformer(model_name="xlm-roberta-base", n_heads=12, n_layers=12,intermediate_size=768*4, hidden_size=768)

model = QETransformer()

print(sum(p.numel() for p in model.embeddings.parameters()))
print(sum(p.numel() for p in model.encoder_layers[0].parameters()))

print(sum(p.numel() for p in model.mlp.parameters()) +sum(p.numel() for p in model.reg_head.parameters()))




"""
s = time.time()

cut = [108, 102, 92, 89, 84, 79, 77, 77, 77, 77, 67, 58, 58, 58, 55, 55, 43, 41, 36, 36, 36, 28, 23, 21]
cut = [88,83,73,68,64,63,60,60,60,56,46,42,42,42,34,34,24,21,15,15,14,4,2,1]
cut= [103,94,77,71,65,65,59,59,59,57,48,39,39,39,33,33,20,14,9,9,9,3,2,1]
cut = [81,75,60,54,49,46,41,41,41,37,28,25,25,25,12,12,2,2,1,1,1,1,1,1]
cut = [96,93,94,79,77,72,71,71,71,67,63,54,54,54,50,50,40,37,32,32,32,23,18,16]
cut = [102,93,77,73,64,60,56,56,56,56,45,37,37,36,29,29,19,12,6,6,3,1,1,1]
cut = [103, 93, 79, 73, 66, 61, 57, 57, 57, 53, 45, 43, 43, 43, 35, 35, 25, 19, 14, 14, 14, 4, 1, 1]
cut = [97,84,68,63,56,49,42,42,42,41,32,29,29,29,24,24,15,10,6,6,5,1,1,1]
#cut = [128 for i in range(24)]

total_sa = 0
total_mlp = 0
for i in range(layers):
    print(i)
    start=time.time()
    att_output = model.MHattention(input, input, input, mask)["output"]

    #if i < 6:
    #    att_output = att_output[:,:60,:]
    #    mask = mask[:,:60]
    #elif i >=6 and i < 12:
    #    att_output = att_output[:,:40,:]
    #    mask = mask[:,:40]
    #if i >=6 and i < 18:
    #    att_output = att_output[:,:30,:]
    #    mask = mask[:,:30]
    #elif i >= 18:
    #    att_output = att_output[:,:30,:]
    #    mask = mask[:,:30]
    att_output = att_output[:, :cut[i], :]
    mask = mask[:, :cut[i]]

    end=time.time()
    total_sa += end-start
    print(1000*(end-start))

    start=time.time()
    output = model.fc(att_output)["output"] + att_output
    input = output
    end=time.time()
    total_mlp += end-start
    print(1000*(end-start))


avg_sa = 1000*total_sa/24
avg_mlp = 1000*total_mlp/24
print(avg_sa, avg_mlp)
print((avg_sa+avg_mlp)*24)
"""
