from math import sqrt
import numpy as np
from collections import defaultdict

pcs = defaultdict(list)
rmses = defaultdict(list)
lds = {}
directions = {}
layers = {}

for l in open("output.t_logit.txt"):
    if l.startswith("::"):
        continue
    elif l.startswith("config"):
        unlabel =  "unlabel" in l
        l = l.strip().split("/")
        ld = l[-3]
        l = l[-1].split("_")
        direction = l[1]
        layer = int(l[2])
        lds[ld] = 1
        directions[direction] = 1
        layers[layer] = 1
    else:
        pc, mse = l.strip().split()
        pc = float(pc)
        rmse = sqrt(float(mse))
        pcs[(ld, direction, layer, unlabel)].append(pc)
        rmses[(ld, direction, layer, unlabel)].append(rmse)

for ld in sorted(lds):
    for direction in sorted(directions):
        for layer in sorted(layers):
            k1 = (ld, direction, layer, True)
            k2 = (ld, direction, layer, False)

            pc_k1 = np.array(pcs[k1]).mean()
            rmse_k1 = np.array(rmses[k1]).mean()
            pc_k2 = np.array(pcs[k2]).mean()
            rmse_k2 = np.array(rmses[k2]).mean()

            print("%s\t%s\t%s\t%.3f\t%.3f"%(ld, direction, layer, pc_k1, pc_k2))
