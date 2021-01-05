import json
import numpy as np
from modules.utils import convert_to_class
from scipy.stats import pearsonr
from sklearn.metrics import f1_score
from collections import defaultdict

lds = ["en_de","en_zh","ro_en","et_en","si_en","ne_en", "ru_en"]
def convert_to_binary(cls, dataset):
    if dataset=="qe":
        return int(cls >= 3)
    else:
        return int(cls == 1)

def get_gains(testrs):
    return ["%.3f (%.1f%%)"%(float(testr), ((float(testr)/float(testrs[-1]))-1)*100) for testr in testrs]

def get_pc(preds, gold_labels):
    return pearsonr(preds, gold_labels)[0]

def get_f1(preds, gold_labels):
    return f1_score(preds, gold_labels, average="micro")

gold_labels = {}
for cur_ld in ["en_de", "en_zh", "ro_en", "et_en", "si_en", "ne_en", "ru_en", "blog"]:
    gold_labels[cur_ld] = {}

    if cur_ld == "blog":
        goldf = "data/blogs/test.tsv"
        reg_labels = [float(l.split("\t")[0]) for l in open(goldf)]
        gold_labels[cur_ld]["regression"] = reg_labels

    else:
        gold_labels[cur_ld] = {}
        gold_labels[cur_ld]["regression"] = []
        goldf = "data/%s/test20.%s.df.short.tsv"%(cur_ld.replace("_","-"), cur_ld.replace("_",""))
        for i, l in enumerate(open(goldf)):
            if i > 0:
                da_score = float(l.strip().split("\t")[4])
                gold_labels[cur_ld]["regression"].append(da_score)

    #cls labels
    dataset = "qe" if cur_ld != "blog" else "blog"
    gold_labels[cur_ld]["classification"] = [convert_to_class(label, dataset, normalized=False)  for label in gold_labels[cur_ld]["regression"]]
    #gold_labels[cur_ld]["bi_classification"] = [convert_to_class(label, dataset, normalized=False, binary=True)  for label in gold_labels[cur_ld]["regression"]]
   
    cls = defaultdict(int)
    for l in gold_labels[cur_ld]["classification"]:
        cls[l] += 1
    #print(cur_ld, cls)
    major_cls = max(cls.keys(), key=lambda x: cls[x])
    print(cur_ld, f1_score([major_cls]*len(gold_labels[cur_ld]["classification"]), gold_labels[cur_ld]["classification"], average="micro"))


