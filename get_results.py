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

def get_f1(preds, gold_labels, average="macro"):
    return f1_score(preds, gold_labels, average=average)

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
    gold_labels[cur_ld]["bi_classification"] = [convert_to_class(label, dataset, normalized=False, binary=True)  for label in gold_labels[cur_ld]["regression"]]

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def get_results(cur_ld, task, model, layer):
    dataset = "qe" if cur_ld != "blog" else "blog"
    run_results = []
    for run in range(5):
        results = defaultdict(dict)
        predf = "new_configs/xlmr_%s_%s/%s/%s/run%s/best.test.output"%(model, layer, task, cur_ld, run)
        try:
            preds = [float(l) for l in open(predf)]
        except:
            #print(predf)
            preds = [0.0] * 1000

        if cur_ld == "all":
            preds_chunks = chunks(preds, 1000)
            chunk_lds = lds
        else:
            preds_chunks = [preds]
            chunk_lds = [cur_ld]

        if task == "regression":
            for chunk_ld, preds_chunk in zip(chunk_lds, preds_chunks):
                results[chunk_ld]["regression"] = get_pc(preds_chunk, gold_labels[chunk_ld]["regression"])
                #multiclass classification
                cls_preds = [convert_to_class(label, dataset, normalized=True) for label in preds_chunk]
                results[chunk_ld]["classification"] = get_f1(cls_preds, gold_labels[chunk_ld]["classification"], average="micro")

                #binary classification
                cls_preds = [convert_to_class(label, dataset, normalized=True, binary=True) for label in preds_chunk]
                results[chunk_ld]["bi_classification"] = get_f1(cls_preds, gold_labels[chunk_ld]["bi_classification"], average="binary")

        elif task == "classification":
            for chunk_ld, preds_chunk in zip(chunk_lds, preds_chunks):
                #multiclass classification
                results[chunk_ld]["classification"] = get_f1(preds_chunk, gold_labels[chunk_ld]["classification"], average="micro")

        else:
            for chunk_ld, preds_chunk in zip(chunk_lds, preds_chunks):
                #multiclass classification
                results[chunk_ld]["bi_classification"] = get_f1(preds_chunk, gold_labels[chunk_ld]["bi_classification"], average="binary")

        run_results.append(results)

    # getmean
    mean_results = {}
    for chunk_ld in run_results[0].keys():
        mean_results[chunk_ld] = {}
        for metric in run_results[0][chunk_ld].keys():
            mean_results[chunk_ld][metric] = np.array([run_result[chunk_ld][metric] for run_result in run_results]).mean()

    return mean_results

all_results = {}
for cur_ld in ["all", "en_de", "en_zh", "ro_en", "et_en", "si_en", "ne_en", "ru_en", "blog"]:
    all_results[cur_ld] = {}
    for task in ["regression", "classification"]:
        all_results[cur_ld][task] = {}
        for model in ["base", "large"]:
            all_results[cur_ld][task][model] = {}
            layers = [0,2,5,8,11] if model == "base" else [0,2,5,8,11,14,17,20,23]
            for layer in layers:
                results = get_results(cur_ld, task, model, layer)
                all_results[cur_ld][task][model][layer] = results

print(json.dumps(all_results, indent=4))
