import json
import sys
import numpy as np
from modules.utils import convert_to_class
from scipy.stats import pearsonr
from sklearn.metrics import f1_score
from collections import defaultdict

file = sys.argv[1]
metric = sys.argv[2]

with open(file) as f:
    all_results = json.load(f)

#lds = ["si_en", "ne_en", "ro_en", "ne_en", "en_de", "en_zh", "ru_en"]

lds = ["si_en", "ro_en", "ne_en", "en_de", "en_zh", "ru_en"]


if metric == "micro":
    metric_name = "Micro F1"
elif metric == "macro":
    metric_name = "Macro F1"
else:
    metric_name = "Binary F1"

#header
layers = [0,2,5,8,11,14,17,20,23]

def nice_print(name, l):
    max_n = max(l)
    print("%s: %s" % (name, "\t".join(["%.2f(%.0f%%)"%(x, 100*(x/max_n-1)) for x in l])))

cls = "bi_classification" if metric == "binary" else "classification"

for model in ["base", "large"]:
    reg_pc = []
    reg_f1 = []
    reg_f1_bi = []
    reg_ml_pc = []
    reg_ml_f1 = []
    reg_ml_f1_bi = []

    cls_f1 = []
    cls_f1_bi = []
    cls_ml_pc = []
    cls_ml_f1 = []
    cls_ml_f1_bi = []

    for layer in layers:
        bm_regl = []
        bm_clsl = []
        bm_bi_clsl = []

        mm_regl = []
        mm_clsl = []
        mm_bi_clsl = []

        cls_bm_clsl = []
        cls_bm_bi_clsl = []
        cls_mm_clsl = []
        cls_mm_bi_clsl = []

        for cur_ld in lds:
            if (model=="base" and layer >= 12):
                continue

            results_regression = all_results[cur_ld]["regression"][model][str(layer)]
            results_classification = all_results[cur_ld]["classification"][model][str(layer)]
            results_classification_bi = all_results[cur_ld]["bi_classification"][model][str(layer)]

            if "cur_ld" != "blog":
                ml_results_regression = all_results["all"]["regression"][model][str(layer)]
                ml_results_classification = all_results["all"]["classification"][model][str(layer)]
                ml_results_classification_bi = all_results["all"]["bi_classification"][model][str(layer)]


            for ld in results_regression:
                metrics_regression = results_regression[ld]
                metrics_classification = results_classification[ld]
                metrics_classification_bi = results_classification_bi[ld]

                if cur_ld != "blog":
                    ml_metrics_regression = ml_results_regression[ld]
                    ml_metrics_classification = ml_results_classification[ld]
                    ml_metrics_classification_bi = ml_results_classification_bi[ld]


            bm_regl.append(metrics_regression["regression"])
            bm_clsl.append(metrics_regression["classification"])
            bm_bi_clsl.append(metrics_regression["bi_classification"])

            mm_regl.append(ml_metrics_regression["regression"])
            mm_clsl.append(ml_metrics_regression["classification"])
            mm_bi_clsl.append(ml_metrics_regression["bi_classification"])

            cls_bm_clsl.append(metrics_classification["classification"])
            cls_bm_bi_clsl.append(metrics_classification_bi["bi_classification"])
            cls_mm_clsl.append(ml_metrics_classification["classification"])
            cls_mm_bi_clsl.append(ml_metrics_classification_bi["bi_classification"])


        if not (model=="base" and layer >= 12):

            reg_pc.append(np.array(bm_regl).mean())
            reg_ml_pc.append(np.array(mm_regl).mean())

            reg_f1.append(np.array(bm_clsl).mean())
            reg_ml_f1.append(np.array(mm_clsl).mean())
            cls_f1.append(np.array(cls_bm_clsl).mean())
            cls_ml_f1.append(np.array(cls_mm_clsl).mean())

            reg_f1_bi.append(np.array(bm_bi_clsl).mean())
            reg_ml_f1_bi.append(np.array(mm_bi_clsl).mean())
            cls_f1_bi.append(np.array(cls_bm_bi_clsl).mean())
            cls_ml_f1_bi.append(np.array(cls_mm_bi_clsl).mean())

    nice_print("bl_pc",reg_pc)
    nice_print("ml_pc",reg_ml_pc)

    nice_print("bl_f1", reg_f1)
    nice_print("ml_f1", reg_ml_f1)
    nice_print("bl_cls_f1", cls_f1)
    nice_print("ml_cls_f1", cls_ml_f1)

    nice_print("bi_bi_f1", reg_f1_bi)
    nice_print("ml_bi_f1", reg_ml_f1_bi)
    nice_print("cls_bi_f1", cls_f1_bi)
    nice_print("ml_cls_bi_f1", cls_ml_f1_bi)
