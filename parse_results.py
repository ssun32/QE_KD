import json
import sys
import numpy as np
from modules.utils import convert_to_class
from scipy.stats import pearsonr
from sklearn.metrics import f1_score
from collections import defaultdict

file = sys.argv[1]
ld = sys.argv[2]
metric = sys.argv[3]

with open(file) as f:
    all_results = json.load(f)

if ld == "low":
    lds = ["si_en", "ne_en"]
    name = "Low resource LDs"
elif ld == "mid":
    lds = ["ro_en", "et_en"]
    name = "Mid resource LDs"
elif ld == "high":
    lds = ["en_de", "en_zh", "ru_en"]
    name = "High resource LDs"
else:
    lds = ["blog"]
    name = "Blog"

if metric == "micro":
    metric_name = "Micro F1"
elif metric == "macro":
    metric_name = "Macro F1"
else:
    metric_name = "Binary F1"

#lds = ["en_de", "en_zh", "ro_en", "et_en", "si_en", "ne_en", "ru_en", "blog"]

#header
layers = [0,2,5,8,11,14,17,20,23]
headers = ["LD", "XLMR", "OBJ", "MTC", "MDL"] + [l+1 for l in layers]
print("\\begin{table*}")
print("\\centering")
print("\\begin{tabular}{cccccccccccccc}")
print("\\toprule")
print("&".join(["","","","", "\\multicolumn{%s}{c}{\\textbf{Layers}}"%len(layers)])+"\\\\")
print("\\cmidrule{4-14}")
print("&".join(["\\textbf{%s}"%h for h in headers])+"\\\\")
print("\\bottomrule")

cls = "bi_classification" if metric == "binary" else "classification"
for cur_ld in lds:
    for model in ["base", "large"]:
        pc_res = []
        f1_res = []
        f1_res2 = []
        ml_pc_res = []
        ml_f1_res = []
        ml_f1_res2 = []

        for layer in layers:
       
            if str(layer) not in all_results[cur_ld]["regression"][model]:
                pc_res.append("-")
                f1_res.append("-")
                f1_res2.append("-")

                ml_pc_res.append("-")
                ml_f1_res.append("-")
                ml_f1_res2.append("-")
                continue

            results_regression = all_results[cur_ld]["regression"][model][str(layer)]
            results_classification = all_results[cur_ld][cls][model][str(layer)]

            if "cur_ld" != "blog":
                ml_results_regression = all_results["all"]["regression"][model][str(layer)]
                ml_results_classification = all_results["all"][cls][model][str(layer)]

            for ld in results_regression:
                metrics_regression = results_regression[ld]
                metrics_classification = results_classification[ld]
                pc_res.append("%.2f"%metrics_regression["regression"])
                f1_tmp = "%.2f"%metrics_regression[cls]
                f1_tmp2 = "%.2f"%metrics_classification[cls]
                max_f1 = max(float(f1_tmp), float(f1_tmp2))

                if cur_ld != "blog":
                    ml_metrics_regression = ml_results_regression[ld]
                    ml_metrics_classification = ml_results_classification[ld]
                    ml_pc_res.append("%.2f"%ml_metrics_regression["regression"])
                    ml_f1_tmp = "%.2f"%ml_metrics_regression[cls]
                    ml_f1_tmp2 = "%.2f"%ml_metrics_classification[cls]
                    max_f1 = max(max_f1, float(ml_f1_tmp), float(ml_f1_tmp2))


                f1_res.append("\\textbf{%s}"%f1_tmp if float(f1_tmp) == max_f1 else "%s"%f1_tmp)
                f1_res2.append("\\textbf{%s}"%f1_tmp2 if float(f1_tmp2) == max_f1 else "%s"%f1_tmp2)

                if cur_ld != "blog":
                    ml_f1_res.append("\\textbf{%s}"%ml_f1_tmp if float(ml_f1_tmp) == max_f1 else "%s"%ml_f1_tmp)
                    ml_f1_res2.append("\\textbf{%s}"%ml_f1_tmp2 if float(ml_f1_tmp2) == max_f1 else "%s"%ml_f1_tmp2)


"""
        firstline = ["\multirow{%s}{*}{%s}"%(6 if cur_ld != "blog" else 3, model), 
                     "\multirow{%s}{*}{REG}"%(4 if cur_ld != "blog" else 2), 
                     "\\multirow{%s}{*}{PC}"%(2 if cur_ld != "blog" else 1), "BL"]+pc_res
        if model == "base":
            firstline  = ["\multirow{%s}{*}{%s}"%(12 if cur_ld != "blog" else 6, cur_ld.replace("_", "-"))] + firstline
        else:
            firstline = [""] + firstline
        print("&".join(firstline)+"\\\\")
        if cur_ld != "blog":
            print("&".join(["", "", "", "", "ML"]+ml_pc_res)+"\\\\")
        print("\\cmidrule{5-14}")
        print("&".join(["", "", "", "\\multirow{%s}{*}{F1}" % (2 if cur_ld != "blog" else 1), "BL"]+f1_res)+"\\\\")
        if cur_ld != "blog":
            print("&".join(["", "", "", "", "ML"]+ml_f1_res)+"\\\\")
        print("\\cmidrule{3-14}")

        print("&".join(["", "", "\\multirow{%s}{*}{CLS}"% (2 if cur_ld != "blog" else 1), "\\multirow{%s}{*}{F1}"% (2 if cur_ld != "blog" else 1), "BL"]+f1_res2)+"\\\\")

        if cur_ld != "blog":
            print("&".join(["", "", "", "", "ML"]+ml_f1_res2)+"\\\\")

        if model == "base":
            print("\\cmidrule{2-14}")
        else:
            print("\\bottomrule")

print("\\end{tabular}")
print("\\caption{\label{citation-guide}")
print("%s (%s)"%(name, metric_name))
print("}")
print("\\end{table*}")
"""
