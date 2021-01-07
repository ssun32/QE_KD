from glob import glob
import numpy as np

def get_scores(f):
    t = None
    for f in glob(f):
        print(f)
        if t is None:
            t = np.array([float(s)*100 for s in open(f)])
        else:
            t += np.array([float(s)*100 for s in open(f)])
    t/=5
    t = np.rint(t).clip(0, 100)

    return t


s1 = get_scores("new_configs2/xlmr_large_2/regression/si_en/run*/best.test.output")
s2 = get_scores("new_configs2/xlmr_large_23/regression/si_en/run*/best.test.output")

s = []
for i, l in enumerate(open("data/si-en/test20.sien.df.short.tsv")):
    if i > 0:
        s.append(float(l.split("\t")[4]))

s = np.rint(np.array(s))
print(s1[:100])
print(s2[:100])
print(s[:100])
