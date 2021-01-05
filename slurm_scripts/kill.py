import json
import os
from subprocess import check_output, call

log = check_output("squeue -u ssfei81", shell=True)
for l in log.decode("utf-8").split("\n")[2:-1]:
    if not l.strip().split()[2].startswith("xlmr"):
        continue
    qid = l.strip().split()[0]
    rid = qid.split("_")[-1]
    config = check_output("ls ../configs/models/*/*/kd_config_*.json | sed -n %sp"%rid, shell=True).decode("utf-8").strip()

    with open(config) as f:
        j = json.load(f)

    logf = os.path.join("..", j["output_dir"], j["checkpoint_prefix"]+"log")

    train_losses, dev_losses, pcs, mses = [], [], [], []
    for l in open(logf):
        l = l.strip()
        if "Train loss" in l:
            train_loss = float(l.split()[-1])
        elif "Dev loss" in l:
            l = l.split()
            dev_loss = float(l[3])
            mse = float(l[5])
            pc = float(l[-1].split(":")[-1])

            train_losses.append(train_loss)
            dev_losses.append(dev_loss)
            pcs.append(pc)
            mses.append(mse)

    early_stop = 0
    best_dev_loss = -100000
    for pc, mse, dev_loss, train_loss in zip(pcs, mses, train_losses, dev_losses):
        if -dev_loss > best_dev_loss:
            best_dev_loss = -dev_loss
            early_stop = 0
        else:
            early_stop += 1

        if early_stop > 1:
            print("Killing %s" % qid)
            #call("scancel %s" % qid, shell=True)
            break
