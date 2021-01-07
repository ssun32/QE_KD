import os
import json
from glob import glob
n_runs=5
batch_size_per_gpu=8
 
ids = {
 "all":[("en","de"),("en","zh"),("ro","en"),("et","en"),("si","en"),("ne","en"), ("ru", "en")],
 "en_de":[("en","de")],
 "en_zh":[("en","zh")],
 "ro_en":[("ro","en")],
 "et_en":[("et","en")],
 "si_en":[("si","en")],
 "ne_en":[("ne","en")],
 "ru_en":[("ru","en")]
 }

#H1 experiments
def get_files(lcodes, split="train"):
    tsv_file, mt_file, wp_file = [], [], []
    for src_lcode, tgt_lcode in lcodes:
        if split=="test":
            cur_split="test20"
        else:
            cur_split=split
        filedir = "data/%s-%s"%(src_lcode, tgt_lcode)
        tsv_file += glob("%s/%s.*.tsv" % (filedir,cur_split))
    return tsv_file

def make_config(train, 
                dev, 
                test, 
                exp="H1",
                use_layers=None,
                n_layers=24,
                n_heads=16,
                hidden_size=1024,
                intermediate_size=4096,
                max_layer = 23,
                train_layer=1,
                encoder_copy_mapping=None,
                encoder_sup_mapping=None,
                use_transferset=False,
                cfg_file_name="config.json",
                checkpoint_prefix="kdmini_",
                name="all",
                n_runs=1, 
                epochs=100,
                learning_rate=1e-4,
                eval_interval=20,
                batch_size_per_gpu=24,
                model_name="xlm-roberta-large",
                task="regression",
                dataset = "qe",
                checkpoint_path=None,
                config_prefix="new_configs2"):

    for run in range(n_runs):
        config = {
                  "model_name":model_name,
                  "task":task,
                  "dataset":dataset,
                  "epochs":epochs,
                  "use_layers": use_layers,
                  "n_layers": n_layers,
                  "n_heads": n_heads,
                  "hidden_size": hidden_size,
                  "intermediate_size": intermediate_size,
                  "max_layer": max_layer,
                  "train_layer":train_layer,
                  "encoder_copy_mapping":encoder_copy_mapping,
                  "encoder_sup_mapping":encoder_sup_mapping,
                  "use_transferset":use_transferset,
                  "batch_size_per_gpu": batch_size_per_gpu,
                  "learning_rate":learning_rate,
                  #"checkpoint_path": "new_configs/models/%s/%s/run%s/model.pt"%(task, name, run),
                  "checkpoint_path": checkpoint_path,
                  "checkpoint_prefix": checkpoint_prefix,
                  "eval_interval": eval_interval * len(train),
                  "train":[],
                  "dev":[],
                  "test":[]
                }
        output_dir = "%s/%s/%s/%s/run%s/" % (config_prefix, exp, task, name, run)
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, cfg_file_name), "w") as fout:
            config["output_dir"] =  output_dir

            for split_name, split in [("train", train), ("dev", dev), ("test", test)]:
                for id in split:
                    if dataset == "qe":
                        tsv_file = get_files(ids[id], split=split_name)
                    else:
                        tsv_file = ["data/blogs/%s.tsv"%split_name]
                    if len(tsv_file) > 1:
                        id = "all"
                    config[split_name].append({"id": id, "tsv_file":tsv_file})
            print(json.dumps(config, indent=4), file=fout)

#single languages
all_lds = ["_".join(ld) for ld in ids["all"]]


for prefix in ["new_configs3"]:
    for n_layers, n_heads, hidden_size, intermediate_size, exp_name, model_name, checkpoint_path in \
            [(24, 16, 1024, 4096, "xlmr_large", "xlm-roberta-large", "./xlmr_models/xlm-roberta-large.pt"),
             (12, 12, 768, 3072, "xlmr_base", "xlm-roberta-base", "./xlmr_models/xlm-roberta-base.pt")]:

        #for task in ["regression", "ordinal_regression", "classification", "bi_classification"]:
        for task in ["bi_classification"]:

            if model_name == "xlm-roberta-base": 
                layers = [0,3,5,8,11]
            else:
                layers = [0,3,5,8,11,14,17,20,23]

            for max_layer in layers:
                if prefix == "powerbert_configs" and max_layer != layers[-1]:
                    continue
                for ld in all_lds + ["all"] + ["blog"]:
                    make_config([ld], 
                                [ld], 
                                [ld], 
                                exp="%s_%s"%(exp_name,max_layer),
                                n_layers = n_layers, 
                                n_heads = n_heads,
                                hidden_size = hidden_size,
                                intermediate_size = intermediate_size,
                                max_layer = max_layer,
                                model_name = model_name,
                                task=task,
                                dataset="blog" if ld=="blog" else "qe",
                                name=ld, 
                                n_runs=5, 
                                batch_size_per_gpu=8, 
                                learning_rate=1e-6,
                                epochs=50, 
                                eval_interval = 100 if ld=="blog" else 20,
                                checkpoint_path=checkpoint_path,
                                config_prefix=prefix)

for n_layers, n_heads, hidden_size, intermediate_size, exp_name, model_name, checkpoint_path in \
        [(24, 16, 1024, 4096, "xlmr_large", "xlm-roberta-large", "./xlmr_models/xlm-roberta-large.pt"),
         (12, 12, 768, 3072, "xlmr_base", "xlm-roberta-base", "./xlmr_models/xlm-roberta-base.pt")]:

    for task in ["regression", "bi_classification"]:
        for division in [1, 2,4,8,16]:
            int_size = int(intermediate_size/division)
            for ld in all_lds + ["all"]:
                make_config([ld], 
                            [ld], 
                            [ld], 
                            exp="%s_%s"%(exp_name,int_size),
                            n_layers = n_layers, 
                            n_heads = n_heads,
                            hidden_size = hidden_size,
                            intermediate_size = int_size,
                            max_layer = n_layers - 1,
                            model_name = model_name,
                            task=task,
                            dataset="qe",
                            name=ld, 
                            n_runs=5, 
                            batch_size_per_gpu=8, 
                            learning_rate=1e-6,
                            epochs=50, 
                            eval_interval = 100 if ld=="blog" else 20,
                            checkpoint_path=checkpoint_path,
                            config_prefix="kd_configs")

"""
#kd cfg
#drop top
batch_size=8
learning_rate=1e-4
  
for l in [1,6,12,18]:
    n_layers = l+1
    mapping = (
        ("top", l, {i:i for i in range(l)}, {l:23}),
        ("bottom", 0, {1+i:24-l+i for i in range(l)}, {0:23-l})
        )
    for style, train_layer, encoder_copy_mapping, encoder_sup_mapping in mapping:
        for use_transferset in [True, False]:
            suffix = "_unlabel" if use_transferset else ""
            epochs = 3 if use_transferset else 50
            checkpoint_prefix = "kdmini_%s_%s%s_"%(style, l, suffix)
            kd_cfg_file = "kd_config_%s_%s%s.json"%(style, l, suffix)

            all_lds = ["_".join(ld) for ld in ids["all"]]
            for ld in all_lds + ["all"]:
                make_config(
                        [ld], 
                        [ld], 
                        [ld], 
                        exp="models", 
                        name=ld, 
                        cfg_file_name=kd_cfg_file,
                        checkpoint_prefix=checkpoint_prefix,
                        encoder_copy_mapping=encoder_copy_mapping,
                        encoder_sup_mapping=encoder_sup_mapping,
                        n_layers=n_layers,
                        train_layer=train_layer,
                        use_transferset=use_transferset,
                        n_runs=5, 
                        epochs=epochs,
                        learning_rate=learning_rate,
                        batch_size_per_gpu=batch_size)
"""
