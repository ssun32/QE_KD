import os
import json
from glob import glob
n_runs=5
batch_size_per_gpu=8
 
ids = {
 "all":[("en","de"),("en","zh"),("ro","en"),("et","en"),("si","en"),("ne","en"), ("ru", "en")],
 "sharing_src":[("en","de"), ("en","zh")],
 "sharing_tgt":[("ro","en"),("et","en"),("si","en"),("ne","en"), ("ru", "en")],
 "random2":[("en","zh"),("ro","en")],
 "0shot_no_ende":[("en","zh"),("ro","en"),("et","en"),("si","en"),("ne","en"), ("ru","en")],
 "0shot_no_enzh":[("en","de"),("ro","en"),("et","en"),("si","en"),("ne","en"), ("ru","en")],
 "0shot_no_roen":[("en","de"),("en","zh"),("et","en"),("si","en"),("ne","en"), ("ru","en")],
 "0shot_no_eten":[("en","de"),("en","zh"),("ro","en"),("si","en"),("ne","en"), ("ru","en")],
 "0shot_no_sien":[("en","de"),("en","zh"),("ro","en"),("et","en"),("ne","en"), ("ru","en")],
 "0shot_no_neen":[("en","de"),("en","zh"),("ro","en"),("et","en"),("si","en"), ("ru","en")],
 "0shot_no_ruen":[("en","de"),("en","zh"),("ro","en"),("et","en"),("si","en"), ("ne","en")],
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
                n_layers=13,
                train_layer = 1,
                copy_embeddings=False,
                encoder_copy_mapping=None,
                encoder_sup_mapping=None,
                copy_mlp=False,
                use_transferset=False,
                cfg_file_name="config.json",
                checkpoint_prefix="kdmini_",
                name="all",
                n_runs=1, 
                epochs=100,
                learning_rate=1e-4,
                batch_size_per_gpu=24,
                checkpoint_path=None,
                sample_dict={}):

    for run in range(n_runs):
        config = {
                  "model_name":"xlm-roberta-large",
                  "epochs":epochs,
                  "use_layers": use_layers,
                  "n_layers": n_layers,
                  "train_layer":train_layer,
                  "copy_embeddings":copy_embeddings,
                  "encoder_copy_mapping":encoder_copy_mapping,
                  "encoder_sup_mapping":encoder_sup_mapping,
                  "copy_mlp":copy_mlp,
                  "use_transferset":use_transferset,
                  "batch_size_per_gpu": batch_size_per_gpu,
                  "learning_rate":learning_rate,
                  "checkpoint_path": "configs/models/%s/run%s/best.pt"%(name, run),
                  "checkpoint_prefix": checkpoint_prefix,
                  "accum_grad": 1,
                  "eval_interval": 20 * len(ids[train[0]]),
                  "loss_fn": "mse",
                  "train":[],
                  "dev":[],
                  "test":[]
                }
        output_dir = "configs/%s/%s/run%s/" % (exp, name, run)
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, cfg_file_name), "w") as fout:
            config["output_dir"] =  output_dir

            for split_name, split in [("train", train), ("dev", dev), ("test", test)]:
                for id in split:
                    tsv_file = get_files(ids[id], split=split_name)
                    if len(tsv_file) > 1:
                        id = "all"
                    config[split_name].append({"id": id, "tsv_file":tsv_file})
            print(json.dumps(config, indent=4), file=fout)

#single languages
all_lds = ["_".join(ld) for ld in ids["all"]]
for ld in all_lds + ["all"]:
    make_config([ld], [ld], [ld], exp="models_linformer", n_layers = 24, name=ld, n_runs=5, batch_size_per_gpu=8, learning_rate=1e-6)

#kd cfg
#drop top
batch_size=8
learning_rate=1e-4
copy_embeddings=True
copy_mlp=True
  
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
                        copy_embeddings=copy_embeddings,
                        encoder_copy_mapping=encoder_copy_mapping,
                        encoder_sup_mapping=encoder_sup_mapping,
                        copy_mlp=copy_mlp,
                        n_layers=n_layers,
                        train_layer=train_layer,
                        use_transferset=use_transferset,
                        n_runs=5, 
                        epochs=epochs,
                        learning_rate=learning_rate,
                        batch_size_per_gpu=batch_size)
