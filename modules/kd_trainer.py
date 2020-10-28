import os, sys
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr
from tqdm import tqdm
import logging

class KDTrainer(object):
    def __init__(self, 
                 teacher_model, 
                 student_model,
                 output_dir,
                 train_dataloader,
                 dev_dataloader,
                 test_dataloader,
                 transfer_dataloader,
                 copy_embeddings=False,
                 encoder_copy_mapping={},
                 encoder_sup_mapping={},
                 copy_mlp=False,
                 checkpoint_prefix = "kdmini_",
                 use_transferset=False,
                 learning_rate=1e-6,
                 epochs=20,
                 eval_interval = 500):

        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.test_dataloader = test_dataloader
        self.transfer_dataloader = transfer_dataloader

        self.copy_embeddings = copy_embeddings
        self.copy_mlp = copy_mlp
        self.encoder_copy_mapping = encoder_copy_mapping
        self.encoder_sup_mapping = encoder_sup_mapping

        self.use_transferset=use_transferset

        #intialize model
        self.gpu=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.teacher_model = teacher_model
        self.teacher_model = torch.nn.DataParallel(self.teacher_model)
        self.teacher_model = self.teacher_model.to(self.gpu)
        self.teacher_model.eval()

        self.student_model = student_model
        self.student_model = torch.nn.DataParallel(self.student_model)
        self.student_model = self.student_model.to(self.gpu)
        self.student_model.train()

        n_layers = len(self.student_model.module.encoder_layers)

        for param in self.teacher_model.parameters():
            param.requires_grad = False

        #initialize optimizer
        self.optimizer = torch.optim.AdamW(self.student_model.parameters(), lr=learning_rate)

        self.epochs = epochs
        self.eval_interval = eval_interval

        self.log_file = os.path.join(output_dir, "%slog"%checkpoint_prefix)
        self.best_model_path = os.path.join(output_dir, "%sbest.pt"%checkpoint_prefix)
        self.model_checkpoint_path = os.path.join(output_dir, "%scheckpoint.pt"%checkpoint_prefix)

        #initialize training parameters
        self.cur_epoch = 0
        self.global_steps = 0
        self.best_eval_result = -100000
        self.early_stop_count = 0
        self.stages = []

        self.stages.append("embeddings")
        for i in range(len(self.student_model.module.encoder_layers)):
            self.stages.append("layer.%s"%i)
        self.stages.append("mlp")

        self.freeze_parameters()
        self.init_logging()

        #copy parameters
        self.copy()

    def init_logging(self):
        logging.basicConfig(filename=self.log_file,
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)

    def copy(self):
        if self.copy_embeddings:
            print("Copying embedding...")
            self.student_model.module.embeddings.load_state_dict(
                    self.teacher_model.module.transformer.embeddings.state_dict()
                    )
        if self.copy_mlp:
            print("Copying mlp...")
            self.student_model.module.mlp.load_state_dict(
                    self.teacher_model.module.mlp.state_dict()
                    )
        if self.encoder_copy_mapping:
            def copy_encoder(s_encoder_layer, t_encoder_layer):
                #copy multiheadattention
                s_encoder_layer.MHattention.query.load_state_dict(
                        t_encoder_layer.attention.self.query.state_dict())
                s_encoder_layer.MHattention.key.load_state_dict(
                        t_encoder_layer.attention.self.key.state_dict())
                s_encoder_layer.MHattention.value.load_state_dict(
                        t_encoder_layer.attention.self.value.state_dict())
                s_encoder_layer.MHattention.dense.load_state_dict(
                        t_encoder_layer.attention.output.dense.state_dict())
                s_encoder_layer.MHattention.layer_norm.load_state_dict(
                        t_encoder_layer.attention.output.LayerNorm.state_dict())

                #copy feedforward
                s_encoder_layer.fc[0].load_state_dict(
                        t_encoder_layer.intermediate.dense.state_dict())
                s_encoder_layer.fc[2].load_state_dict(
                        t_encoder_layer.output.dense.state_dict())

                s_encoder_layer.ln.load_state_dict(
                        t_encoder_layer.output.LayerNorm.state_dict())

            for src, tgt in self.encoder_copy_mapping.items():
                print("Copying encoder layer %s" % src)
                copy_encoder(
                    self.student_model.module.encoder_layers[int(src)], 
                    self.teacher_model.module.transformer.encoder.layer[int(tgt)])

    def forward(self, model, batch):
        batch = [{k:v.to(self.gpu) for k,v in b.items()} for b in batch]
        predicted_scores, transformer_outputs = model(batch)
        return predicted_scores, transformer_outputs

    def calc_loss(self, 
                 gold_outputs,
                 t_predicted_outputs,
                 s_predicted_outputs,
                 t_transformer_outputs, 
                 s_transformer_outputs,
                 attention_mask,
                 cur_stage):

        t_hs = t_transformer_outputs["hidden_states"]
        s_hs = s_transformer_outputs["hidden_states"]

        t_att = t_transformer_outputs["attentions"]
        s_att = s_transformer_outputs["log_attentions"]

        mse_loss = torch.nn.MSELoss(reduction="none")
        kl_loss = torch.nn.KLDivLoss(reduction="none")

        if cur_stage == "mlp":
            loss = torch.nn.MSELoss(reduction="none")(s_predicted_outputs, gold_outputs)

        elif cur_stage == "embeddings":
            loss = (mse_loss(s_hs[0], t_hs[0]).mean(dim=-1) * attention_mask).sum(dim=-1) / attention_mask.sum(dim=-1)

        elif "layer" in cur_stage:
            s_layer_n = int(cur_stage.split(".")[-1])
            t_layer_n = self.encoder_sup_mapping[str(s_layer_n)]
            loss = (mse_loss(s_hs[s_layer_n], t_hs[t_layer_n]).mean(dim=-1) * attention_mask).sum(dim=-1) / attention_mask.sum(dim=-1)

            #attention_mask = attention_mask.unsqueeze(1)
            #att_loss = kl_loss(s_att[s_layer_n], t_att[t_layer_n]) * attention_mask.unsqueeze(-1) * attention_mask.unsqueeze(1)
            #att_loss = att_loss.sum(dim=-1) / attention_mask.sum(dim=-1).unsqueeze(-1)
            #att_loss = att_loss.sum(dim=-1) / attention_mask.sum(dim=-1)
            #att_loss = att_loss.mean(dim=-1)
            #loss += att_loss

            loss += torch.nn.MSELoss(reduction="none")(s_predicted_outputs, t_predicted_outputs)
            #loss += torch.nn.MSELoss(reduction="none")(s_predicted_outputs, gold_outputs)

        loss = loss.mean()
        return loss

    def freeze_parameters(self):
        for param in self.student_model.parameters():
            param.requires_grad = False

    #unfreeze parameters related to current stage
    def unfreeze_parameters(self, cur_stage):
        if "layer" in cur_stage:
            layer_n = int(cur_stage.split(".")[-1])
            for param in self.student_model.module.encoder_layers[layer_n].parameters():
                param.requires_grad = True
        elif cur_stage == "embeddings":
            for param in self.student_model.module.embeddings.parameters():
                param.requires_grad = True
        elif cur_stage == "mlp":
            for param in self.student_model.module.mlp.parameters():
                param.requires_grad = True

        for name, param in self.student_model.named_parameters():
            if param.requires_grad:
                print(name, param.shape)

    def multi_stage_train(self):
        prev_stage = None
        while self.stages:
            #init optimizer and set learning rate
            learning_rate = 1e-6 if self.stages[0] == "mlp" else 1e-3
            self.optimizer = torch.optim.AdamW(self.student_model.parameters(), lr=learning_rate)

            if prev_stage is not None:
                elf.student_model.module.load_state_dict(torch.load(self.best_model_path.replace("best", "%s.best"%prev_stage)))

            self.cur_epoch = 0
            self.global_steps = 0
            self.best_eval_result = -100000
            self.early_stop_count = 0
            self.train(self.stages[0])

            #move to next stage
            prev_stage = self.stages[0]
            self.stages = self.stages[1:]


    #evaluate on dev dataset, return pearson_correlation and mse
    def eval(self, cur_stage = None):
        if cur_stage is None:
            cur_stage = self.stages[0]

        self.student_model.eval()
        total_loss = 0
        total_batches = 0
        all_predicted_scores, all_actual_scores = [], []
        with torch.no_grad():
            for batch, z_scores, da_scores in tqdm(self.dev_dataloader):

                attention_mask = batch[0]["attention_mask"].to(self.gpu)
                s_predicted_outputs, s_transformer_outputs  = self.forward(self.student_model, batch)
                t_predicted_outputs, t_transformer_outputs  = self.forward(self.teacher_model, batch)

                if not torch.isnan(s_predicted_outputs).any() and not torch.isnan(t_predicted_outputs).any(): 
                    loss = self.calc_loss(torch.tensor(z_scores).to(self.gpu),
                                          t_predicted_outputs,
                                          s_predicted_outputs,
                                          t_transformer_outputs, 
                                          s_transformer_outputs,
                                          attention_mask,
                                          cur_stage)

                    total_loss += loss.item() * s_predicted_outputs.size(0)
                    total_batches += s_predicted_outputs.size(0)

                s_predicted_outputs[torch.isnan(s_predicted_outputs)] = 0

                all_predicted_scores += s_predicted_outputs.flatten().tolist()
                all_actual_scores += z_scores

        all_predicted_scores = np.array(all_predicted_scores)
        all_actual_scores = np.array(all_actual_scores)

        pearson_correlation = pearsonr(all_predicted_scores, all_actual_scores)[0]
        mse = np.square(np.subtract(all_predicted_scores, all_actual_scores)).mean()
        self.student_model.train()
        return pearson_correlation, mse, total_loss/total_batches

    #evaluate on test dataset, return pearson_correlation and mse
    def test(self):
        self.student_model.eval()
        all_predicted_scores, all_actual_scores = [], []
        with torch.no_grad():
            for batch, z_scores, da_scores in tqdm(self.test_dataloader):
                s_predicted_outputs, _  = self.forward(self.student_model, batch)
                s_predicted_outputs[torch.isnan(s_predicted_outputs)] = 0
                all_predicted_scores += s_predicted_outputs.flatten().tolist()
                all_actual_scores += z_scores

        all_predicted_scores = np.array(all_predicted_scores)
        all_actual_scores = np.array(all_actual_scores)

        pearson_correlation = pearsonr(all_predicted_scores, all_actual_scores)[0]
        mse = np.square(np.subtract(all_predicted_scores, all_actual_scores)).mean()
        self.student_model.train()
        return pearson_correlation, mse

    #main train loop
    def train(self, cur_stage):
        #init
        self.unfreeze_parameters(cur_stage)
        if self.use_transferset and cur_stage != "mlp":
            dataloader = self.transfer_dataloader
        else:
            dataloader = self.train_dataloader

        for epoch in range(self.cur_epoch, self.epochs):
            print("Epoch ", epoch)
            total_loss = 0
            total_output_loss = 0
            total_batches = 0
            for batch, z_scores, da_scores in tqdm(dataloader):
                z_scores = torch.tensor(z_scores).to(self.gpu)
                attention_mask = batch[0]["attention_mask"].to(self.gpu)

                t_predicted_outputs, t_transformer_outputs = self.forward(self.teacher_model, batch)
                s_predicted_outputs, s_transformer_outputs = self.forward(self.student_model, batch)
            
                if torch.isnan(s_predicted_outputs).any() or torch.isnan(t_predicted_outputs).any(): 
                    continue

                loss = self.calc_loss(z_scores,
                                      t_predicted_outputs,
                                      s_predicted_outputs,
                                      t_transformer_outputs, 
                                      s_transformer_outputs,
                                      attention_mask,
                                      cur_stage)

                loss.backward()
                self.optimizer.step()
                self.student_model.zero_grad()

                cur_batch_size = s_predicted_outputs.size(0)
                total_batches += cur_batch_size
                total_loss += loss.item() * cur_batch_size

                avg_loss = total_loss/total_batches
                log = "Stage:%s Epoch %s Global steps: %s Train loss: %.4f\n" %(cur_stage, epoch, self.global_steps, avg_loss)

                self.global_steps += 1
                if self.global_steps % self.eval_interval == 0:
                   
                    pearson_correlation, mse, dev_loss = self.eval(cur_stage)
                    log += "Stage:%s Dev loss: %.4f mse: %.4f r:%.4f\n" % (cur_stage, dev_loss, mse, pearson_correlation)
                    logging.info(log)

                    if True or cur_stage == "mlp":
                        if pearson_correlation > self.best_eval_result:
                            self.best_eval_result = pearson_correlation
                            torch.save(self.student_model.module.state_dict(), self.best_model_path.replace("best", "%s.best"%cur_stage))
                            self.early_stop_count = 0
                        else:
                            self.early_stop_count += 1
                    else:
                        if -dev_loss > self.best_eval_result:
                            self.best_eval_result = -dev_loss
                            self.early_stop_count = 0
                            torch.save(self.student_model.module.state_dict(), self.best_model_path.replace("best", "%s.best"%cur_stage))
                        else:
                            self.early_stop_count += 1


                    if self.early_stop_count > 20:
                        return 

            #save checkpoint
            self.cur_epoch += 1
            torch.save(self, self.model_checkpoint_path)
