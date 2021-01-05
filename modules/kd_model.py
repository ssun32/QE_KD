import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.qe_transformer import QETransformer

class KDModel(nn.Module):
    def __init__(
            self,
            teacher_model,
            student_model,
            encoder_sup_mapping={},
	    encoder_copy_mapping={}):

        super().__init__()
        self.teacher_model = teacher_model
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.teacher_model.eval()

        self.student_model = student_model
        self.student_model.train()

        self.encoder_sup_mapping = encoder_sup_mapping
        self.encoder_copy_mapping = encoder_copy_mapping

        self.copy()
        self.freeze_parameters()


    def forward(self, **kwargs):
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**kwargs)
        student_outputs = self.student_model(**kwargs)

        return {
                #"teacher_outputs": teacher_outputs,
                #"student_outputs": student_outputs,
                "student_output": student_outputs["qe_scores"],
                "loss": self.calc_loss(teacher_outputs, student_outputs)}


    def calc_loss(self, teacher_outputs, student_outputs):
        t_hs = teacher_outputs["encoder_outputs"]
        s_hs = student_outputs["encoder_outputs"]

        mse_loss = nn.MSELoss(reduction="none")

        loss = mse_loss(teacher_outputs["qe_scores"], student_outputs["qe_scores"])

        for src, tgt in self.encoder_sup_mapping.items():
            src, tgt = int(src), int(tgt)
            t_encoder_fc = t_hs[src]["fc_outputs"]
            s_encoder_fc = s_hs[tgt]["fc_outputs"]
            attn_mask = s_hs[tgt]["att_mask"]

            for n in ["input", "fc2_output"]:
                loss += (mse_loss(t_encoder_fc[n], s_encoder_fc[n]).mean(dim=-1) * attn_mask).sum(dim=-1) / attn_mask.sum(dim=-1)

        return loss.mean()


    def freeze_parameters(self):
        for param in self.student_model.parameters():
            param.requires_grad = False
        for encoder_layer in self.student_model.encoder_layers:
            for name,param in encoder_layer.fc.named_parameters():
                if "proj" in name:
                    param.requires_grad = True

    def copy(self):
        print("Copying embedding...")
        self.student_model.embeddings.load_state_dict(
                self.teacher_model.embeddings.state_dict()
                )
        print("Copying mlp...")
        self.student_model.mlp.load_state_dict(
            self.teacher_model.mlp.state_dict()
            )

        if self.encoder_copy_mapping:
            for src, tgt in self.encoder_copy_mapping.items():
                src, tgt = int(src), int(tgt)
                print("Copying encoder layer %s from teacher to encoder layer %s of student" % (src, tgt))
                self._copy_encoder(
                    self.teacher_model.encoder_layers[src],
                    self.student_model.encoder_layers[tgt])

    def _copy_encoder(self, s_encoder_layer, t_encoder_layer):
        #copy multiheadattention
        t_encoder_layer.MHattention.load_state_dict(
                s_encoder_layer.MHattention.state_dict())

        #copy feedforward
        t_encoder_layer.fc.load_state_dict(
                s_encoder_layer.fc.state_dict(), strict=False)
