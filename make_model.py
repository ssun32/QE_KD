import torch
from transformers import XLMRobertaConfig, XLMRobertaModel, AutoModel
from modules.qe_transformer import QETransformer

model_name = "xlm-roberta-base"
print("loading old model...")
old_model = AutoModel.from_pretrained(model_name)
print("loading new model...")
n_layers = 12
new_model = QETransformer(
		vocab_size = 250002,
		hidden_size = 768,
		intermediate_size = 768*4,
		n_heads = 12,
		n_layers = n_layers, 
                model_name = model_name)

def copy(old_model, new_model):
    print("Copying embedding...")
    new_model.embeddings.load_state_dict(
	    old_model.embeddings.state_dict()
	    )

    def copy_encoder(s_encoder_layer, t_encoder_layer):
	#copy multiheadattention
        s_encoder_layer.MHattention.query.load_state_dict(
                t_encoder_layer.attention.self.query.state_dict())
        s_encoder_layer.MHattention.key.load_state_dict(
                t_encoder_layer.attention.self.key.state_dict())
        s_encoder_layer.MHattention.value.load_state_dict(
                t_encoder_layer.attention.self.value.state_dict())
        s_encoder_layer.MHattention.fc.load_state_dict(
                t_encoder_layer.attention.output.dense.state_dict())
        s_encoder_layer.MHattention.ln.load_state_dict(
                t_encoder_layer.attention.output.LayerNorm.state_dict())

        #copy feedforward
        s_encoder_layer.fc.fc1[0].load_state_dict(
                t_encoder_layer.intermediate.dense.state_dict())
        s_encoder_layer.fc.fc2[0].load_state_dict(
                t_encoder_layer.output.dense.state_dict())

        s_encoder_layer.fc.ln.load_state_dict(
                t_encoder_layer.output.LayerNorm.state_dict())

    for i in range(n_layers):
        print("Copying encoder layer %s" % i)
        copy_encoder(
            new_model.encoder_layers[i], 
            old_model.encoder.layer[i])


copy(old_model, new_model)
torch.save(new_model.state_dict(), "xlmr_models/%s.pt"%model_name)
