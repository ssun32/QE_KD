import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class QETransformer(nn.Module):

    def __init__(self, model_name, use_layers=None):

        super(QETransformer, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dim = self.transformer.get_input_embeddings().weight.shape[1]

        #number of layers
        n_layers = len(self.transformer.encoder.layer)

        if use_layers is None:
            self.use_layers = {l:1 for l in range(n_layers)}
        else:
            self.use_layers = {l:1 for l in use_layers}

        self.mlp = nn.Sequential(
                        nn.Linear(self.dim, 4*self.dim), 
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(4*self.dim, 1))

        self.loss_fn = torch.nn.MSELoss(reduction="none")

    def forward(self, batch):
        transformer_outputs = self.transformer_forward(self.transformer, **batch[0])

        encodings = transformer_outputs["hidden_states"][-1]
        cls_encoding = encodings[:,0,:]
        predicted_scores = self.mlp(cls_encoding).squeeze()

        return predicted_scores, transformer_outputs


    def transformer_forward(
	qe_instance, 
	transformer_instance,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None):

        self = transformer_instance

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        encoder_extended_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        ### get embedding_output
        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)

        hidden_states = embedding_output

        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.encoder.layer):
            if i in qe_instance.use_layers:
                layer_head_mask = head_mask[i] if head_mask is not None else None
                layer_outputs = layer_module(
                    hidden_states,
                    extended_attention_mask,
                    layer_head_mask,
                    output_attentions=True
                )
                hidden_states = layer_outputs[0]

                all_hidden_states = all_hidden_states + (hidden_states,)
                all_attentions = all_attentions + (layer_outputs[1],)

        return {"embeddings": embedding_output,
                "hidden_states": all_hidden_states, 
                "attentions": all_attentions}
