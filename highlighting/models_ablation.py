"""
(1) BertForHighlightPrediction
(2) BertForHighlightSpanDetection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, KLDivLoss
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import BertModel, BertPreTrainedModel

class BertForHighlightPrediction(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, **model_kwargs):
        super().__init__(config)
        # self.model_args = model_kargs["model_args"]
        self.num_labels = config.num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.tokens_clf = nn.Linear(config.hidden_size, config.num_labels)

        self.tau = model_kwargs.pop('tau', 1)
        self.gamma = model_kwargs.pop('gamma', 1)
        self.soft_labeling = model_kwargs.pop('soft_labeling', False)

        self.init_weights()
        self.softmax = nn.Softmax(dim=-1)
        self.model_kwargs = model_kwargs

    def forward(self,
                input_ids=None, 
                probs=None, # soft-labeling
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        tokens_output = outputs[0]
        highlight_logits = self.tokens_clf(self.dropout(tokens_output))

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            active_loss = attention_mask.view(-1) == 1
            active_logits = highlight_logits.view(-1, self.num_labels)
            active_labels = torch.where(
                    active_loss,
                    labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            loss_ce = loss_fct(active_logits, active_labels)

            loss_kl = 0
            if self.tau != -1:
                with torch.no_grad():
                    model_kd = self.model_kwargs['zero-shot'].to(self.device)
                    outputs_kd = model_kd.forward(
                            input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states,
                            return_dict=return_dict,
                    )
                    probabilities_kl = self.softmax(outputs_kd.logits / self.tau)
                    target_p = probabilities_kl.view(-1, 2)

            if self.soft_labeling:
                loss_fct = KLDivLoss(reduction='sum')
                active_mask = (attention_mask * token_type_ids).view(-1, 1) # BL 1
                n_active = (active_mask == 1).sum()
                active_mask = active_mask.repeat(1, 2) # BL 2
                input_logp = F.log_softmax(active_logits / self.tau, -1) # BL 2
                # target_p = torch.cat(( (1-probs).view(-1, 1), probs.view(-1, 1)), -1) # BL 2
                loss_kl = loss_fct(input_logp, target_p * active_mask) / n_active

            loss = self.gamma * loss_ce + (1-self.gamma) * loss_kl

            # print("Loss:\n")
            # print(loss)
            # print(loss_kl)
            # print(loss_ce)

        return TokenClassifierOutput(
            loss=loss,
            logits=highlight_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def inference(self, batch):

        batch_inputs, batch_info = batch

        for k in batch_inputs:
            batch_inputs[k] = batch_inputs[k].to(self.device)

        with torch.no_grad():
            outputs = self.forward(**batch_inputs)
            probabilities = self.softmax(self.tokens_clf(outputs.hidden_states[-1]))
            predictions = torch.argmax(probabilities, dim=-1)

            # active filtering
            active_tokens = batch_inputs['attention_mask'] == 1
            active_predictions = torch.where(
                active_tokens,
                predictions,
                torch.tensor(-1).type_as(predictions)
            )

            batch_outputs = {
                    "probabilities": probabilities[:, :, 1].detach(), # shape: (batch, length)
                    "active_predictions": predictions.detach(),
                    "active_tokens": active_tokens,
            } 

            return batch_outputs, batch_info

