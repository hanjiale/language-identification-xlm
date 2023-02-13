import torch
import torch.nn as nn
from util import get_model_classes, get_args
import torch.nn.functional as F
import random


class Model_ft(torch.nn.Module):

    def __init__(self, args, num_labels=None, mask_token_id=None, label_name=None):

        super().__init__()
        assert args.tuning_type == "ft"
        model_classes = get_model_classes()
        model_class = model_classes[args.tuning_type]
        self.model = model_class['model'].from_pretrained(args.model_name_or_path)

        self.loss_fct = nn.CrossEntropyLoss()
        self.hidden_size = self.model.config.hidden_size
        self.classifier = nn.Linear(self.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, label_id=None):

        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask
                             )
        logits = self.classifier(outputs[1])

        if label_id != None:
            loss = self.loss_fct(logits.view(-1, logits.size(-1)), label_id.view(-1))
            pred = torch.argmax(logits, -1)
            return loss, pred
        else:
            pred = torch.argmax(logits, -1)
            return pred


class Model_pt(torch.nn.Module):

    def __init__(self, args, num_labels=None, mask_token_id=None, label_name=None):

        super().__init__()
        assert args.tuning_type == "pt"
        model_classes = get_model_classes()
        model_class = model_classes[args.tuning_type]
        self.model = model_class['model'].from_pretrained(args.model_name_or_path)

        self.loss_fct = nn.CrossEntropyLoss()
        self.label_name = [l[0] for l in label_name]
        self.mask_token_id = mask_token_id

    def forward(self, input_ids, attention_mask, label_id=None):

        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             return_dict=True,
                             output_hidden_states=True,
                             )
        logits = outputs.logits
        logits = logits * torch.where(input_ids==self.mask_token_id, torch.ones_like(input_ids), torch.zeros_like(input_ids)).unsqueeze(-1)
        logits = torch.mean(logits, dim=1)
        logits = logits[:, self.label_name]
        if label_id != None:
            loss = self.loss_fct(logits.view(-1, logits.size(-1)), label_id.view(-1))
            pred = torch.argmax(logits, -1)
            return loss, pred
        else:
            pred = torch.argmax(logits, -1)
            return pred
