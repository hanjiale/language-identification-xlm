import argparse
import torch
import transformers
from transformers import XLMRobertaConfig, XLMRobertaTokenizer, XLMRobertaModel, XLMRobertaForMaskedLM, AdamW, get_linear_schedule_with_warmup
from collections import Counter
import numpy as np
import random


_LANG_LABLES = ["ar", "bg", "de", "el", "en", "es", "fr", "hi", "it", "ja", "nl", "pl", "pt", "ru", "sw",
                "th", "tr", "ur", "vi", "zh"]
_LANG_NAMES = ["Arabic", "Bulgarian", "German", "Greek", "English", "Spanish", "French", "Hindi", "Italian",
               "Japanese", "Dutch", "Polish", "Portuguese", "Russian", "Swahili", "Thai", "Turkish", "Urdu",
               "Vietnamese", "Chinese"]

_GLOBAL_ARGS = None

_MODEL_CLASSES = {
    'pt': {
        'config': XLMRobertaConfig,
        'tokenizer': XLMRobertaTokenizer,
        'model': XLMRobertaForMaskedLM,
    },
    'ft': {
        'config': XLMRobertaConfig,
        'tokenizer': XLMRobertaTokenizer,
        'model': XLMRobertaModel,
    }
}


def get_args_parser():

    parser = argparse.ArgumentParser(description="Command line interface for Relation Extraction.")

    # Required parameters
    parser.add_argument("--data_dir", type=str, required=True,
                        help="The input data dir. Should contain the data files for the task.")
    parser.add_argument("--name", type=str,
                        help="The input file name.")
    parser.add_argument("--tuning_type", default="pt", type=str, required=True, choices=_MODEL_CLASSES.keys(),
                        help="The type of tuning methods, fine-tuning or prompt tuning.")
    parser.add_argument("--model_name_or_path", default="xlm-roberta-base", type=str, required=True,
                        help="Path to the pre-trained model or shortcut name")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written")
    parser.add_argument("--isTest", action='store_true',
                        help="only test or not.")

    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after tokenization for PET. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for PET training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for PET evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass in PET.")
    parser.add_argument("--num_train_epochs", default=3, type=float,
                        help="Total number of training epochs to perform in PET.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform in PET. Override num_train_epochs.")
    parser.add_argument("--load_ckpt_path", default="", type=str,
                        help="")

    # Other optional parameters
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where to store the pre-trained models downloaded from S3.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")

    parser.add_argument("--weight_decay", default=1e-5, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=0.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--dropout_prob", type=float, default=0.2)

    args = parser.parse_args()
    args.n_gpu = torch.cuda.device_count()

    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args
    return args

def get_args():
    return _GLOBAL_ARGS

def get_model_classes():
    return _MODEL_CLASSES

def get_labels():
    return _LANG_LABLES

def get_label_names():
    return _LANG_NAMES

def get_f1_score(output, label):
    num_labels = max(label) + 1
    correct_by_relation = Counter()
    guess_by_relation = Counter()
    gold_by_relation = Counter()

    for i in range(len(output)):
        guess = output[i]
        gold = label[i]

        guess_by_relation[guess] += 1
        gold_by_relation[gold] += 1
        if gold == guess:
            correct_by_relation[gold] += 1

    f1_by_relation = Counter()
    recall_by_relation = Counter()
    prec_by_relation = Counter()
    for i in range(num_labels):
        recall = 0
        if gold_by_relation[i] > 0:
            recall = correct_by_relation[i] / gold_by_relation[i]
        precision = 0
        if guess_by_relation[i] > 0:
            precision = correct_by_relation[i] / guess_by_relation[i]
        if recall + precision > 0:
            f1_by_relation[i] = 2 * recall * precision / (recall + precision)
        else:
            f1_by_relation[i] = 0.
        recall_by_relation[i] = recall
        prec_by_relation[i] = precision

    micro_f1 = 0
    if sum(guess_by_relation.values()) != 0 and sum(correct_by_relation.values()) != 0:
        recall = sum(correct_by_relation.values()) / sum(gold_by_relation.values())
        prec = sum(correct_by_relation.values()) / sum(guess_by_relation.values())
        micro_f1 = 2 * recall * prec / (recall + prec)

    return micro_f1, f1_by_relation

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_optimizer(model, train_dataloader):

    args = get_args()
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    cur_model = model.module if hasattr(model, 'module') else model

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in cur_model.model.named_parameters() if not any(nd in n for nd in no_decay)],'weight_decay': args.weight_decay},
        {'params': [p for n, p in cur_model.model.named_parameters() if any(nd in n for nd in no_decay)],'weight_decay': 0.0},
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=t_total//10, num_training_steps=t_total)


    return optimizer, scheduler

def get_model(Model, num_labels=None, mask_token_id=None, label_name=None):
    args = get_args()
    model = Model(args, num_labels=num_labels, mask_token_id=mask_token_id, label_name=label_name)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.cuda()
    return model

def get_tokenizer(special=[]):
    args = get_args()
    model_classes = get_model_classes()
    model_config = model_classes[args.tuning_type]
    tokenizer = model_config['tokenizer'].from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None, use_fast=False)
    tokenizer.add_tokens(special)
    return tokenizer