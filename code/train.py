import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from tqdm import tqdm, trange
import numpy as np
from util import get_args_parser, get_f1_score, set_seed, get_optimizer, get_model, get_tokenizer
from data_processor import IDDataset, IDDataset_prompt
from model import Model_ft, Model_pt


def evaluate(model, val_dataloader, isTest=False):
    model.eval()
    all_labels = []
    all_pred = []

    start_test_time = time.time()
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            labels = batch[-1].numpy().tolist()
            batch = [item.cuda() for item in batch[:2]]
            pred = model(*batch).detach().cpu().numpy().tolist()

            all_pred += pred
            all_labels += labels

        mi_f1, ma_f1 = get_f1_score(all_pred, all_labels)

    if isTest:
        end_test_time = time.time()
        print("***** Test *****")
        print("mi_f1 {}, ma_f1 {}".format(mi_f1, ma_f1))
        print(mi_f1)
        print("test time cost", end_test_time - start_test_time)

    return mi_f1, ma_f1


def train(args, model, train_dataset, val_dataset, test_dataset):
    val_batch_size = args.per_gpu_eval_batch_size
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=val_batch_size)

    train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

    val_sampler = SequentialSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=val_batch_size)

    optimizer, scheduler = get_optimizer(model, train_dataloader)
    mx_res = 0.0
    hist_mi_f1 = []
    hist_ma_f1 = []
    start_train_time = time.time()
    path = args.output_dir + "/"
    os.makedirs(path, exist_ok=True)

    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        model.train()
        model.zero_grad()
        tr_loss = 0.0
        global_step = 0
        for step, batch in enumerate(train_dataloader):

            batch = [item.cuda() for item in batch]
            loss, logits = model(*batch)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1
                sys.stdout.write('step: {0:4} | loss: {1:2.6f}%'.format(step + 1, tr_loss / global_step) + '\r')
                sys.stdout.flush()

        mi_f1, ma_f1 = evaluate(model, val_dataloader)

        print("***** Epoch {} *****: mi_f1 {}, ma_f1 {}".format(epoch, mi_f1, ma_f1))
        hist_mi_f1.append(mi_f1)
        hist_ma_f1.append(ma_f1)
        if mi_f1 >= mx_res:
            mx_res = mi_f1
            torch.save(model.state_dict(), args.output_dir + "/" + 'best_parameter' + ".pkl")
        if epoch == args.num_train_epochs - 1:
            torch.save(model.state_dict(), args.output_dir + "/" + 'final_parameter' + ".pkl")
    end_train_time = time.time()
    print(hist_mi_f1)
    print(mx_res)
    print("train time cost", end_train_time - start_train_time)

    model.load_state_dict(torch.load(args.output_dir + "/" + 'best_parameter' + ".pkl"))
    mi_f1, ma_f1 = evaluate(model, test_dataloader, True)

    return mi_f1, ma_f1


if __name__ == "__main__":
    args = get_args_parser()
    set_seed(args.seed)
    tuning_type = args.tuning_type
    Dataset = IDDataset if tuning_type == "ft" else IDDataset_prompt
    tokenizer = get_tokenizer()
    isTest = args.isTest
    test_dataset = Dataset(
        path=args.data_dir,
        name='test.csv',
        tokenizer=tokenizer,
        format="csv"
    )
    model = get_model(Model_ft, num_labels=test_dataset.num_labels) if tuning_type == "ft" else get_model(Model_pt,
                                                                                                          mask_token_id=tokenizer.mask_token_id,
                                                                                                          label_name=test_dataset.lang_names_ids)
    if not args.isTest:
        train_dataset = Dataset(
            path=args.data_dir,
            name='train.csv',
            tokenizer=tokenizer,
            format="csv"
        )

        val_dataset = Dataset(
            path=args.data_dir,
            name='valid.csv',
            tokenizer=tokenizer,
            format="csv"
        )

        train(args, model, train_dataset, val_dataset, test_dataset)

    else:
        # model.load_state_dict(torch.load(args.output_dir + "/" + 'best_parameter' + ".pkl"))
        val_batch_size = args.per_gpu_eval_batch_size
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=val_batch_size)
        evaluate(model, test_dataloader, isTest)
