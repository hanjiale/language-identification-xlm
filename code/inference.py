import os
import sys
import time
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from util import get_args_parser, get_model, get_tokenizer
from data_processor import IDDataset, IDDataset_prompt
from model import Model_ft, Model_pt


def evaluate(model, val_dataset, val_dataloader, pred_file_name):
    model.eval()
    all_pred = []
    lang_names = val_dataset.lang_names
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            batch = [item.cuda() for item in batch[:2]]
            pred = model(*batch).detach().cpu().numpy().tolist()

            all_pred += pred

    datas = val_dataset.datas
    assert len(datas) == len(all_pred)
    print("******************Language Identification******************")
    with open(pred_file_name, 'w') as f:
        for line, pred in zip(datas, all_pred):
            pred_name = lang_names[pred]
            f.write(pred_name + ":" + line)
            print(pred_name + ":" + line)


if __name__ == "__main__":
    start_test_time = time.time()
    args = get_args_parser()
    tuning_type = args.tuning_type
    Dataset = IDDataset if tuning_type == "ft" else IDDataset_prompt
    tokenizer = get_tokenizer()

    if args.name.endswith(".txt"):
        test_dataset = Dataset(
            path=args.data_dir,
            name=args.name,
            tokenizer=tokenizer,
            format="txt"
        )
        val_batch_size = args.per_gpu_eval_batch_size
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=val_batch_size)
    else:
        raise NotImplementedError("This file format is not yet supported!")

    model = get_model(Model_ft, num_labels=test_dataset.num_labels) if tuning_type == "ft" else get_model(Model_pt, mask_token_id=tokenizer.mask_token_id, label_name=test_dataset.lang_names_ids)
    model.load_state_dict(torch.load(args.output_dir + "/" + 'best_parameter' + ".pkl"))
    evaluate(model, test_dataset, test_dataloader, args.output_dir+"/"+args.name.split(".")[0]+"_pred"+"."+args.name.split(".")[1])

    end_test_time = time.time()
    print("test time cost", end_test_time - start_test_time)
    print("please check the predicted outputs in ", args.output_dir+"/"+args.name.split(".")[0]+"_pred"+"."+args.name.split(".")[1])

