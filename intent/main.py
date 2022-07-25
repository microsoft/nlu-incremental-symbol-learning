# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import pathlib 
import json 
import pdb 
import random
from collections import defaultdict

path_to_min_pair_utils = pathlib.Path(__file__).resolve().parent.parent.joinpath("minimal_pair_utils")
import sys
sys.path.insert(0, str(path_to_min_pair_utils))

from tqdm import tqdm 
import torch
from torch import optim 
import numpy as np 
from datasets import load_dataset 

from model import Classifier
from data import (batchify, 
                  batchify_min_pair, 
                  batchify_double_in_batch, 
                  batchify_double_in_data, 
                  batchify_by_source_trigger, 
                  batchify_mask_source_trigger, 
                  batchify_weight_source_trigger, 
                  batchify_sample_source_trigger,
                  split_by_intent, 
                  random_split)

from dro_loss import GroupDROLoss
def get_accuracy(pred, true, intent_of_interest = None):
    pred_classes = torch.argmax(pred, dim=1).detach().cpu()
    true = true.detach().cpu() 
    correct = torch.sum(pred_classes == true) 
    accuracy = float(correct) / true.shape[0]

    if intent_of_interest is not None: 
        true_idxs = true == intent_of_interest
        pred_of_interest = pred_classes[true_idxs]
        true_of_interest = true[true_idxs]
        intent_correct = torch.sum(pred_of_interest == true_of_interest) 
        if true_of_interest.shape[0] == 0:
            intent_accuracy = 0.0
        else:
            intent_accuracy = float(intent_correct) / true_of_interest.shape[0]
    else:
        intent_accuracy = 0.0

    return accuracy, intent_accuracy

def train_epoch(model, train_data, loss_fxn, optimizer, intent_of_interest):
    all_loss = []
    all_accs = []
    intent_accs = []
    model.train() 
    for batch in tqdm(train_data):
        optimizer.zero_grad() 
        pred_classes = model(batch)
        true_classes = batch['label']
        acc, intent_acc = get_accuracy(pred_classes, true_classes, intent_of_interest)
        all_accs.append(acc)
        intent_accs.append(intent_acc)
        if "weight" in batch.keys():
            loss_by_example = loss_fxn(pred_classes, true_classes)
            weighted_loss = loss_by_example * batch['weight']
            loss = torch.mean(weighted_loss)
        else:
            loss = loss_fxn(pred_classes, true_classes)

        loss.backward()
        optimizer.step()
        all_loss.append(loss.item())
    return np.mean(all_loss), np.mean(all_accs), np.mean(intent_accs)

def eval_epoch(model, eval_data, loss_fxn, intent_of_interest = None, output_individual_preds = False):
    all_accs = []
    all_loss = []
    intent_accs = []
    model.eval() 
    if output_individual_preds:
        individ_preds = [] 
    else:
        individ_preds = None
    with torch.no_grad():
        for batch in tqdm(eval_data):
            pred_classes = model(batch)
            true_classes = batch['label']
            acc, intent_acc = get_accuracy(pred_classes, true_classes, intent_of_interest)
            if output_individual_preds:
                for i, (pred_class, true_class) in enumerate(zip(pred_classes, true_classes)): 
                    input = batch['input_str'][i]
                    individ_preds.append({"input": input, "pred": torch.softmax(pred_class,dim=0).detach().cpu().numpy().tolist(), "true": true_class.detach().cpu().numpy().tolist()})
            all_accs.append(acc)
            intent_accs.append(intent_acc)
            loss = loss_fxn(pred_classes, true_classes)
            all_loss.append(loss.item())
    return np.mean(all_loss), np.mean(all_accs), np.mean(intent_accs), individ_preds

def generate_lookup_table(train_data, intent_of_interest):
    train_src, train_tgt, train_idx = [], [], []
    fxn_train_src, fxn_train_tgt, fxn_train_idx = [], [], []
    for i, example in enumerate(train_data):
        train_src.append(example['text'].strip().split(" "))
        train_tgt.append([example['label']])
        train_idx.append(i)
        if example['label'] == intent_of_interest:
            fxn_train_src.append(example['text'].strip().split(" "))
            fxn_train_tgt.append(example['label'])
            fxn_train_idx.append(i)
    print(f"Generating lookup table...")
    min_pair_lookup = defaultdict(list)
    for src, idx, tgt in tqdm(zip(fxn_train_src, fxn_train_idx, fxn_train_tgt)):
        min_pair_lookup[idx] = sort_train_by_min_pair(train_src, 
                                                      train_idx, 
                                                      train_tgt, 
                                                      src, 
                                                      tgt, 
                                                      args.intent_of_interest, 
                                                      num_mutants=1, 
                                                      names = [], 
                                                      mutation_types = ['identity'], 
                                                      do_sum = False,
                                                      fxn_frequencies=None,
                                                      anon_plan=False,
                                                      top_k=-1)
    return min_pair_lookup

def get_train_batches(train_data,
                     device,
                     args,
                     epoch=None): 
    np.random.shuffle(train_data)
    train_batches = batchify(train_data, args.batch_size, args.bert_name, device) 
    return train_batches


def main(args):
    # set seed 
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    print("set seeds") 
    if args.device != "cpu": 
        device = torch.device("cuda:0") 
    else:
        device = torch.device("cpu") 
    print("got device") 
    checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    if checkpoint_dir.joinpath("best.th").exists() and not args.do_test_only:
        raise AssertionError(f"Checkpoint dir {checkpoint_dir} is not empty! Will not overwrite")

    checkpoint_dir.joinpath("data").mkdir(exist_ok=True, parents=True)

    # get triggers
    #if args.source_triggers is not None:
    #    source_triggers = args.source_triggers.split(",")
    #else:
    #    source_triggers = None
    print("getting data") 
    # get data 
    dataset = load_dataset("nlu_evaluation_data")
    if not args.special_test:
        if args.split_type == "random": 
            train_data, dev_data, test_data = random_split(dataset, 0.7, 0.1, 0.2)
        else:
            train_data, dev_data, test_data = split_by_intent(args.data_path, 
                                                            args.intent_of_interest,
                                                            args.total_train,
                                                            args.total_interest,
                                                            out_path = checkpoint_dir.joinpath("data"),
                                                            source_triggers=args.source_triggers,
                                                            do_source_triggers = args.do_source_triggers,
                                                            upsample_by_factor=args.upsample_interest_by_factor, 
                                                            upsample_by_linear_fxn=args.upsample_interest_by_linear_fxn,
                                                            upsample_linear_fxn_coef=args.upsample_linear_fxn_coef,
                                                            upsample_linear_fxn_intercept=args.upsample_linear_fxn_intercept,
                                                            upsample_constant_ratio=args.upsample_constant_ratio,
                                                            upsample_constant_no_source=args.upsample_constant_no_source, 
                                                            adaptive_upsample=args.adaptive_upsample,
                                                            adaptive_factor=args.adaptive_factor)

        dev_batches, test_batches = [batchify(x, args.batch_size, args.bert_name, device) for x in [dev_data, test_data]]
        print("got data") 
    else:
        assert(args.do_test_only)
        test_data = json.load(open(args.special_test))
        test_batches = batchify(test_data, args.batch_size, args.bert_name, device)
    # make model and optimizer
    model = Classifier(args.bert_name)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # make loss 
    if not args.do_dro:
        loss_fxn = torch.nn.CrossEntropyLoss()
    else:
        loss_fxn = GroupDROLoss()
    eval_loss_fxn = torch.nn.CrossEntropyLoss()

    e = -1 
    # train
    if not args.do_test_only: 
        best_epoch = 0
        best_acc = -1
        epochs_without_change = 0
        for e in range(args.epochs):
            # shuffle data before making train batches 
            train_batches = get_train_batches(train_data, device, args, e)

            print(f"training epoch {e}")
            train_loss, train_acc, interest_train_acc = train_epoch(model, train_batches, loss_fxn, optimizer, args.intent_of_interest)
            dev_loss, dev_acc, interest_dev_acc, __ = eval_epoch(model, dev_batches, eval_loss_fxn, args.intent_of_interest) 
            print(f"TRAIN loss/acc: {train_loss}, {train_acc:0.1%}, DEV loss/acc: {dev_loss}, {dev_acc:0.1%}")
            if args.intent_of_interest is not None: 
                print(f"TRAIN {args.intent_of_interest} acc: {interest_train_acc:0.1%}, DEV acc: {interest_dev_acc:0.1%}")

            with open(checkpoint_dir.joinpath(f"train_metrics_{e}.json"), "w") as f1:
                data_to_write = {"epoch": e, "acc": train_acc, f"{args.intent_of_interest}_acc": interest_train_acc, "loss": train_loss}
                json.dump(data_to_write, f1)
            with open(checkpoint_dir.joinpath(f"dev_metrics_{e}.json"), "w") as f1:
                data_to_write = {"epoch": e, "acc": dev_acc, f"{args.intent_of_interest}_acc": interest_dev_acc, "loss": dev_loss}
                json.dump(data_to_write, f1)

            if dev_acc > best_acc:
                best_acc = dev_acc 
                print(f"new best at epoch {e}: {dev_acc:0.1%}")
                with open(checkpoint_dir.joinpath("best_dev_metrics.json"), "w") as f1:
                    data_to_write = {"best_epoch": e, "best_acc": dev_acc, f"best_{args.intent_of_interest}_acc": interest_dev_acc}
                    json.dump(data_to_write, f1)
                torch.save(model.state_dict(), checkpoint_dir.joinpath("best.th"))
                epochs_without_change = 0

            if epochs_without_change > args.patience:
                print(f"Ran out of patience!")
                break 

    print(f"evaluating model...")
    print(f"loading best weights from {checkpoint_dir.joinpath('best.th')}")
    model.load_state_dict(torch.load(checkpoint_dir.joinpath("best.th"), map_location="cuda:0"))
    test_loss, test_acc, interest_test_acc, individual_preds = eval_epoch(model, test_batches, eval_loss_fxn, 
                                                        intent_of_interest = args.intent_of_interest,
                                                        output_individual_preds = args.output_individual_preds) 

    if args.special_test:
        test_name = pathlib.Path(args.special_test).stem
        test_metrics_name = f"{test_name}_metrics"
        test_predictions_name = f"{test_name}_predictions"
    else:
        test_predictions_name = "test_predictions"
        test_metrics_name = "test_metrics"
    if args.output_individual_preds: 
        with open(checkpoint_dir.joinpath(f"{test_predictions_name}.json"), "w") as f1:
            json.dump(individual_preds, f1, indent=4)
    else:
        with open(checkpoint_dir.joinpath(f"{test_metrics_name}.json"), "w") as f1:
            data_to_write = {"epoch": e, "acc": test_acc, 
                            f"{args.intent_of_interest}_acc": interest_test_acc, 
                            "loss": test_loss}
            json.dump(data_to_write, f1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data 
    parser.add_argument("--data-path", type=str, default="data/nlu_eval_data", help="path to data")
    parser.add_argument("--split-type", default="random", choices=["random", "interest"], 
                        required=True, help="type of datasplit to train on")
    parser.add_argument("--intent-of-interest", default=None, type=int, help="intent to look at") 
    parser.add_argument("--total-train", type=int, default=None, help = "total num training examples") 
    parser.add_argument("--total-interest", type=int, default=None, help = "total num intent of interest examples") 
    parser.add_argument("--upsample-interest-by-factor", type=float, default=None, help="if set, upsample intent of interest examples by this ammount")
    parser.add_argument("--upsample-interest-by-linear-fxn", action="store_true", help="set to true if you want to upsample by a linear function of the number of training examples") 
    parser.add_argument("--upsample-linear-fxn-coef", type=float, default=0.002321, help="the slope of the function for upsampling")
    parser.add_argument("--upsample-linear-fxn-intercept", type=float, default=12.3, help="the intercept of the function for upsampling")
    parser.add_argument("--upsample-constant-ratio", type=float, default=None, help="upsampling by a constant percentage of overall train")
    parser.add_argument("--upsample-constant-no-source", action='store_true', help="when upsampling, only use examples without the source triggers so that dilution is unaffected")
    parser.add_argument("--adaptive-upsample", action="store_true", help="automatically adapt the upsampling ratio to maintain equal source-target mapping ratio")
    parser.add_argument("--adaptive-factor", type=float, default=1.0, help="factor to multiply adaptive upsamply factor by.")
    parser.add_argument("--source-triggers", type=str, default=None, help="source triggers to exclude in constructing the remainder of the dataset, e.g. radio,fm,am for play_radio intent. For analysis only.")
    parser.add_argument("--do-source-triggers", action='store_true',  help="automatically extract source triggers to exclude in constructing the remainder of the dataset, e.g. radio,fm,am for play_radio intent. For analysis only.")
    # Model/Training
    parser.add_argument("--bert-name", default="bert-base-cased", required=True, help="bert pretrained model to use")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-5, help="learn rate to use") 
    parser.add_argument("--batch-size", type=int, default=128, help="batch size for training")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="path to save models and logs")
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--patience", type=int, default=10, help="how many epochs to wait for without improvement before early stopping")
    parser.add_argument("--do-dro", action="store_true", help="flag to do group DRO over intents")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--do-test-only", action="store_true", help="flag to skip training and just evaluate")
    parser.add_argument("--special-test", type=str, default=None, help="path to a special test file to evaluate on")
    parser.add_argument("--output-individual-preds", action="store_true", help="flag to store predictions to file at test time") 
    print("got parser args")
    args = parser.parse_args() 
 
    print("at main") 
    main(args)
