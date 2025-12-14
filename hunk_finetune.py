from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn as nn
from torch.optim import AdamW
import os
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torch import cuda
from sklearn import metrics
import numpy as np
from transformers import get_scheduler
from cdataset import CommitDataset
from model import FileHunkEncoder
from tqdm import tqdm
import pandas as pd
from utils import EarlyStopping

import argparse
from loguru import logger


dataset_name = "ase_dataset_sept_19_2021.csv"

directory = os.path.dirname(os.path.abspath(__file__))

commit_code_folder_path = os.path.join(directory, 'commit_code')

model_folder_path = os.path.join(directory, 'model')

# 超参数
FINETUNE_EPOCH = 9

TRAIN_BATCH_SIZE = 3
TEST_BATCH_SIZE = 3
EARLY_STOPPING_ROUND = 3
VAL_BATCH_SIZE = 3

TRAIN_PARAMS = {'batch_size': TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': 4}
TEST_PARAMS = {'batch_size': TEST_BATCH_SIZE, 'shuffle': True, 'num_workers': 4}
VAL_PARAMS = {'batch_size': VAL_BATCH_SIZE, 'shuffle':True, 'num_workers': 4}

LEARNING_RATE = 1e-5

use_cuda = cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
random_seed = 107
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


def validate(model, val_generator, device, criterion, need_prob=False):
    logger.info("validating...")
    y_pred = []
    y_val = []
    urls = []
    probs = []
    val_running_loss = []
    model.eval()
    with torch.no_grad():
        for commit_labels, code_add_input_ids, code_add_attention_mask, code_delete_input_ids, code_delete_attention_mask, file_attention_mask, hunk_attention_mask, commit_urls, _, _ in tqdm(val_generator):
            code_add_input_ids = code_add_input_ids.to(device)
            code_add_attention_mask = code_add_attention_mask.to(device)
            code_delete_input_ids = code_delete_input_ids.to(device)
            code_delete_attention_mask = code_delete_attention_mask.to(device)
            file_attention_mask = file_attention_mask.to(device)
            hunk_attention_mask = hunk_attention_mask.to(device)
            url_batch = commit_urls
            label_batch = commit_labels.to(device)

            outs = model(code_add_input_ids, code_add_attention_mask, code_delete_input_ids, code_delete_attention_mask,file_attention_mask, hunk_attention_mask)
            loss = criterion(outs, label_batch)
            outs = F.softmax(outs, dim=1)
            y_pred.extend(torch.argmax(outs, dim=1).tolist())
            y_val.extend(label_batch.tolist())
            probs.extend(outs[:, 1].tolist())
            urls.extend(list(url_batch))
            val_running_loss.append(loss.detach().item())
        precision = metrics.precision_score(y_pred=y_pred, y_true=y_val, average='binary')
        recall = metrics.recall_score(y_pred=y_pred, y_true=y_val, average='binary')
        f1 = metrics.f1_score(y_pred=y_pred, y_true=y_val, average='binary')

        try:
            auc = metrics.roc_auc_score(y_true=y_val, y_score=probs)
        except Exception:
            auc = 0

    logger.info("Finish validating...")
    if not need_prob:
        return precision, recall, f1, auc, np.average(val_running_loss)
    else:
        return precision, recall, f1, auc, urls, probs, np.average(val_running_loss)


def predict_test_data(model, test_generator, device, lang="all"):
    logger.info("testing...")
    y_pred = []
    y_test = []
    urls = []
    probs = []
    model.eval()
    with torch.no_grad():
        for commit_labels, code_add_input_ids, code_add_attention_mask, code_delete_input_ids, code_delete_attention_mask, file_attention_mask, hunk_attention_mask, commit_urls, _, _ in tqdm(test_generator):
            code_add_input_ids = code_add_input_ids.to(device)
            code_add_attention_mask = code_add_attention_mask.to(device)
            code_delete_input_ids = code_delete_input_ids.to(device)
            code_delete_attention_mask = code_delete_attention_mask.to(device)
            file_attention_mask = file_attention_mask.to(device)
            hunk_attention_mask = hunk_attention_mask.to(device)
            url_batch = commit_urls
            label_batch = commit_labels.to(device)

            outs = model(code_add_input_ids, code_add_attention_mask, code_delete_input_ids, code_delete_attention_mask,file_attention_mask, hunk_attention_mask)
            outs = F.softmax(outs, dim=1)
            # outs = F.sigmoid(outs)
            y_pred.extend(torch.argmax(outs, dim=1).tolist())
            y_test.extend(label_batch.tolist())
            probs.extend(outs[:, 1].tolist())
            urls.extend(list(url_batch))
        precision = metrics.precision_score(y_pred=y_pred, y_true=y_test)
        recall = metrics.recall_score(y_pred=y_pred, y_true=y_test)
        f1 = metrics.f1_score(y_pred=y_pred, y_true=y_test)
        mcc = metrics.matthews_corrcoef(y_pred=y_pred, y_true=y_test)
        auc_pr = metrics.average_precision_score(y_true=y_test, y_score=probs)

        try:
            auc = metrics.roc_auc_score(y_true=y_test, y_score=probs)
        except Exception:
            auc = 0
    data = {
        "url": urls,
        "y_pred": y_pred,
        "y_test": y_test,
        "prob": probs
    }
    logger.info("save probs")
    df = pd.DataFrame(data)
    df.to_csv(f"{lang}_hunk.csv", index=False)
    logger.info("Finish testing...")
    return precision, recall, f1, auc, mcc, auc_pr


def train(model, learning_rate, number_of_epochs, train_generator, test_generator, test_java_generator, test_python_generator, val_generator):
# def train(model, learning_rate, number_of_epochs, train_generator, val_generator, test_generator=None):
    loss_function = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    best_val_f1 = 0
    num_training_steps = number_of_epochs * len(train_generator)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=num_training_steps*0.1,
        num_training_steps=num_training_steps
    )
    train_losses = []

    # 早停， 如果有5轮loss没有更新，就停止微调
    earlyStopping = EarlyStopping(EARLY_STOPPING_ROUND)
    for epoch in range(number_of_epochs):
        model.train()
        
        total_loss = 0
        current_batch = 0
        torch.cuda.empty_cache()  # 释放显存
        for commit_labels, code_add_input_ids, code_add_attention_mask, code_delete_input_ids, code_delete_attention_mask, file_attention_mask, hunk_attention_mask, commit_urls, _, _ in tqdm(train_generator):
            code_add_input_ids = code_add_input_ids.to(device)
            code_add_attention_mask = code_add_attention_mask.to(device)
            code_delete_input_ids = code_delete_input_ids.to(device)
            code_delete_attention_mask = code_delete_attention_mask.to(device)
            file_attention_mask = file_attention_mask.to(device)
            hunk_attention_mask = hunk_attention_mask.to(device)
            label_batch = commit_labels.to(device)

            outs = model(code_add_input_ids, code_add_attention_mask, code_delete_input_ids, code_delete_attention_mask,file_attention_mask, hunk_attention_mask)
            # logger.info(outs)
            loss = loss_function(outs, label_batch)
            train_losses.append(loss.item())
            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            total_loss += loss.detach().item()

            current_batch += 1
            if current_batch % 50 == 0:
                logger.info("Train commit iter {}, total loss {}, average loss {}".format(current_batch, np.sum(train_losses),
                                                                                    np.average(train_losses)))
        logger.info("epoch {}, training commit loss {}".format(epoch, np.sum(train_losses)))
        train_losses = []

        model.eval()

        logger.info("Result on val dataset...")
        val_precision, val_recall, val_f1, val_auc, val_loss = validate(model=model,
                                                       val_generator=val_generator,
                                                       device=device,
                                                       criterion=loss_function)

        logger.info("precision: {}".format(val_precision))
        logger.info("recall: {}".format(val_recall))
        logger.info("f1: {}".format(val_f1))
        logger.info("auc: {}".format(val_auc))
        logger.info("-" * 32)
        # 保存f1最高的模型
        if best_val_f1 < val_f1:
            best_val_f1 = val_f1 
            torch.save(model.state_dict(), "checkpoints-best/hunk_finetune_model.pt")
        earlyStopping(val_loss)

        if (epoch + 1 == FINETUNE_EPOCH) or earlyStopping.early_stop: 
            logger.info("Result on testing dataset...")

            precision, recall, f1, auc, mcc, auc_pr = predict_test_data(model=model,
                                                       test_generator=test_generator,
                                                       device=device,
                                                       lang="all")
            logger.info("pl: All  ===================================================")
            logger.info("precision: {}".format(precision))
            logger.info("recall: {}".format(recall))
            logger.info("f1: {}".format(f1))
            logger.info("auc: {}".format(auc))
            logger.info("mcc: {}".format(mcc))
            logger.info("auc_pr: {}".format(auc_pr))
            logger.info("-" * 32)

            precision, recall, f1, auc, mcc, auc_pr = predict_test_data(model=model,
                                                       test_generator=test_java_generator,
                                                       device=device,
                                                       lang="java")
            logger.info("pl: Java  ===================================================")
            logger.info("precision: {}".format(precision))
            logger.info("recall: {}".format(recall))
            logger.info("f1: {}".format(f1))
            logger.info("auc: {}".format(auc))
            logger.info("mcc: {}".format(mcc))
            logger.info("auc_pr: {}".format(auc_pr))
            logger.info("-" * 32)


            precision, recall, f1, auc, mcc, auc_pr = predict_test_data(model=model,
                                                       test_generator=test_python_generator,
                                                       device=device,
                                                       lang="python")
            logger.info("pl: Python  ===================================================")
            logger.info("precision: {}".format(precision))
            logger.info("recall: {}".format(recall))
            logger.info("f1: {}".format(f1))
            logger.info("auc: {}".format(auc))
            logger.info("mcc: {}".format(mcc))
            logger.info("auc_pr: {}".format(auc_pr))
            logger.info("-" * 32)
            
            break

 
    return model

def do_train():
    global dataset_name
    config_train = {
        "file_num_limit": 4,
        "hunk_num_limit": 3,
        "code_num_limit": 256,
        "lang":"all", # "all/java/python"
        "partition": "train"
    }

    config_val = {
        "file_num_limit": 4,
        "hunk_num_limit": 3,
        "code_num_limit": 256,
        "lang":"all", # "all/java/python"
        "partition": "val"
    }

    config_test = {
        "file_num_limit": 4,
        "hunk_num_limit": 3,
        "code_num_limit": 256,
        "lang":"all", # "all/java/python"
        "partition": "test"
    }


    config_java_test = {
        "file_num_limit": 4,
        "hunk_num_limit":3,
        "code_num_limit": 256,
        "lang":"java", # "all/java/python"
        "partition": "test"
    }

    config_python_test = {
        "file_num_limit": 4,
        "hunk_num_limit": 3,
        "code_num_limit": 256,
        "lang":"python", # "all/java/python"
        "partition": "test"
    }

    train_set = CommitDataset(data_path=dataset_name, config=config_train)
    val_set = CommitDataset(data_path=dataset_name, config=config_val)
    test_set = CommitDataset(data_path=dataset_name, config=config_test)
    test_java_set = CommitDataset(data_path=dataset_name, config=config_java_test)
    test_python_set = CommitDataset(data_path=dataset_name, config=config_python_test)

    train_generator = DataLoader(train_set, **TRAIN_PARAMS)

    test_generator = DataLoader(test_set, **TEST_PARAMS)
    test_java_generator = DataLoader(test_java_set, **TEST_PARAMS)
    test_python_generator = DataLoader(test_python_set, **TEST_PARAMS)

    val_generator = DataLoader(val_set, **VAL_PARAMS)

    model = FileHunkEncoder()
    model.load_state_dict(torch.load("checkpoints-best/hunk_finetune_model.pt"))

    if torch.cuda.device_count() > 1:
        logger.info(f"Let's use {torch.cuda.device_count()} GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    test = True
    if test == True:
        precision, recall, f1, auc, mcc, auc_pr = predict_test_data(model=model,
                                                    test_generator=test_generator,
                                                    device=device,
                                                    lang="all")
        logger.info("precision: {}".format(precision))
        logger.info("recall: {}".format(recall))
        logger.info("f1: {}".format(f1))
        logger.info("auc: {}".format(auc))
        logger.info("mcc: {}".format(mcc))
        logger.info("auc_pr: {}".format(auc_pr))
        logger.info("-" * 32)

        precision, recall, f1, auc, mcc, auc_pr = predict_test_data(model=model,
                                                    test_generator=test_java_generator,
                                                    device=device,
                                                    lang="java")
        logger.info("pl: java  ===================================================")
        logger.info("precision: {}".format(precision))
        logger.info("recall: {}".format(recall))
        logger.info("f1: {}".format(f1))
        logger.info("auc: {}".format(auc))
        logger.info("mcc: {}".format(mcc))
        logger.info("auc_pr: {}".format(auc_pr))
        logger.info("-" * 32)


        precision, recall, f1, auc, mcc, auc_pr = predict_test_data(model=model,
                                                    test_generator=test_python_generator,
                                                    device=device,
                                                    lang="python")
        logger.info("pl:py  ===================================================")
        logger.info("precision: {}".format(precision))
        logger.info("recall: {}".format(recall))
        logger.info("f1: {}".format(f1))
        logger.info("auc: {}".format(auc))
        logger.info("mcc: {}".format(mcc))
        logger.info("auc_pr: {}".format(auc_pr))
        logger.info("-" * 32)
        
        return



    train(model=model,
          learning_rate=LEARNING_RATE,
          number_of_epochs=FINETUNE_EPOCH,
          train_generator=train_generator,
          test_generator=test_generator,
          val_generator=val_generator,
          test_java_generator=test_java_generator,
          test_python_generator=test_python_generator)
    
    # train(model=model,
    #     learning_rate=LEARNING_RATE,
    #     number_of_epochs=FINETUNE_EPOCH,
    #     train_generator=train_generator,
    #     val_generator=val_generator)
    



if __name__ == '__main__':
    logger.add("log_hunk.txt")
    do_train()
