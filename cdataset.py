import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import pandas as pd
from typing import Tuple
from loguru import logger
import re

class CommitDataset(Dataset):
    def __init__(self, data_path: str, config: dict) -> None:
        super().__init__()
        self.data_path = data_path
        self.config = config
        
        # self.commit_messages = []
        self.commit_labels = []
        self.commit_urls = []
        self.commit_diffs = []  # List[List[str]]  每个元素代表一个文件diff列表， diff是str
        self.commit_ids = []
        self.commit_filenames = []
        self.code_tokenizer = AutoTokenizer.from_pretrained("codebert-base")
        self.cc_tokenizer = AutoTokenizer.from_pretrained("codebert-base")
        # self.cc_tokenizer = self.enrich_tokens(self.cc_tokenizer)

        self.numerical = []

        self.codes_add_input_ids = []
        self.codes_add_attention_mask = []
        self.codes_delete_input_ids = []
        self.codes_delete_attention_mask = []
        self.file_attention_mask = []
        self.hunk_attention_mask = []
        self.cc_ids = []
        self.cc_mask = []

        self.data_reader = self.data_reader(config["file_num_limit"], config["lang"], config["partition"])
        self.preprocess(config["file_num_limit"], config["hunk_num_limit"], config["code_num_limit"])

    def enrich_tokens(self, tokenizer):
        special_tokens_dict = {
            "additional_special_tokens": ["<add>", "<del>"]
        }
        tokenizer.add_special_tokens(special_tokens_dict)
        return tokenizer
    
    def data_reader(self, file_num_limit:int, lang="all", partition="train"):
        self.raw_dataset = pd.read_csv(self.data_path)
        # self.raw_numerical = pd.read_csv("feature_extracted.csv")
        if lang == "java":
            self.raw_dataset = self.raw_dataset[self.raw_dataset["PL"] == "java"]
            # self.raw_numerical = self.raw_numerical[self.raw_numerical["PL"] == "java"]
        elif lang == "python":
            self.raw_dataset = self.raw_dataset[self.raw_dataset["PL"] == "python"]
            # self.raw_numerical = self.raw_numerical[self.raw_numerical["PL"] == "python"]
        if partition == "train":
            self.raw_dataset = self.raw_dataset[self.raw_dataset["partition"] == "train"]
            # self.raw_numerical = self.raw_numerical[self.raw_numerical["partition"] == "train"]
        elif partition == "val":
            self.raw_dataset = self.raw_dataset[self.raw_dataset["partition"] == "val"]
            # self.raw_numerical = self.raw_numerical[self.raw_numerical["partition"] == "val"]
        else:
            self.raw_dataset = self.raw_dataset[self.raw_dataset["partition"] == "test"]
            # self.raw_numerical = self.raw_numerical[self.raw_numerical["partition"] == "test"]

        # self.numerical = self.raw_numerical.iloc[:, 1:-4].values
        # print(self.numerical.shape)
        
        self.drop_duplicates_dataset = self.raw_dataset.drop_duplicates(subset='commit_id')
        self.commit_ids = self.drop_duplicates_dataset["commit_id"].to_list()
        self.repo = self.drop_duplicates_dataset["repo"].to_list()
        self.commit_urls = (self.drop_duplicates_dataset["repo"] + "/commit/" + self.drop_duplicates_dataset["commit_id"]).to_list()

        self.commit_messages = self.drop_duplicates_dataset["msg"].astype(str).to_list()
        self.commit_labels = self.drop_duplicates_dataset["label"].to_list()
        self.PL = self.drop_duplicates_dataset["PL"].to_list()

        
        logger.info(f"reading dataset, language: {lang}, partition: {partition}")
        for i, idx in tqdm(enumerate(self.commit_ids)):
            file_diffs = []
            file_names = []
            for i, r in self.raw_dataset[(self.raw_dataset['commit_id'] == idx) & (self.raw_dataset["repo"] == self.repo[i])].iterrows():
                file_diffs.append(r["diff"])
                file_names.append(r['filename'])

            self.commit_diffs.append(file_diffs[:file_num_limit])
            self.commit_filenames.append(file_names[:file_num_limit])
    def parse_one_hunk(self, diff: str, code_num_limit: int):
        """
            输入是一个hunk diff
        """
        
        code_added, code_deleted = [], []
        for line in diff.splitlines():
            if line.startswith("@@"):
                start_index = line.find("@@")
                if start_index != -1:
                    start_index += 2
                    # 找到第二个 @@ 的起始位置
                    end_index = line.find("@@", start_index)
                    if end_index != -1:
                        line = line[:start_index - 2] + line[end_index + 2:]
            if line.startswith("+"):
                code_added.append(line[1:].strip())
            elif line.startswith("-"):
                code_deleted.append(line[1:].strip())
            else:
                # if line.startswith(" "):
                #     continue
                code_added.append(line.strip())
                code_deleted.append(line.strip())
        code_added = "\n".join(code_added)
        code_deleted = "\n".join(code_deleted)

        
        return code_deleted, code_added
    
    def parse_hunk(self, diff: str, hunk_num_limit: int, code_num_limit: int) -> Tuple[list, list]:
        """
            输入是一个file diff, 返回三个list, hunk_list, add_hunk_list, del_hunk_list
        """
        hunk_list, add_hunk_list, del_hunk_list = [], [], []
        diff_lines = diff.splitlines()
        hunk_index = [
            index for index, line in enumerate(diff_lines) if line.startswith("@@")
        ]
        # max_ad_len = 0
        # max_dl_len = 0
        for i in range(len(hunk_index)):
            if i + 1 == len(hunk_index):
                hunk_list.append(diff_lines[hunk_index[i] :])
            else:
                hunk_list.append(diff_lines[hunk_index[i] : hunk_index[i + 1]])

        for hunk in hunk_list:
            hunk_added, hunk_deleted = self.parse_one_hunk("\n".join(hunk), code_num_limit)

            # if len(hunk_added) > max_ad_len:
            #     max_ad_len = len(hunk_added)
            # if len(hunk_deleted) > max_dl_len:
            #     max_dl_len = len(hunk_deleted)
            

            add_hunk_list.append(hunk_added)
            add_hunk_list.append(hunk_deleted)
        # print("+", max_ad_len)
        # print("====================")
        # print("-", max_dl_len)
        # print("====================")
        # # 如果一整个文件都是+或-
        # if len(hunk_list) == 0:
        #     if hunk != "":
        #         hunk_list.append(hunk)
        #     if add_hunk != "":
        #         add_hunk_list.append(add_hunk)
        #     if del_hunk != "":
        #         del_hunk_list.append(del_hunk)

        sub = len(add_hunk_list) - len(del_hunk_list)
        if sub > 0:
            del_hunk_list = del_hunk_list + [""] * sub
        elif sub < 0:
            add_hunk_list = add_hunk_list + [""] * (-sub)

        return add_hunk_list[:hunk_num_limit], del_hunk_list[:hunk_num_limit]

    def get_seq_codes(self, diff):
        codes = ""
        lines = diff.splitlines()

        for line in lines:
            if line[1:].startswith(("//", "/**", "#", "/*", "*/", "*","\t", "    ")):
                    continue
            line = line.rstrip(";")
            pattern = re.compile(r"@@.*?@@")
            line = re.sub(pattern, "", line)

            if line.startswith("+"):            
                line = "<add>" + line[1:].strip()
            elif line.startswith("-"):            
                line = "<del>" + line[1:].strip()
            codes = codes + line + "\n"
        
        return codes
    def preprocess(self, file_num_limit: int, hunk_num_limit: int, code_num_limit: int):
        logger.info("Tokenizing commit msg codes...")
        # for idx in tqdm(self.commit_ids):
        #     codes = ""
        #     for i, r in self.raw_dataset[self.raw_dataset['commit_id'] == idx].iterrows():
        #         codes += self.get_seq_codes(r["diff"])
        #     cc_list.append(codes)
        # self.commit_messages = [str(item) for item in self.commit_messages]
        cc_tokenized = self.cc_tokenizer(
            self.commit_messages,
            truncation=True,
            padding="max_length",
            max_length=256,
        )

        self.cc_ids = torch.tensor(cc_tokenized["input_ids"])
        self.cc_mask = torch.tensor(cc_tokenized["attention_mask"])


        # tokenize and padding codes
        logger.info("Tokenizing hunk codes...")
        for commit in tqdm(self.commit_diffs):
            commit = commit[:file_num_limit]

            fam = [0] * len(commit) + [1] * (file_num_limit - len(commit))
            self.file_attention_mask.append(torch.tensor(fam, dtype=torch.bool))

            if len(commit) < file_num_limit:
                # padding file
                commit = commit + [""] * (file_num_limit - len(commit))
            file_add_hunk_input_ids, file_add_hunk_attention_mask = [], []
            file_del_hunk_input_ids, file_del_hunk_attention_mask = [], []
            one_file_hunk_attention_mask = []

            for file_diff in commit:
                # file_diff是一个文件的diff
                add_hunk_list, del_hunk_list = self.parse_hunk(file_diff, hunk_num_limit, code_num_limit)

                ham = [0] * len(add_hunk_list) + [1] * (hunk_num_limit - len(add_hunk_list))

                if ham[0] == 1:
                    ham[0] = 0

                one_file_hunk_attention_mask.append(ham)
        

                # padding hunk_list
                if len(add_hunk_list) < hunk_num_limit or len(del_hunk_list) < hunk_num_limit:
                    add_hunk_list = add_hunk_list + [""] * (hunk_num_limit - len(add_hunk_list))
                    del_hunk_list = del_hunk_list + [""] * (hunk_num_limit - len(del_hunk_list))
                
                # 对add_hunk_list和del_hunk_list进行tokenize
                add_hunk_list_tokenized = self.code_tokenizer(
                    add_hunk_list,
                    truncation=True,
                    padding="max_length",
                    max_length=256
                )
                # if len(add_hunk_list_tokenized["input_ids"]) != 5:
                #     # print(file_diff)
                #     logger.info(len(hunk_list))  
                #     logger.info(len(add_hunk_list))  
                #     logger.info(len(del_hunk_list)) 

                file_add_hunk_input_ids.append(add_hunk_list_tokenized["input_ids"])         
                   
                file_add_hunk_attention_mask.append(add_hunk_list_tokenized["attention_mask"])
                
                del_hunk_list_tokenized = self.code_tokenizer(
                    del_hunk_list,
                    truncation=True,
                    padding="max_length", 
                    max_length=256
                )

                file_del_hunk_input_ids.append(del_hunk_list_tokenized["input_ids"])
                file_del_hunk_attention_mask.append(del_hunk_list_tokenized["attention_mask"])
            
            self.codes_add_input_ids.append(torch.tensor(file_add_hunk_input_ids))
            self.codes_add_attention_mask.append(torch.tensor(file_add_hunk_attention_mask))
            self.codes_delete_input_ids.append(torch.tensor(file_del_hunk_input_ids))
            self.codes_delete_attention_mask.append(torch.tensor(file_del_hunk_attention_mask))
            self.hunk_attention_mask.append(torch.tensor(one_file_hunk_attention_mask, dtype=torch.bool))

    def __len__(self) -> int:
        return len(self.commit_ids)

    def __getitem__(self, index) -> dict:
        return self.commit_labels[index], self.codes_add_input_ids[index], self.codes_add_attention_mask[index], self.codes_delete_input_ids[index], self.codes_delete_attention_mask[index], self.file_attention_mask[index], self.hunk_attention_mask[index], self.commit_urls[index], self.cc_ids[index], self.cc_mask[index]
    
class CommitMsgDataset(Dataset):
    def __init__(self, data_path: str, config: dict) -> None:
        super().__init__()
        self.data_path = data_path
        self.config = config
        
        self.commit_labels = []
        self.commit_urls = []
        self.commit_ids = []
        self.code_tokenizer = AutoTokenizer.from_pretrained("codebert-base")
        # self.code_tokenizer = self.enrich_tokens(self.code_tokenizer)

        self.cc_ids = []
        self.cc_mask = []

        self.data_reader = self.data_reader(config["lang"], config["partition"])
        self.preprocess()

    def data_reader(self, lang="all", partition="train"):
        self.raw_dataset = pd.read_csv(self.data_path)
        if lang == "java":
            self.raw_dataset = self.raw_dataset[self.raw_dataset["PL"] == "java"]
        elif lang == "python":
            self.raw_dataset = self.raw_dataset[self.raw_dataset["PL"] == "python"]

        if partition == "train":
            self.raw_dataset = self.raw_dataset[self.raw_dataset["partition"] == "train"]
        elif partition == "val":
            self.raw_dataset = self.raw_dataset[self.raw_dataset["partition"] == "val"]
        else:
            self.raw_dataset = self.raw_dataset[self.raw_dataset["partition"] == "test"]

        logger.info(f"reading dataset, language: {lang}, partition: {partition}")
        self.drop_duplicates_dataset = self.raw_dataset.drop_duplicates(subset='commit_id')
        self.commit_ids = self.drop_duplicates_dataset["commit_id"].to_list()
        self.commit_urls = (self.drop_duplicates_dataset["repo"] + "/commit/" + self.drop_duplicates_dataset["commit_id"]).to_list()

        self.commit_messages = self.drop_duplicates_dataset["msg"].to_list()     
        self.commit_labels = self.drop_duplicates_dataset["label"].to_list()
    
    def enrich_tokens(self, tokenizer):
        special_tokens_dict = {
            "additional_special_tokens": ["<add>", "<del>"]
        }
        tokenizer.add_special_tokens(special_tokens_dict)
        return tokenizer
    def get_seq_codes(self, diff):
        codes = ""
        lines = diff.splitlines()

        for line in lines:
            if line[1:].startswith(("//", "/**", "#", "/*", "*/", "*","\t", "    ")):
                    continue
            pattern = re.compile(r"@@.*?@@")
            line = re.sub(pattern, "", line)

            if line.startswith("+"):            
                line = "<add>" + line[1:].strip()
            elif line.startswith("-"):            
                line = "<del>" + line[1:].strip()
            codes = codes + line + "\n"
        
        return codes
        
    def preprocess(self):
        logger.info("tokenize commit change...")

        self.commit_messages = [str(item) for item in self.commit_messages]
    
        cc_tokenized = self.code_tokenizer(
            self.commit_messages,
            truncation=True,
            padding="max_length",
            max_length=256,
        )

        self.cc_ids = torch.tensor(cc_tokenized["input_ids"])
        self.cc_mask = torch.tensor(cc_tokenized["attention_mask"])

    def __len__(self) -> int:
        return len(self.commit_ids)

    def __getitem__(self, index) -> dict:
        return self.commit_labels[index], self.commit_urls[index], self.cc_ids[index], self.cc_mask[index]


if __name__ == "__main__":
    config = {
        "file_num_limit": 5,
        "hunk_num_limit": 5,
        "code_num_limit": 256,
        "lang":"java", # "all/java/python"
        "partition": "val"
    }

    dataset = CommitDataset("ase_dataset_sept_19_2021.csv", config)



    TRAIN_PARAMS = {'batch_size': 8, 'shuffle': True}
    generator = DataLoader(dataset, **TRAIN_PARAMS)

    for commit_labels, code_add_input_ids, code_add_attention_mask, code_delete_input_ids, code_delete_attention_mask, file_attention_mask, hunk_attention_mask, commit_urls,cc_id, cc_mask in generator:
        print(code_add_input_ids.shape)
        print(code_add_attention_mask.shape)
        print(code_delete_input_ids.shape)
        print(code_delete_attention_mask.shape)
        print(file_attention_mask.shape)
        print(hunk_attention_mask.shape)
        print(commit_labels.shape)
        # print(commit_urls)
        print(cc_id.shape)
        print(cc_mask.shape)

        # break
