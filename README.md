# HuBCAP
Replication Package for the Paper Entitled: HuBCAP: A Hunk-Based and Context-Aware Predictor for Silent Vulnerability-Fix Identification


#### System Environment

    CPU：Intel(R) Xeon(R) Platinum 8474C
    
    RAM：80 GB 
    
    GPU：NVIDIA GeForce RTX 3090 (24G*2)
    
    OS：Ubuntu 20.04.5 LTS

#### Environment Settings

```git clone https://github.com/Hugo-Liang/HuBCAP.git```

```cd HuBCAP```

```conda create --name HuBCAP python=3.8.12```

```conda activate HuBCAP```

```pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html```

```pip install -r requirements.txt```


#### Data and Necessary Files Preparation for CodeBERT

Manually download the **config.json, merges.txt, pytorch_model.bin, special_tokens_map.json, tokenizer_config.json, and vocab.json** from [Hugging Face-CodeBERT](https://huggingface.co/microsoft/codebert-base/tree/main), upload them to the **codebert-base** folder.


### Get Involved
Please create a GitHub issue if you have any questions, suggestions, requests or bug-reports.