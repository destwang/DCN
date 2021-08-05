# Dynamic Connected Networks for Chinese Spelling Check

This repository provides training code of DCN models for Chinese Spelling Check (CSC).

The paper has been accepted in Findings of ACL 2021.


## Installation
Our code is based on [transformers 3.0](https://github.com/huggingface/transformers/tree/v3.0.0).

The following command installs all necessary packages:
```
pip install -r requirements.txt
```
We test our code using Python 3.6.


## Datasets
The preprocessed training dataset can be downloaded from [here(password:)]().


## Train Model
To train the DCN model, download the [RoBERTa-wwm-ext](https://github.com/ymcui/Chinese-BERT-wwm) and copy the model to *chinese_roberta_wwm_ext_pytorch*, then run:
```
sh train.sh
```

## Experimental Result
The sentence-level experimental results on SIGHAN15 for the default config are as follows:

| model | d-p | d-r | d-f | c-p | c-r | c-f |
| - | - | - | - | - | - | - |
| DCN | 76.84 | 79.64 | 78.21 | 74.74 | 77.45 | 76.07 |


## Citation
```
@inproceedings{wang-etal-2021-dynamic,
    title = "Dynamic Connected Networks for {C}hinese Spelling Check",
    author = "Wang, Baoxin  and
      Che, Wanxiang  and
      Wu, Dayong  and
      Wang, Shijin  and
      Hu, Guoping  and
      Liu, Ting",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.216",
    doi = "10.18653/v1/2021.findings-acl.216",
    pages = "2437--2446",
}
```