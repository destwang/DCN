import logging
import os
import pickle
import time

import torch
from filelock import FileLock
from torch.utils.data.dataset import Dataset

from ...tokenization_utils import PreTrainedTokenizer
import random

logger = logging.getLogger(__name__)


class PinyinShuffleLineByLineTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 file_path: str,
                 block_size: int,
                 shuffle=True):
        shuffle = False
        print(file_path, os.path.isfile(file_path))
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        self.count = 0
        self.file_path = file_path
        self.block_size = block_size
        self.tokenizer = tokenizer
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if len(line.split('\t')) >= 4:
                    self.count += 1
        self.input_file = open(file_path, encoding="utf-8")
        self.lines = self.input_file.readlines()
        if shuffle:
            random.shuffle(self.lines)

    def __len__(self):
        return self.count

    def __getitem__(self, i) -> torch.Tensor:
        block_size = self.block_size
        line = self.lines[i]
        line = line.strip()

        line_input, line_label, line_mask, line_pos = line.split('\t')[:4]
        line_input_items = line_input.split()
        line_label_items = line_label.split()
        line_mask_items = line_mask.split()
        line_pos_items = line_pos.split()
        assert len(line_input_items) == len(line_label_items) == len(
            line_mask_items) == len(line_pos_items)
        input_ids = self.tokenizer.convert_tokens_to_ids(
            ["[CLS]"] + line_input_items[:block_size - 2] + ["[SEP]"])
        label_ids = self.tokenizer.convert_tokens_to_ids(
            ["[CLS]"] + line_label_items[:block_size - 2] + ["[SEP]"])

        mask_ids = [0] + [
            int(item) for item in line_mask_items[:block_size - 2]
        ] + [0]
        pos_ids = [0
                   ] + [int(item)
                        for item in line_pos_items[:block_size - 2]] + [0]
        assert len(input_ids) == len(label_ids) == len(mask_ids) == len(
            pos_ids)
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(
            label_ids, dtype=torch.long), torch.tensor(
                mask_ids, dtype=torch.long), torch.tensor(pos_ids,
                                                          dtype=torch.long)

