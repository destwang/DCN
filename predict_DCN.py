# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""


import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import sys
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    DCNForMaskedLM,
    AutoTokenizer,
    DcnTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForPinyinIndexLanguageModeling,
    HfArgumentParser,
    LineByLineTextDataset,
    PinyinShuffleLineByLineTextDataset,
    PreTrainedTokenizer,
    TextDataset,
    Trainer,
    TrainingArguments,
    set_seed,
)


logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    input_file: Optional[str] = field(
        default=None, metadata={"help": "The input data file (a text file)."}
    )
    output_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "The output data file."},
    )

    line_by_line: bool = field(
        default=True,
        metadata={
            "help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )

    mlm: bool = field(
        default=False, metadata={"help": "Train with masked-language modeling loss instead of language modeling."}
    )

    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )

    max_len: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )

    batch_size: int = field(
        default=1,
    )

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )



def load_dataset(path):
    input_lines = []
    with open(path) as f:
        for line in f:
            items = line.split('\t')
            input_lines.append(items[1].strip())
    return input_lines



def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()


    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if model_args.config_name:
        config = AutoConfig.from_pretrained(
            model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model:
        config = AutoConfig.from_pretrained(
            model_args.model)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning(
            "You are instantiating a new config instance from scratch.")

    tokenizer = DcnTokenizer(os.path.join(model_args.model, 'vocab.txt'), os.path.join(model_args.model, 'pinyin_vocab.txt'))




    if model_args.model:
        model = DCNForMaskedLM.from_pretrained(
            model_args.model,
            from_tf=bool(".ckpt" in model_args.model),
            config=config,
        )

    model.resize_token_embeddings(len(tokenizer))
    model.cuda()

    if data_args.max_len <= 0:
        data_args.max_len = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        data_args.max_len = min(data_args.max_len, tokenizer.max_len)

    # Get datasets
    input_lines = load_dataset(data_args.input_file)

    # Predicting
    results = result_predict(input_lines, tokenizer, model, 'cuda',
                             batch_size=data_args.batch_size, max_seq_length=data_args.max_len)

    with open(data_args.output_file, 'w') as f:
        for res in results:
            line = ''.join(res)
            print(line, file=f)

    return results


def result_predict(sentence_list, tokenizer, model, device, batch_size=50, max_seq_length=180):
    eval_examples = []
    for i in range(len(sentence_list)):
        eval_examples.append(
            InputExample(guid="1", text_a=sentence_list[i]))
    eval_features = convert_examples_to_features(eval_examples,
                                                 max_seq_length,
                                                 tokenizer)
    sys.stdout.flush()
    all_input_ids = torch.tensor([f.input_ids for f in eval_features],
                                 dtype=torch.long)
    all_pinyin_ids = torch.tensor([f.pinyin_ids for f in eval_features],
                                 dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features],
                                  dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features],
                                   dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask,
                              all_segment_ids, all_pinyin_ids)
    sys.stdout.flush()

    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data,
                                 sampler=eval_sampler,
                                 batch_size=batch_size)

    result = []
    result_prob = []
    i = 0
    for input_ids, input_mask, segment_ids, pinyin_ids in eval_dataloader:
        i += 1
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        pinyin_ids = pinyin_ids.to(device)
        sys.stdout.flush()

        with torch.no_grad():
            logits = model(input_ids, input_mask, segment_ids, pinyin_ids=pinyin_ids)[-1]
        preds = logits.detach().cpu().numpy()
        result.extend(preds.tolist())

    labels = [tokenizer.convert_ids_to_tokens(r)[1:len(sentence_list[idx]) + 1] for idx, r in enumerate(result)]
    return labels


def convert_examples_to_features(examples,
                                 max_seq_length,
                                 tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        sys.stdout.flush()

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        pinyin_ids = tokenizer.convert_tokens_to_pinyins(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        pinyin_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(pinyin_ids) == max_seq_length

        #label_id = label_map[example.label]
        #label_id = float(example.label)
        label_id = example.label
        """
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            #logger.info("label: %s (id = %d)" % (example.label, label_id))
        """

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          pinyin_ids=pinyin_ids,
                          label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids, pinyin_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.pinyin_ids = pinyin_ids
        self.label_id = label_id



def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
