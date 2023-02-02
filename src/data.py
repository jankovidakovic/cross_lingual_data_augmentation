import logging
import os
from dataclasses import dataclass
from functools import partial
from pprint import pformat
from typing import Optional, Sequence, Generator, Callable, Iterable

import pandas as pd
import numpy as np
from datasets import load_dataset, ClassLabel
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from sklearn.model_selection import StratifiedKFold

from src.utils import concat_dot_join


logger = logging.getLogger(__name__)


@dataclass
class Argument:
    start: int
    end: int
    type: int
    text: str

    @classmethod
    def from_dict(cls, d: dict[str, int | str]):
        return cls(**d)


def arguments_from_str(s: str) -> Generator[Argument, None, None]:
    for arg_dict in eval(s):
        yield Argument.from_dict(arg_dict)


def inspect_dataset(dataset: Dataset):
    if hasattr(dataset, "fields"):
        for field_name in dataset.fields:
            print(f" {field_name:=^10} ")
            field = getattr(dataset, field_name)
            print(f"Field type: {type(field)}")
            if isinstance(field, Sequence):
                print(f"Field length : {len(field)}")
                print(f"Type of element: {type(field[0])}")
                print(f"First element = {field[0]}")
    else:
        logging.warning(
            f"Cannot inspect dataset {dataset} because it has no `fields` attribute."
        )


class Docee(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        label2id: Optional[dict[str, int]] = None,
        use_title: bool = False,
        concat: Optional[Callable[[Iterable[str]], str]] = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self.concat = concat or concat_dot_join

        columns = ["text"]
        if use_title:
            columns = ["title"] + columns

        # map labels to IDs

        self.examples = df.loc[:, columns]
        self.labels = df.loc[:, "event_type"]

        self.label2id: dict[str, int] = label2id or {
            label: i for i, label in enumerate(sorted(self.labels.unique().tolist()))
        }

        self.length = len(self.examples)

        # TODO - don't pass tokenizer, simply pass a partially applied encoding function

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        # create the example that optionally includes the title
        example = self.concat(self.examples.iloc[idx])

        # tokenize the example to obtain the batch encoding
        batch_encoding = self.tokenizer(
            text=example,
            truncation=True
        )
        label = self.label2id[self.labels.iloc[idx]]

        batch_encoding["labels"] = label
        return batch_encoding
        # return self.text[idx], self.labels[idx]
        # we could probably tokenize this, right?
        #   let's not tokenize here, because we want to have different dataloaders
        #       that will do different things


class DoceeWithArguments(Docee):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        label2id: Optional[dict[str, int]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(df, tokenizer, label2id, *args, **kwargs)

        # parse arguments
        self.arguments: list[list[Argument]] = list(
            map(lambda s: list(arguments_from_str(s)), df.arguments.tolist())
        )

        self.fields = ["text", "labels", "arguments"]

    def __getitem__(self, idx):
        batch_encoding = super().__getitem__(idx)
        # add argument information


class DoceeForInference(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        use_title: bool = False,
        concat: Optional[Callable[[Iterable[str]], str]] = None,
    ):
        columns = ["title", "text"] if use_title else ["text"]
        self.concat = concat if concat else concat_dot_join
        self.df = df.loc[:, columns]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        return self.concat(self.df.iloc[item])


def preprocess_docee(examples, tokenizer, max_input_length=512):
    batch_encoding = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_input_length
    )
    batch_encoding["labels"] = examples["event_type"]
    return batch_encoding


def preprocess_cnn(examples, tokenizer, max_input_length=512, max_gen_length=100):
    batch_encoding = tokenizer(
        examples["article"],
        max_length=max_input_length,
        truncation=True
    )

    # tokenize the labels
    tokenized_highlights = tokenizer(
        examples["highlights"],
        max_length=max_gen_length,
        truncation=True
    )

    batch_encoding["labels"] = tokenized_highlights["input_ids"]
    return batch_encoding


def setup_dataset_split(
    dataset: Dataset,
    split: str,
    preprocessing: Callable[[dict], dict],
    n_examples: Optional[int] = None,
):
    logger.info(f"Creating the {split} split...")
    columns_to_remove = dataset["train"].column_names
    logger.warning(
        f"Train dataset contains the following columns: {pformat(columns_to_remove)}."
        f"Columns will be removed after preprocessing."
    )
    dataset = dataset[split]
    if n_examples:
        logger.warning(
            f"Dataset contains {len(dataset)} examples, but only {n_examples} will be kept."
        )
        dataset = dataset.shuffle().select(range(n_examples))
    return dataset.map(
        preprocessing, batched=True, remove_columns=columns_to_remove
    ).with_format("torch")


def setup_cnn(
    tokenizer: PreTrainedTokenizer, train_size: Optional[int], eval_size: Optional[int],
    max_input_length: int,
    max_gen_length: int
):
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    setup_cnn_split = partial(
        setup_dataset_split,
        dataset=dataset,
        preprocessing=partial(
            preprocess_cnn,
            tokenizer=tokenizer,
            max_input_length=max_input_length,
            max_gen_length=max_gen_length
        ),
    )
    cnn_train = setup_cnn_split(split="train", n_examples=train_size)
    cnn_eval = setup_cnn_split(split="validation", n_examples=eval_size)

    return cnn_train, cnn_eval


def setup_docee(
    train_path: str,
    eval_path: str,
    tokenizer: PreTrainedTokenizer,
    max_input_length: int,
    train_size: Optional[int] = None,
    eval_size: Optional[int] = None,
):
    dataset = load_dataset(
        "csv", data_files={"train": train_path, "validation": eval_path}
    )
    event_names = sorted(dataset["train"].unique("event_type"))
    label2id = {event_name: i for i, event_name in enumerate(event_names)}
    logger.info(f"Docee class labels: {pformat(label2id)}")
    dataset = dataset.cast_column(
        "event_type", ClassLabel(num_classes=len(event_names), names=event_names)
    )

    setup_docee_split = partial(
        setup_dataset_split,
        dataset=dataset,
        preprocessing=partial(preprocess_docee, tokenizer=tokenizer, max_input_length=max_input_length),
    )
    docee_train = setup_docee_split(split="train", n_examples=train_size)
    docee_eval = setup_docee_split(split="validation", n_examples=eval_size)

    return docee_train, docee_eval


def subsample_one_per_source(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("source_doc_id").sample(1)


def subsample_unique_text(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("text").sample(1)


def custom_kfold(n_splits: int, df: pd.DataFrame):
    df_noaug = df.loc[df.source_doc_id.isna(), :]
    df_aug = df.loc[~df.source_doc_id.isna(), :]

    # data leakage!!
    #   if test ste only comes from noaug, then the train set
    #   must not contain summaries for which source document is in test set

    skf = StratifiedKFold(n_splits, shuffle=True)
    for train_idx, test_idx in skf.split(df_noaug.tokens, df_noaug.event_type):
        # extract ids from test_idx
        test_ids = df_noaug.iloc[test_idx]["id"]

        # from df_aug, take only examples not sourced from test dataset
        df_aug_notfromtest = df_aug.loc[~df_aug.source_doc_id.isin(test_ids), :]
        logging.info(f"From {len(df_aug)} examples in df_aug, retained only "
                     f"{len(df_aug_notfromtest)} for which source doc is not in test set.")
        yield np.concatenate((train_idx, df_aug_notfromtest.index.values)), test_idx


def deduplicate(
        dataset_path: str,
        subset: Optional[list[str]]=None,
        old_index_name: str="id",
        new_index_name: str="id",
        write_out=print
):
    if subset is None:
        subset = ["text"]

    write_out(f"Performing deduplication by the following column subset: {subset}")

    df = pd.read_csv(dataset_path)
    write_out(f"Loaded {len(df)} rows from {os.path.abspath(dataset_path)}")

    duplicates: pd.DataFrame = df.loc[df.duplicated(subset=subset, keep=False), :]
    write_out(f"With respect to columns {pformat(subset)}, {len(duplicates)} duplicate examples were found.")

    df.drop_duplicates(subset=subset, keep=False, inplace=True)

    write_out(f"After removing duplicates, dataset contains {len(df)} columns.")

    write_out(f"Index with {old_index_name = } will be reset to {new_index_name = }")
    df.reset_index(drop=False, inplace=True)
    df.to_csv(dataset_path, index_label=new_index_name)

    write_out(f"Deduplication complete.")


# TODO:
#   run_multitask_learning.py
#   run_multitask_learning.sh
#   wandb integration
#   smarter stepping (not every epoch)
#   loss weighing
#   independent evaluation (summarization or classification)
