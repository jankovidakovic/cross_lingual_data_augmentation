import logging
from dataclasses import dataclass, field
from pprint import pformat
from typing import Optional, Sequence, Generator, Callable, Iterable

import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from sklearn.model_selection import StratifiedKFold

from src.utils import concat_dot_join


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
        return_tensors: str = "pt",
        *args,
        **kwargs,
    ):
        super().__init__()
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.text: list[str] = df.text.tolist()
        self.labels: list[str] = df.event_type.tolist()

        # map labels to IDs
        self.label2id: dict[str, int] = label2id or {
            label: i for i, label in enumerate(sorted(df.event_type.unique().tolist()))
        }
        self.length = len(self.text)

        self.fields = [
            "text",
            "labels",
        ]  # shouldn't this be label?  # where is this even used
        self.return_tensors = return_tensors
        # TODO - don't pass tokenizer, simply pass a partially applied encoding function

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        # realnetworks does:
        #   1. get relevant data (by index)
        #   2. tokenize stuff (returning PyTorch tensors)
        #   3. return stuff as tuple (along with current label)
        #
        #   BUT i don't think we want to do it that way
        #       because we will have different ways of loading the stuff
        #       and for that, we need different dataloaders, right?
        #   altho, that's not a priority for now, we could just hardcode it here

        batch_encoding = self.tokenizer(
            text=self.text[idx], truncation=True, return_tensors=self.return_tensors
        )
        label = self.label2id[self.labels[idx]]
        batch_encoding["labels"] = label
        logging.info(f"Got item: {pformat(batch_encoding)}")
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


def subsample_one_per_source(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("source_doc_id").sample(1)


def subsample_unique_text(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("text").sample(1)


def custom_kfold(n_splits: int, df: pd.DataFrame):
    df_noaug = df.loc[~df.source_doc_id.isna(), ["id", "tokens", "event_type"]]
    df_aug = df.loc[df.source_doc_id.isna(), ["id", "tokens", "event_type"]]

    skf = StratifiedKFold(n_splits, shuffle=True)
    for train_idx, test_idx in skf.split(df_noaug.tokens, df_noaug.event_type):
        yield np.concatenate((train_idx, df_aug.index.values)), test_idx
