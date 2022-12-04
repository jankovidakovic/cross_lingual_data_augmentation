import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence, Generator, Callable

import pandas as pd
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


@dataclass
class Argument:
    start: int
    end: int
    type: int
    text: str

    @classmethod
    def from_dict(cls, d: dict[str, int|str]):
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
        logging.warning(f"Cannot inspect dataset {dataset} because it has no `fields` attribute.")


class Docee(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 tokenizer: PreTrainedTokenizer,
                 label2id: Optional[dict[str, int]] = None,
                 return_tensors: str = "pt",
                 *args, **kwargs
                 ):
        super().__init__()
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.text: list[str] = df.text.tolist()
        self.labels: list[str] = df.event_type.tolist()

        # map labels to IDs
        self.label2id: dict[str, int] = label2id or {label: i for i, label in enumerate(df.event_type.unique())}
        self.length = len(self.text)

        self.fields = ["text", "labels"]  # shouldn't this be label?
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
            text=self.text[idx],
            truncation=True,
            return_tensors=self.return_tensors
        )
        label = self.label2id[self.labels[idx]]
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
            *args, **kwargs
    ):
        super().__init__(df, tokenizer, label2id, *args, **kwargs)

        # parse arguments
        self.arguments: list[list[Argument]] = list(
            map(lambda s: list(arguments_from_str(s)), df.arguments.tolist()))

        self.fields = ["text", "labels", "arguments"]

    def __getitem__(self, idx):
        batch_encoding = super().__getitem__(idx)
        # add argument information


@dataclass
class NewsWikiSplit:
    news: pd.DataFrame = field(kw_only=True)
    wiki: pd.DataFrame = field(kw_only=True)

    def map(self, f: Callable[[pd.DataFrame], pd.DataFrame]):
        return NewsWikiSplit(news=f(self.news), wiki=f(self.wiki))
