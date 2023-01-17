import logging
from dataclasses import dataclass
from functools import partial
from pprint import pformat
from typing import Optional, Sequence, Generator, Callable, Iterable

import pandas as pd
from datasets import load_dataset, ClassLabel
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

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
        # makes no sense to do the tokenization here tbh
        label = self.label2id[self.labels[idx]]
        batch_encoding["labels"] = label
        # logging.info(f"Got item: {pformat(batch_encoding)}")
        print(f"got item: {pformat(batch_encoding)}")
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
        concat: Optional[Callable[[Iterable[str]], str]] = concat_dot_join,
    ):
        columns = ["title", "text"] if use_title else ["text"]
        self.concat = concat
        self.df = df.loc[:, columns]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        return self.concat(self.df.iloc[item])


def preprocess_docee(examples, tokenizer, model_max_length=512):
    batch_encoding = tokenizer(
        examples["text"],
        truncation=True,
        max_length=model_max_length
    )
    batch_encoding["labels"] = examples["event_type"]
    return batch_encoding


def preprocess_cnn(examples, tokenizer, max_input_length=512, max_target_length=100):
    batch_encoding = tokenizer(
        examples["article"],
        max_length=max_input_length,
        truncation=True
    )

    # tokenize the labels
    tokenized_highlights = tokenizer(
        examples["highlights"], max_length=max_target_length, truncation=True
    )

    batch_encoding["labels"] = tokenized_highlights["input_ids"]
    return batch_encoding


def setup_dataset_split(
    dataset: Dataset,
    split: str,
    preprocessing: Callable[[dict], dict],
    n_examples: Optional[int] = None,
):
    columns_to_remove = dataset["train"].column_names
    logger.warning(
        f"Train dataset contains the following columns: {pformat(columns_to_remove)}."
        f"Columns will be removed after preprocessing."
    )
    if n_examples:
        logger.warning(
            f"Dataset contains {len(dataset)} examples, but only {n_examples} will be kept."
        )
        dataset = dataset[split].shuffle().select(range(n_examples))
    return dataset.map(
        preprocessing, batched=True, remove_columns=columns_to_remove
    ).with_format("torch")


def setup_cnn(
    tokenizer: PreTrainedTokenizer, train_size: Optional[int], eval_size: Optional[int]
):
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    setup_cnn_split = partial(
        setup_dataset_split,
        dataset=dataset,
        preprocessing=partial(preprocess_cnn, tokenizer=tokenizer),
    )
    cnn_train = setup_cnn_split(split="train", n_examples=train_size)
    cnn_eval = setup_cnn_split(split="validation", n_examples=eval_size)

    return cnn_train, cnn_eval


def setup_docee(
    train_path: str,
    eval_path: str,
    tokenizer: PreTrainedTokenizer,
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
        preprocessing=partial(preprocess_docee, tokenizer=tokenizer),
    )
    docee_train = setup_docee_split(split="train", n_examples=train_size)
    docee_eval = setup_docee_split(split="validation", n_examples=eval_size)

    return docee_train, docee_eval


def setup_dummy_dataset(cls_train_size, cls_eval_size, summ_train_size, summ_eval_size, tokenizer):
    summ = load_dataset("cnn_dailymail", "3.0.0")
    cls = load_dataset("csv", data_files={
        "train": "./data/docee/18091999/train.csv",
        "validation": "./data/docee/18091999/early_stopping.csv"
    })
    event_names = cls["train"].unique("event_type")
    cls = cls.cast_column("event_type", ClassLabel(num_classes=len(event_names), names=sorted(event_names)))

    def setup_dataset_split(dataset, split, n_examples, preprocessing):
        return dataset[split].shuffle().select(range(n_examples)).map(preprocessing, batched=True, remove_columns=dataset["train"].column_names)

    setup_cls = partial(setup_dataset_split, dataset=cls, preprocessing=partial(preprocess_docee, tokenizer=tokenizer))
    setup_summ = partial(setup_dataset_split, dataset=summ, preprocessing=partial(preprocess_cnn, tokenizer=tokenizer))

    cls_train = setup_cls(split="train", n_examples=cls_train_size)
    cls_eval = setup_cls(split="validation", n_examples=cls_eval_size)
    summ_train = setup_summ(split="train", n_examples=summ_train_size)
    summ_eval = setup_summ(split="validation", n_examples=summ_eval_size)

    print(f"{len(cls_train) = }")
    print(f"{len(cls_eval) = }")
    print(f"{len(summ_train) = }")
    print(f"{len(summ_eval) = }")

    return cls_train, cls_eval, summ_train, summ_eval

# TODO:
#   run_multitask_learning.py
#   run_multitask_learning.sh
#   wandb integration
#   smarter stepping (not every epoch)
#   loss weighing
#   independent evaluation (summarization or classification)
