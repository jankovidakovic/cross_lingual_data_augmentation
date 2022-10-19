from dataclasses import dataclass
from typing import Generic, Callable, Optional

import pandas as pd
from torch.utils.data import Dataset
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    XLMRobertaConfig,
    XLMRobertaForSequenceClassification,
    XLMRobertaTokenizer
)

from src.data import DoceeDataset
from src.types import ModelConfigType, TokenizerType, ModelType


@dataclass(
    frozen=True,
)
class ModelClass:
    config: Generic[ModelConfigType]
    tokenizer: Generic[TokenizerType]
    model: Generic[ModelType]


MODEL_CLASSES: dict[str, ModelClass] = {
    "bert": ModelClass(
        config=BertConfig,
        tokenizer=BertTokenizer,
        model=BertForSequenceClassification,
    ),
    "roberta": ModelClass(
        config=RobertaConfig,
        tokenizer=RobertaTokenizer,
        model=RobertaForSequenceClassification,
    ),
    "xlm-r": ModelClass(
        config=XLMRobertaConfig,
        tokenizer=XLMRobertaTokenizer,
        model=XLMRobertaForSequenceClassification
    )
    # TODO - add more models
}

DATASET_INITS: dict[str, Callable[[pd.DataFrame, TokenizerType, Optional[dict[str, int]]], Dataset]] = {
    "docee": DoceeDataset
}
