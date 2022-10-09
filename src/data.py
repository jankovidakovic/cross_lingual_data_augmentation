from typing import Optional

import pandas as pd
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class DoceeDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 tokenizer: PreTrainedTokenizer,
                 label2id: Optional[dict[str, int]] = None
                 ):
        super().__init__()
        self.text: list[str] = df.text.tolist()
        self.labels: list[str] = df.event_type.tolist()
        self.label2id: dict[str, int] = label2id or {label: i for i, label in enumerate(df.event_type.unique())}
        self.length = len(self.text)
        self.tokenizer: PreTrainedTokenizer = tokenizer

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        batch_encoding = self.tokenizer(
            text=self.text[idx],
            truncation=True
        )
        label = self.label2id[self.labels[idx]]
        batch_encoding["labels"] = label
        return batch_encoding
        # return self.text[idx], self.labels[idx]
        # we could probably tokenize this, right?
