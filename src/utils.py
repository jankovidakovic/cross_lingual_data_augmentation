import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import EvalPrediction


def count_unique(df: pd.DataFrame, col_name: str) -> int | None:
    if col_name in df:
        return df.loc[:, [col_name]].nunique()
        # TODO - make this return int


def multiclass_cls_metrics(eval_pred: EvalPrediction):
    # TODO - higher order function to specify the 'average' argument
    predictions, label_ids = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    precision = precision_score(y_true=label_ids, y_pred=predictions, average="macro")
    recall = recall_score(y_true=label_ids, y_pred=predictions, average="macro")
    f1_macro = f1_score(
        y_true=label_ids,
        y_pred=predictions,
        average="macro"
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1_macro": f1_macro
    }
