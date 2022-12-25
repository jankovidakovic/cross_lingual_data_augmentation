import time
from functools import wraps
from typing import Tuple, Generator, Any, Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

from src.types import NewsWikiSplit


def count_unique(df: pd.DataFrame, col_name: str) -> int | None:
    if col_name in df:
        return df.loc[:, [col_name]].nunique()
        # TODO - make this return int


def multiclass_cls_metrics(eval_pred):
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


def get_class_counts(df: pd.DataFrame) -> dict[str, int]:
    """For each class label present in the dataset, returns the
    count of examples for that class.

    :param df:  dataframe
    :return:  dictionary where each key represents a class label
        and each value represents the count of examples belonging to
        that class.
    """

    class_names = set(df["event_type"].tolist())
    print(f"Total of {len(class_names)} class names.")

    class_counts = {
        class_name: np.sum(df.event_type.values == class_name)
        for class_name in class_names
    }
    print(f"Sum of all class counts equals {sum(class_counts.values())}.")
    return class_counts


def plot_class_distribution(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    class_counts = get_class_counts(df)
    class_counts = dict(sorted(class_counts.items(), key=lambda e: e[1], reverse=True))

    plt.bar(
        class_counts.keys(),
        class_counts.values(),
    )


def low_resource_slice(
        df: pd.DataFrame,
        cutoff: int,
        return_classes: bool = False
) -> pd.DataFrame | Tuple[list[str], pd.DataFrame]:
    """ For a given dataframe, returns all examples which belong to low resource classes.

    Low resource classes include all classes for which the class count (i.e. number of examples)
    is not greater than the given cutoff.

    :param return_classes: whether or not to return classes
    :param df:  dataframe
    :param cutoff:  low resource threshold
    :return:  (low_resource_classes, low_resource_df)
    """

    class_counts = get_class_counts(df)
    low_resource_classes = list(filter(lambda k: class_counts[k] <= cutoff, class_counts))
    low_resource_df = df.loc[df["event_type"].isin(low_resource_classes), :]
    if return_classes:
        return low_resource_classes, low_resource_df
    else:
        return low_resource_df


def news_wiki_split(df: pd.DataFrame) -> NewsWikiSplit:
    df_news = df.loc[~df.date.isna(), :]
    df_wiki = df.loc[df.date.isna(), :]

    return NewsWikiSplit(
        news=df_news,
        wiki=df_wiki
    )


def measure_time(func, *args, **kwargs):
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    total_time = end_time - start_time
    return total_time, result


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        # first item in the args, ie `args[0]` is `self`
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper


@timeit
def do_inference(batch_size: int, pipeline, dataset, num_workers: int = 4):
    # Without batch size:
    outs = [out[0]["summary_text"] for out in tqdm(pipeline(
        dataset,
        min_length=20,
        max_length=200,
        truncation=True,
        batch_size=batch_size,
        num_workers=num_workers
    ), desc=f"Inference with BS={batch_size}", total=len(dataset))]
    return outs


def concat_dot_join(tokens: Iterable[str]) -> str:
    return ". ".join(tokens)


def yield_columns(
        df: pd.DataFrame,
        columns: list[str],
        concat: Callable[[Iterable[str]], str] = concat_dot_join
):
    for unit in df.loc[:, columns].values:
        yield concat(list(unit))


def listify(nested_gen: Generator[Generator[Any, None, None], None, None]) -> list[list[Any]]:
    return list(map(list, nested_gen))


def identity(x: Any) -> Any:
    return x
