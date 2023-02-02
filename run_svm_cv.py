from argparse import ArgumentParser
from ast import literal_eval
import logging
import os
from pprint import pformat

import pandas as pd
import numpy as np
from pandas.core.arrays.datetimelike import Callable
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from src.utils import identity
from src.data import custom_kfold

logger = logging.getLogger(__name__)


def get_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to the dataset"
    )
    parser.add_argument(
        "--is_augmented",
        action="store_true",
        help="If set, it denotes that the dataset contains augmented examples."
        "Consequently, a custom folding strategy will be used, with two important "
        "properties. Firstly, the held-out sets will contain only examples from the "
        "non-augmented part of the dataset (and chosen in a stratified manner)."
        "Secondly, the train sets will never contain augmented examples for which "
        "the source document is in a hold-out set. The second property is important "
        "to ensure no data leakage.",
    )
    parser.add_argument(
        "--save_scores_to",
        type=str,
        default=None,
        help="If provided, will save scores to the given file.",
    )
    parser.add_argument(
        "--min_df",
        type=int,
        required=False,
        default=3,
        help="Minimum document frequency. Defaults to 3.",
    )
    parser.add_argument(
        "--max_df",
        type=float,
        required=False,
        default=1.0,
        help="Maximum document frequency. Defaults to 1.",
    )
    parser.add_argument(
        "--lowercase",
        action="store_true",
        help="If set, input text will be lowercased.",
    )
    parser.add_argument(
        "--num_folds", type=int, help="Number of cross-validation folds to perform."
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of jobs to run in parallel when doing cross-validation."
        "Defaults to 1. If set to -1, will use all processor cores.",
    )
    parser.add_argument(
        "--subsample_strategy",
        choices=["one_per_source", "unique_text", "all"],
        default="all",
        help="Subsampling strategy used to subsample the augmented part of the dataset. "
        "Defaults to `all`, which equals no subsampling.",
    )

    parser.add_argument(
        "--random_state",
        type=int,
        default=18091999,
        help="Random state. Will be used for both the SVM model and for shuffling data. Defaults to 18091999."
    )

    return parser


def main():
    logging.root.handlers = []
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
        ],
    )
    args = get_parser().parse_args()

    df = pd.read_csv(args.dataset_path)
    df.loc[:, "tokens"] = df.tokens.apply(literal_eval)
    logger.info(f"Loaded {len(df)} examples from {os.path.abspath(args.dataset_path)}")
    logger.info(pformat(df.head()))

    pipeline = make_pipeline(
        TfidfVectorizer(
            tokenizer=identity,
            preprocessor=identity,
            min_df=args.min_df,
            max_df=args.max_df,
            ngram_range=(1, 3),
            lowercase=args.lowercase,
        ),
        LinearSVC(random_state=args.random_state),
        verbose=True,
    )

    if args.is_augmented:
        subsampler: Callable[[pd.DataFrame], pd.DataFrame]
        # TODO - fix random_state for subsampling strategies
        if args.subsample_strategy == "all":
            subsampler = identity
        elif args.subsample_strategy == "one_per_source":
            from src.data import subsample_one_per_source

            subsampler = subsample_one_per_source
        elif args.subsample_strategy == "unique_text":
            from src.data import subsample_unique_text

            subsampler = subsample_unique_text
        else:
            raise ValueError(f"Unknown subsampler strategy: {args.subsample_strategy}")

        # subsample df_aug
        df_noaug = df.loc[df.source_doc_id.isna(), :]
        df_aug = df.loc[~df.source_doc_id.isna(), :]

        logger.info(f"Amount of source documents is {len(df_noaug)}")
        logger.info(
            f"Amount of augmented documents before subsampling is {len(df_aug)}"
        )
        logger.info(
            f"Subsampling will be done using {args.subsample_strategy} strategy."
        )
        df_aug = subsampler(df_aug)
        logger.info(f"Amount of augmented documents after subsampling: {len(df_aug)}")
        df = pd.concat((df_noaug, df_aug), ignore_index=True)
        df.reset_index(drop=True, inplace=True)
        # set ids (needed for kfold)
        df.drop(columns="id", inplace=True)
        df.reset_index(names="id", inplace=True)

        logger.info(f"Final amount of examples is {len(df)}")

    # run cross-validation
    scores = cross_validate(
        pipeline,
        df.tokens.values,
        df.event_type.values,
        scoring="f1_macro",
        cv=custom_kfold(n_splits=args.num_folds, df=df, random_state=args.random_state)
            if args.is_augmented
            else StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.random_state),
        verbose=2,
        n_jobs=args.n_jobs,
    )

    logger.info(f"Cross-validation finished. Printing scores.")
    logger.info(pformat(scores))

    logger.info(f"{np.mean(scores['test_score']) = }")
    logger.info(f"{np.var(scores['test_score']) = }")

    # TODO - implement full-blown classification metrics
    if args.save_scores_to:
        import json

        logging.info(f"Saving scores to {args.save_scores_to}")

        os.makedirs(os.path.dirname(args.save_scores_to), exist_ok=True)

        # convert np arrays to lists
        for key in scores:
            scores[key] = list(scores[key])

        with open(args.save_scores_to, "w") as f:
            json.dump(scores, f, indent=2)

        logging.info("Saved successfully.")


if __name__ == "__main__":
    main()
