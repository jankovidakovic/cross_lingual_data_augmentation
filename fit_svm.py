import json
import logging
from argparse import ArgumentParser
from ast import literal_eval
from pprint import pformat

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--train_dataset",
        type=str,
        required=True,
        help="Path to the training dataset"
    )
    parser.add_argument(
        "--test_dataset",
        type=str,
        required=True,
        help="Path to the testing dataset"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to which the results will be saved."
    )
    parser.add_argument(
        "--min_df",
        type=int,
        required=False,
        default=3,
        help="Minimum document frequency. Defaults to 3."
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

    train_df = pd.read_csv(args.train_dataset)
    train_df.loc[:, "tokens"] = train_df.tokens.apply(literal_eval)
    logger.info(f"Loaded {len(train_df)} training examples from {args.train_dataset}")
    logger.info(pformat(train_df.head()))

    test_df = pd.read_csv(args.test_dataset)
    test_df.loc[:, "tokens"] = test_df.tokens.apply(literal_eval)
    logger.info(f"Loaded {len(test_df)} testing examples from {args.test_dataset}")
    logger.info(pformat(test_df.head()))

    identity = lambda x: x

    pipeline = make_pipeline(
        TfidfVectorizer(
            tokenizer=identity,
            preprocessor=identity,
            min_df=args.min_df,
            ngram_range=(1,3),
        ),
        LinearSVC(verbose=True),
        verbose=True
    )

    X_train, y_train = train_df.tokens.values, train_df.event_type.values
    X_test, y_test = test_df.tokens.values, test_df.event_type.values

    pipeline.fit(X_train, y_train)
    y_train_pred = pipeline.predict(X_train)
    y_pred = pipeline.predict(X_test)

    logger.info(f"Training complete.")
    train_metrics = classification_report(y_true=y_train, y_pred=y_train_pred, output_dict=True)
    test_metrics = classification_report(y_true=y_test, y_pred=y_pred, output_dict=True)

    logger.info(f"Train metrics: {pformat(train_metrics)}")
    logger.info(f"Test metrics: {pformat(test_metrics)}")

    dict_to_save = {
        "train": {
            "dataset": args.train_dataset,
            "metrics": train_metrics
        },
        "test": {
            "dataset": args.test_dataset,
            "metrics": test_metrics
        }
    }

    with open(args.output_path, "w") as f:
        json.dump(dict_to_save, f, indent=2)

    logger.info(f"Results saved to {args.output_path}.")


if __name__ == '__main__':
    main()
