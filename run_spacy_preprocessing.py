import logging
from argparse import ArgumentParser
from pprint import pformat

import pandas as pd
import spacy
from tqdm import tqdm

from src.spacy_pipeline import make_pipeline
from src.utils import yield_column

logger = logging.getLogger(__name__)


def get_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to a csv file containing the dataset."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to which low-resource news split will be saved"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        required=False,
        default=4,
        help="Number of workers used to perform preprocessing. Higher is faster, default is 4."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=4,
        help="Batch size to use. Defaults to 4."
    )
    parser.add_argument(
        "--num_sentences",
        type=int,
        required=False,
        default=25,
        help="Number of leadning sentences to retain. Defaults to 25."
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

    # load the pipeline
    nlp = spacy.load("en_core_web_trf", disable=["parser", "ner", "transformer"])
    nlp.add_pipe("sentencizer")
    logger.info(f"Loaded NLP pipeline: {nlp.pipeline}")
    pipeline = make_pipeline(
        nlp=nlp,
        n_process=args.num_workers,
        batch_size=args.batch_size,
        n_sents=args.num_sentences
    )

    # laod the dataset
    df = pd.read_csv(args.input_path)
    logger.info(f"Loaded {len(df)} examples.")
    logger.info(pformat(df.head()))

    text_it = yield_column(df, "text")
    tokens = [
        list(tokens) for tokens in tqdm(pipeline(texts=text_it), desc="Preprocessing", total=len(df))
    ]

    logger.info(f"Preprocessing finished.")
    new_df = df.loc[:, ["event_type"]]
    new_df["tokens"] = tokens
    new_df.to_csv(args.output_path)


if __name__ == '__main__':
    main()
