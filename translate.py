from argparse import ArgumentParser, Namespace
import os
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import logging
from enum import Enum
from typing import Generator, Callable

import pandas as pd
from deep_translator import GoogleTranslator


SUPPORTED_TRANSLATORS = ["google"]


def get_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        default="data/docee/18091999/dev.csv",
        help="Filesystem path to the input dataset"
    )

    parser.add_argument(
        "--fields_to_translate",
        type=str,
        nargs="+",
        default="text",
        help="Names of columns to translate"
    )

    parser.add_argument(
        "--source_language",
        type=str,
        # default="auto",
        default="en",
        help="Source language to translate from. It is assumed that the "
        "data is written in the source language. If unset, defaults to "
        "'auto', which uses the concrete translator to automatically "
        "detect the language"
    )

    parser.add_argument(
        "--target_language",
        type=str,
        # required=True
        default="hr",
        help="Target language for translation."
    )

    parser.add_argument(
        "--translator",
        type=str,
        choices=SUPPORTED_TRANSLATORS,
        default="google",
        help="Translator to use. Defaults to 'google', which uses "
        "Google Translator API."
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default="data/docee/dev_translated.csv",
        help="Output file to which to save the translated dataset."
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to show verbose logging"
    )

    return parser


def validate_args(args: Namespace) -> None:
    # check that the input file exists
    if not os.path.exists(args.input_file):
        raise RuntimeError(f"Input file '{args.input_file}' does not exist")

    # check that the output file does not exist
    if os.path.exists(args.output_file):
        raise RuntimeError(f"Output file '{args.output_file}' already exists")

    # check that the translator is supported
    if args.translator not in SUPPORTED_TRANSLATORS:
        raise RuntimeError(f"Translator '{args.translator}' is not supported")


def get_translator_init(translator_name: str):
    if translator_name == "google":
        return GoogleTranslator

# each long document strategy should have:
# - a function that takes in a long document and returns a result to be translated
# - a function that takes as input translated things and joins them


def maybe_truncate(
    text: str,
    max_length: int = 5000
) -> Generator[str, None, None]:
    if len(text) > max_length:
        logging.warning(
            f"Length of text is {len(text)} and the maximum "
            f" allowed is {max_length}. Document will be truncated."
        )
    yield text[:(max_length-1)]  # is this the correct return type then


class LongDocumentStrategy(Enum):
    # NAME = (f, g)
    # f :: Foldable t => String -> t String
    # g :: Foldable t => t String -> String
    TRUNCATE = (maybe_truncate, lambda gen: "".join(s for s in gen))


def translation_pipeline(
    long_document_strategy: (
        Callable[[str], Generator[str, None, None]],
        Callable[[Generator[str, None, None]], str]
    ),
    translator
) -> str:
    generate_splits, join_splits = long_document_strategy

    def translate(text: str):
        # process the long document
        splits_gen = generate_splits(text)
        translated_gen = (translator.translate(s) for s in splits_gen)
        output = join_splits(translated_gen)
        print(f"Translated output: {output}")
        return output

    return translate


def main():
    args = get_parser().parse_args()
    validate_args(args)

    logging.root.handlers = []
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        # log info only on main process
        level=logging.INFO if args.verbose else logging.WARNING,
        handlers=[
            logging.StreamHandler(),
        ],
    )

    # load the dataset from csv file to a pandas dataframe
    df = pd.read_csv(args.input_file)
    logging.info(
        f"Loaded dataset from '{args.input_file}', contains {len(df)} rows")

    # initialize the translator
    translator_init = get_translator_init(args.translator)
    translator = translator_init(
        source=args.source_language,
        target=args.target_language
    )

    logging.info(f"Loaded translator '{args.translator}'")

    # create a translation pipeline
    pipeline = translation_pipeline(
        long_document_strategy=LongDocumentStrategy.TRUNCATE.value,
        translator=translator
    )

    # translate the fields
    tqdm.pandas()  # able to show progress
    with logging_redirect_tqdm():
        for field in args.fields_to_translate:
            logging.info(f"Translating field '{field}'")
            # create a translation pipeline
            df[field] = df[field].progress_apply(pipeline)  # does this work?
            # somethings broken here
            logging.info(f"Translated field '{field}'")

    # save the translated dataset to csv file
    df.to_csv(args.output_file, index=False)
    logging.info(f"Saved translated dataset to '{args.output_file}'")


if __name__ == '__main__':
    main()
