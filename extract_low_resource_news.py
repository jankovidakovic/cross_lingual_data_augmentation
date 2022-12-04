import logging
from argparse import ArgumentParser
from pprint import pformat

import pandas as pd

from src.utils import low_resource_slice

logger = logging.getLogger(__name__)


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "--cutoff",
        type=int,
        help="Cutoff used to determine low-resource classes."
    )
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
        "--verbose",
        action="store_true",
        default=True
    )

    return parser


def main():
    args = get_parser().parse_args()

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

    df = pd.read_csv(args.input_path)

    # extract news
    df = df.loc[~df["date"].isna(), :]

    # extract low-resource, according to cutoff
    low_resource_classes, df = low_resource_slice(df, args.cutoff, return_classes=True)
    logger.info(f"Extracted {len(low_resource_classes)} classes.")
    logger.info(f"Low resource classes: {pformat(low_resource_classes)}")

    # save df
    df.to_csv(args.output_path, index_label="index")
    logger.info(f"Successfully saved to {args.output_path}")


if __name__ == '__main__':
    main()
