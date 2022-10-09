import logging
import os
import sys
from argparse import ArgumentParser

import pandas as pd
from sklearn.model_selection import train_test_split


def get_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed used to perform the dataset split."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/docee",
        help="Path to output directory."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If set, logging level will be set to info."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        help="Path to the dataset to be split."
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.5,
        help="Percentage of data that will be split into test set."
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

    # check if the given random seed split already exists
    split_path = os.path.join(args.output_dir, str(args.random_seed))
    if os.path.isdir(split_path):
        logging.warning(f"Split by given random seed ({args.random_seed}) already "
                        f"exists in path '{os.path.abspath(split_path)}', exiting.")
        sys.exit(0)

    # load test file
    df = pd.read_csv(args.input_file)
    dev_df, test_df = train_test_split(df, test_size=args.test_size)

    # save splits
    os.makedirs(split_path, exist_ok=True)
    dev_path = os.path.join(split_path, "dev.csv")
    dev_df.to_csv(dev_path)
    logging.info(f"Saved development set to file: {os.path.abspath(dev_path)}")
    test_path = os.path.join(split_path, "test.csv")
    test_df.to_csv(test_path)
    logging.info(f"Saved test set to file: {os.path.abspath(test_path)}")

    logging.info(f"Splitting complete.")


if __name__ == "__main__":
    main()
