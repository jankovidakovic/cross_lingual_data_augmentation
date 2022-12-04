import logging
import os
import sys
from argparse import ArgumentParser, Namespace

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


def get_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed used to perform the dataset split."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        help="Path to the dataset to be split."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/docee",
        help="Path to output directory."
    )
    parser.add_argument(
        "--output_files",
        type=str,
        nargs="+",
        help="Names of output files. Each file represents one split. "
        " For each split, exactly one file name should be given."
    )
    parser.add_argument(
        "--split_sizes",
        type=float,
        nargs="+",
        help="Split sizes. Must sum to 1.0"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If set, logging level will be set to info."
    )
    return parser


def validate_args(args: Namespace) -> None:
    # check if the input file exists
    if not os.path.exists(args.input_file):
        raise RuntimeError(
            f"Given input file: '{os.path.abspath(args.input_file)}'"
            " does not exist."
        )

    # check that the output dir exists
    if not os.path.isdir(args.output_dir):
        raise RuntimeError(
            f"Given output directory: '{args.output_dir}' does not exist."
        )

    # check that split sizes sum to <= 1
    split_sum = sum(args.split_sizes)
    if not np.isclose(split_sum, 1.0):
        raise RuntimeError(
            f"Sum of split sizes must be equal to 1.0, but sum is {split_sum}."
        )

    num_output_files = len(args.output_files)
    num_splits = len(args.split_sizes)
    if num_output_files != num_splits:
        raise RuntimeError(
            f"Expected {num_splits} split names, but got {num_output_files}")


def process_split(
    split_df: pd.DataFrame,
    split_name: str,
    split_no: int,
    split_filename: str
) -> None:
    logging.info(
        f"Split no. {split_no} (named {split_name}): "
        f"{len(split_df)} examples."
    )

    # save the split
    split_df.to_csv(split_filename)
    logging.info(
        f"Saved {split_name} to path to path: "
        f"{os.path.abspath(split_filename)}"
    )


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

    # before proceeding, validate input data
    validate_args(args)  # throws if input data is invalid

    # check if the given random seed split already exists
    split_path = os.path.join(args.output_dir, str(args.random_seed))
    if os.path.isdir(split_path):
        logging.warning(f"Split by given random seed ({args.random_seed}) already "
                        f"exists in path '{os.path.abspath(split_path)}', exiting.")
        sys.exit(0)
        # TODO - split is not not determined solely by random seed
        # we can have multiple different splits for the same random seed
        # so we should check the split existence in a smarter way

    # otherwise, create the split path
    os.makedirs(split_path, exist_ok=True)

    # load dataset
    df = pd.read_csv(args.input_file)

    # split
    for i, (split_name, split_size) in enumerate(zip(args.output_files[:-1], args.split_sizes)):
        split_df, df = train_test_split(
            df, train_size=split_size / sum(args.split_sizes[i:]))
        # TODO - stratified split  (ensure that all labels are present in all splits)
        process_split(
            split_df=split_df,
            split_name=split_name,
            split_no=i,
            split_filename=os.path.join(split_path, f"{split_name}.csv")
        )

    process_split(
        split_df=df,
        split_name=args.output_files[-1],
        split_no=len(args.split_sizes) - 1,
        split_filename=os.path.join(split_path, f"{args.output_files[-1]}.csv")
    )

    logging.info("Splitting complete.")


if __name__ == "__main__":
    main()
