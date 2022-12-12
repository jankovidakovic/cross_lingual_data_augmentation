import json
import logging
from argparse import ArgumentParser
from pprint import pformat
from typing import Iterable

logger = logging.getLogger(__name__)


def get_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--old_results_path",
        type=str,
        required=True,
        help="Filesystem path to old results"
    )
    parser.add_argument(
        "--new_results_path",
        type=str,
        required=True,
        help="Filesystem path to new results"
    )
    parser.add_argument(
        "--results_key",
        type=str,
        required=False,
        default=None,
        help="Key under which the results are stored in loaded files."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to which the comparison result will be saved."
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

    with open(args.old_results, "r") as f:
        old_results = json.load(f)

    with open(args.new_results, "r") as f:
        new_results = json.load(f)

    if args.results_key:
        old_results = old_results[args.results_key]
        new_results = new_results[args.results_key]

    def get_delta_dict(key: str, metrics: Iterable[str] = ("precision", "recall", "f1-score")):
        return dict(zip(metrics, map(lambda metric: {
            "old": old_results[key][metric],
            "new": new_results[key][metric],
            "delta": new_results[key][metric] - old_results[key][metric],
            "support": old_results[key]["support"]
        }, metrics)))

    keys = filter(lambda x: x != "accuracy", old_results)

    comparison = dict(zip(
        keys,
        map(get_delta_dict, keys)
    ))

    logger.info(pformat(comparison))

    with open(args.output_path, "w") as f:
        json.dump(comparison, f, indent=2)

    logger.info(f"Comparison saved to {args.output_path}")


if __name__ == '__main__':
    main()
