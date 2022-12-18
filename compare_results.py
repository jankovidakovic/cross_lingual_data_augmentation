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
        "--result_keys",
        type=str,
        nargs="*",
        required=False,
        default=None,
        help="Keys under which the results are stored in loaded files."
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

    with open(args.old_results_path, "r") as f:
        old_results = json.load(f)

    with open(args.new_results_path, "r") as f:
        new_results = json.load(f)

    if args.result_keys:
        for key in args.result_keys:
            old_results = old_results[key]
            new_results = new_results[key]

    logger.info(f"Old results: {pformat(old_results)}")
    logger.info(f"News results: {pformat(new_results)}")

    def get_delta_dict(key: str, metrics: Iterable[str] = ("precision", "recall", "f1-score")):
        logger.info(f"Computing delta dict for key: {key}")
        return {
            "metrics": dict(zip(metrics, map(lambda metric: {
                    "old": old_results[key][metric],
                    "new": new_results[key][metric],
                    "delta": new_results[key][metric] - old_results[key][metric],
                }, metrics))),
            "support": old_results[key]["support"]
        }

    keys = list(filter(lambda x: x != "accuracy", old_results))
    logger.info(f"Keys: {keys}")

    comparison = dict(zip(
        keys,
        map(get_delta_dict, keys)
    ))

    # logger.info(pformat(comparison))

    with open(args.output_path, "w") as f:
        json.dump(comparison, f, indent=2)

    logger.info(f"Comparison saved to {args.output_path}")


if __name__ == '__main__':
    main()
