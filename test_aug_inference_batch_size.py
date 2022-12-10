import logging
from argparse import ArgumentParser

import pandas as pd
import torch
from transformers import AutoTokenizer, pipeline
from transformers.utils import PaddingStrategy

from src.data import DoceeForInference
from src.utils import do_inference, measure_time

logger = logging.getLogger(__name__)


def get_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--min_batch_size",
        type=int,
        required=False,
        default=1,
        help="Minimum batch size to use."
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        required=False,
        default=1,
        help="Maximum batch size to use."
    )
    parser.add_argument(
        "--device",
        type=int,
        required=True,
        help="Device on which the inference will be performed."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        required=False,
        default=4,
        help="Number of workers. Defaults to 4."
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Filesystem path to input dataset which will be used for inference"
    )

    parser.add_argument(
        "--num_examples",
        type=int,
        required=False,
        default=500,
        help="Number of example to use when testing the batch size. Defaults to 500"
    )

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Name of the model which will be used for inference."
    )

    return parser


def main():
    args = get_parser().parse_args()

    logging.root.handlers = []
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
        ],
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        padding=PaddingStrategy.MAX_LENGTH,
        use_fast=True
    )

    summarizer = pipeline(
        "summarization",
        model=args.pretrained_model_name_or_path,
        tokenizer=tokenizer,
        device=args.device,
        framework="pt"
    )

    logger.info(f"Using model: {args.pretrained_model_name_or_path} on device: {args.device}")

    df = pd.read_csv(args.dataset_path, nrows=args.num_examples)
    logger.info(f"Loaded {len(df)} examples from {args.dataset_path}")

    dataset = DoceeForInference(df)

    low_bs, high_bs = args.min_batch_size, args.max_batch_size
    mid_bs = None

    while low_bs < high_bs:
        mid_bs = (low_bs + high_bs) // 2
        logger.info(f"==========Testing batch size {mid_bs}==========")
        try:
            time, _ = measure_time(do_inference, mid_bs, summarizer, dataset, args.num_workers)
            logger.info(f"Batch size {mid_bs} is okay. Increasing.")
            low_bs = mid_bs + 1
        except RuntimeError as e:
            logger.error(f"Got runtime error: {e}")
            torch.cuda.empty_cache()
            logger.info(f"Batch size {mid_bs} is too big. Reducing.")
            high_bs = mid_bs - 1

    logger.info(f"Optimal batch size for model {args.pretrained_model_name_or_path} is equal to {mid_bs}")


if __name__ == '__main__':
    main()




