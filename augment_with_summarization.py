import argparse
import logging
from pprint import pprint

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline

logger = logging.getLogger(__name__)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("CLI args")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Generative model which will be used for summarization"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed which controls randomness during text generation (if any)"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to a csv file containing the dataset to be augmented."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to which the augmented dataset will be saved."
    )
    parser.add_argument(
        "--device",
        type=int,
        required=True,
        help="Device to use during inference. -1 defaults to CPU, while >=0 specifies a GPU number."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If set, logging level will be set to info."
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=20,
        help="Minimum summarization length. Defaults to 20."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=100,
        help="Maximum summarization length. Defaults to 100."
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=4,
        help="Number of beams to use for beam search. Defaults to 4."
             "Setting this value to 1 means no beam search."
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
    logger.info(f"Loaded {len(df)} examples.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        usa_fast=True
    )
    summarizer = pipeline(
        "summarization",
        model=args.pretrained_model_name_or_path,
        tokenizer=tokenizer,
        device=args.device,
        framework="pt"
    )

    def summarize_text(text: str) -> str:
        return summarizer(
            text,
            min_length=args.min_length,
            max_length=args.max_length,
            num_beams=args.num_beams,
            do_sample=False,
            truncation=True,
        )[0]["summary_text"]  # I guess this is the same for all models?

    summary_df = df.loc[:, ["text", "event_type"]]
    tqdm.pandas()
    summary_df.loc[:, "text"] = summary_df["text"].progress_apply(summarize_text)

    df_to_save = pd.concat((df.loc[:, ["text", "event_type"]], summary_df))
    logging.info(f"Length of concatenated dataset: {len(df_to_save)}")
    logging.info(pprint(df_to_save.head()))
    logging.info(f"Columns: {df_to_save.columns}")

    df_to_save.to_csv(args.output_path, index=False)


if __name__ == '__main__':
    main()
