import argparse
import logging

from transformers import AutoTokenizer

from src.data import setup_cnn, setup_docee
from src.multi_task_learning import setup_models

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(description="Multi-task learning argument parser")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="ainize/bart-base-cnn",
    )  # could also do bart-large
    parser.add_argument(
        "--docee_train_path",
        type=str,
        required=True,
        help="Filesystem path to a CSV file containing the Docee train set."
    )
    parser.add_argument(
        "--docee_eval_path",
        type=str,
        required=True,
        help="Filesystem path to a CSV file containing the Docee eval set."
    )
    parser.add_argument(
        "--summ_train_size",
        type=int,
        default=None,
        required=False,
        help="If provided, only that much summarization examples will be used for training."
    )
    parser.add_argument(
        "--summ_eval_size",
        type=int,
        default=None,
        required=False,
        help="If provided, only that much summarization examples will be used for evaluation."
    )
    parser.add_argument(
        "--cls_train_size",
        type=int,
        default=None,
        required=False,
        help="If provided, only that much classification examples will be used for training."
    )
    parser.add_argument(
        "--cls_eval_size",
        type=int,
        default=None,
        required=False,
        help="If provided, only that much classification examples will be used for evaluation."
    )
    return parser


def main():
    args = get_parser().parse_args()

    # setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
        ],
    )

    logger.info(f"Command line arguments: {args}")

    # set up the tokenizer and the model
    logger.info(f"Creating the tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)
    logger.info(f"Tokenizer successfully created: {args.pretrained_model_name_or_path}")



    # setup datasets
    logger.info(f"Setting up summarization dataset [CNN]...")
    cnn_train, cnn_eval = setup_cnn(
        tokenizer=tokenizer,
        train_size=args.summ_train_size,
        eval_size=args.summ_eval_size
    )
    logger.info(f"CNN/DailyMail dataset successfully set up.")

    logger.info(f"Setting up classification dataset [Docee]...")
    docee_train, docee_eval = setup_docee(
        tokenizer=tokenizer,
        train_path=args.docee_train_path,
        eval_path=args.docee_eval_path,
        train_size=args.cls_train_size,
        eval_size=args.cls_eval_size
    )
    logger.info(f"Successfully set up Docee.")

    logger.info(f"Creating models...")
    models = setup_models(args.pretrained_model_name_or_path)
    logger.info(f"Models successfully created.")

    # create trainable tasks


if __name__ == '__main__':
    main()
