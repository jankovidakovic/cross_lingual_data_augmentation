import argparse
import logging

from transformers import AutoTokenizer, DataCollatorWithPadding, DataCollatorForSeq2Seq
from transformers.utils import PaddingStrategy

from src.data import setup_cnn, setup_docee
from src.multi_task_learning import setup_models, prepare_task, train

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
    parser.add_argument(
        "--cls_learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate used for classification. Defaults to 2e-5."
    )
    parser.add_argument(
        "--summ_learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate used for summarization. Defaults to 1e-5."
    )
    parser.add_argument(
        "--cls_batch_size_train",
        type=int,
        default=1,
        help="Batch size to use for classification training. Defaults to 1."
    )
    parser.add_argument(
        "--cls_batch_size_eval",
        type=int,
        default=1,
        help="Batch size to use for classification evaluation. Defaults to 1."
    )
    parser.add_argument(
        "--cls_grad_acc_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps to use for classification. Defaults to 1."
    )
    parser.add_argument(
        "--summ_batch_size_train",
        type=int,
        default=1,
        help="Batch size to use for summarization training. Defaults to 1."
    )
    parser.add_argument(
        "--summ_batch_size_eval",
        type=int,
        default=1,
        help="Batch size to use for summarization evaluation. Defaults to 1."
    )
    parser.add_argument(
        "--summ_grad_acc_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps to use for summarization. Defaults to 1."
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=2,
        help="Number of epochs to train for. Defaults to 2."
    )
    parser.add_argument(
        "--cls_eval_steps",
        type=int,
        default=100,
        help="Frequency at which the classification performance will be evaluated."
             "Defaults to 100."
    )
    parser.add_argument(
        "--summ_eval_steps",
        type=int,
        default=100,
        help="Frequency at which the summarization performance will be evaluated."
             "Defaults to 100."
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="Frequency at which the models will be saved. Defaults to 100."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory. Defaults to `./outputs`"
    )

    return parser

# wandb is not a priority, we must get this thing to work


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
    logger.info(f"Setting up the classification task...")
    cls_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=PaddingStrategy.MAX_LENGTH,
        return_tensors="pt"
    )

    classification_task = prepare_task(
        name="classification" ,
        model=models["classification"],
        train_dataset=docee_train,
        eval_dataset=docee_eval,
        learning_rate=args.cls_learning_rate,
        per_device_train_batch_size=args.cls_batch_size_train,
        per_device_eval_batch_size=args.cls_batch_size_eval,
        gradient_accumulation_steps=args.cls_grad_acc_steps,
        collate_fn=cls_collator
    )
    logger.info(f"Classification task successfully set up.")

    logger.info(f"Setting up the summarization task...")
    summ_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=PaddingStrategy.MAX_LENGTH,
        return_tensors="pt"
    )
    summarization_task = prepare_task(
        name="summarization",
        model=models["summarization"],
        train_dataset=cnn_train,
        eval_dataset=cnn_eval,
        learning_rate=args.summ_learning_rate,
        per_device_train_batch_size=args.summ_batch_size_train,
        per_device_eval_batch_size=args.summ_batch_size_eval,
        gradient_accumulation_steps=args.summ_grad_acc_steps,
        collate_fn=summ_collator
    )
    logger.info(f"Summarization task successfully set up.")

    tasks = {
        "classification": classification_task,
        "summarization": summarization_task
    }

    train(
        tasks=tasks,
        num_epochs=args.num_epochs,
        tokenizer=tokenizer,
        output_dir=args.output_dir,
        cls_eval_steps=args.cls_eval_steps,
        summ_eval_steps=args.summ_eval_steps,
        save_steps=args.save_steps
    )

    logger.info(f"Experiment completed successfully.")



if __name__ == '__main__':
    main()
