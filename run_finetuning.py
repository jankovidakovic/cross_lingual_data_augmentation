import logging
import os
import subprocess
from pprint import pformat

import pandas as pd
import torch.cuda
import wandb
from transformers import TrainingArguments, IntervalStrategy, Trainer

from src.utils import multiclass_cls_metrics
from src.parsers.run_finetuning_parser import get_parser
from src.constants import MODEL_CLASSES, DATASET_INITS


if __name__ == '__main__':
    args = get_parser().parse_args()

    # TODO - setup wandb
    #   project, entity
    #   run config (hyperparams)

    # TODO - pytorch lightning (should make stuff easier hopefully)

    # setup logging
    log_dir = os.path.join(args.log_dir, args.wandb_run)
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(
        args.log_dir,
        args.wandb_run,
        "run.log"  # good enough for now
    )
    logging.root.handlers = []
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        # log info only on main process
        level=logging.INFO if args.verbose else logging.WARN,
        handlers=[
            logging.FileHandler(filename=log_filename, mode="w"),
            logging.StreamHandler(),
        ],
    )
    logging.info(f"Logging to file: {os.path.abspath(log_filename)}")

    # log commit hash for reproducibility
    git_commit_hash = subprocess.check_output(
        ["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    logging.info(f"Commit hash: {git_commit_hash}")

    # initialize wandb run
    wandb_run = wandb.init(
        project=args.wandb_project,
        entity="jankovidakovic",
        name=args.wandb_run
    )

    # setup environment variables for GPUs
    if not torch.cuda.is_available():
        logging.warning(
            f"Cuda is not available. This is probably not intended, stopping the run...")
        raise RuntimeWarning(f"Cuda is not available. Exiting.")

    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args.gpus)
    logging.info(f"Using CUDA devices: {args.gpus}")

    # setup config, tokenizer and model
    model_type = MODEL_CLASSES[args.model_type]
    logging.info(f"Using model type: {args.model_type}")
    # config = model_type.config.from_pretrained(
    #     pretrained_model_name_or_path=args.config_name or args.pretrained_model_name_or_path,
    #     cache_dir=args.cache_dir,
    #     num_labels=args.num_labels
    # )
    tokenizer = model_type.tokenizer.from_pretrained(
        pretrained_model_name_or_path=args.tokenizer_name or args.pretrained_model_name_or_path,
        cache_dir=args.cache_dir,
        do_lower_case=args.do_lower_case
    )
    # setup datasets

    # check that the filesystem info about the data is correct
    if not os.path.exists(args.train_filename):
        logging.error(
            f"Train filename provided ('{args.train_filename}') not found.")
        raise RuntimeError(
            f"Train filename provided ('{args.train_filename}') not found.")
    # create dataset from train filename
    train_df = pd.read_csv(args.train_filename)
    label2id = {
        label: i
        for i, label in enumerate(sorted(train_df.event_type.unique().tolist()))
    }
    logging.info(f"Labels mapped to IDs: {pformat(label2id)}")

    dataset_init = DATASET_INITS[args.dataset_type]
    # TODO - fix optional in callable
    train_dataset = dataset_init(train_df, tokenizer, label2id)
    # create dev dataset if provided
    if not os.path.exists(args.dev_filename):
        logging.error(
            f"Dev filename provided ('{args.dev_filename}') not found.")
        raise RuntimeError(
            f"Dev filename provided ('{args.dev_filename}') not found.")
    # create dataset from dev filename
    dev_df = pd.read_csv(args.dev_filename)
    # TODO - fix static typing
    dev_dataset = dataset_init(dev_df, tokenizer, label2id)

    # load model

    model = model_type.model.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        cache_dir=args.cache_dir,
        label2id=label2id
    )
    print("========== MODEL ARCHITECTURE ==========")
    print(model)


    # initialize trainer
    optional_kwargs = {}
    if args.max_steps:
        optional_kwargs.update({"max_steps": args.max_steps})
    if args.eval_steps:
        optional_kwargs.update({"eval_steps": args.eval_steps})
    if args.save_steps:
        optional_kwargs.update({"save_steps": args.save_steps})
    if args.num_train_epochs:
        optional_kwargs.update({"num_train_epochs": args.num_train_epochs})
    if args.max_grad_norm:
        optional_kwargs.update({"max_grad_norm": args.max_grad_norm})
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy=IntervalStrategy.STEPS if args.eval_steps else IntervalStrategy.EPOCH,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        logging_strategy=IntervalStrategy.STEPS,
        logging_steps=args.logging_steps,
        save_strategy=IntervalStrategy.STEPS if args.save_steps else IntervalStrategy.EPOCH,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.dataloader_num_workers,
        run_name=args.wandb_run,
        load_best_model_at_end=True,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.metric_for_best_model != "loss",
        report_to=["wandb"],
        **optional_kwargs
    )
    trainer = Trainer(
        args=training_args,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=multiclass_cls_metrics
    )
    trainer.train()
    logging.info(f"Training complete.")

    # best model should be loaded, evaluate it for last checkpoint
    trainer.evaluate()
    wandb_run.finish()
