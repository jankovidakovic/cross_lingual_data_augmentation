from argparse import ArgumentParser
from src.constants import MODEL_CLASSES, DATASET_INITS


def get_parser():
    parser = ArgumentParser()

    # DATA PARAMS 1)
    parser.add_argument(
        "--train_filename",
        type=str,
        required=True,
        help="Filesystem path to a file containing the training set. Relative "
             "paths should work, but should be used with caution."
    )
    parser.add_argument(
        "--dev_filename",
        type=str,
        required=False,
        help="Filesystem path to a file containing the development set. If not "
             "provided, development set will not be used for evaluation during "
             "training. Relative paths should work, but should be used with caution."
    )
    parser.add_argument(
        "--num_labels",
        type=int,
        default=None,
        required=True,
        help="Number of unique labels in dataset. If not provided, will "
             "be calculated as the number of unique values in labels column "
             "in the training dataset. Name of the column containig labels can "
             "be set using the '--label_column' option.",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        required=True,
        choices=DATASET_INITS.keys(),
        help="Dataset type to use."
    )  # TODO - abstract this to some kind of controller layer
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=MODEL_CLASSES.keys(),
        help="Type of the model to use. Model type uniquely identifies types "
             "of config, tokenizer and model."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model, a model checkpoint, or the "
             "name of a model which is available in Huggingface Model Hub."
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path. If not provided, will default to "
             "value of '--pretrained_model_name_or_path'.",
    )
    parser.add_argument(
        "--tokenizer_name",
        default=None,
        type=str,
        help="Pretrained tokenizer name or path. If not provided, will default to "
             "value of '--pretrained_model_name_or_path'.",
    )

    # hyperparams
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="If set, input text will be lowercased during tokenization. "
             "This flag is useful when one is using uncased models (e.g. 'bert-base-uncased')",
    )
    parser.add_argument(
        "--max_seq_length",
        default=None,
        type=int,
        help="Maximum sequence length after tokenization, i.e. maximum "
             "number of tokens that one data example should consist of. "
             "Sequences with less tokens will be padded, sequences with "
             "more tokens will be truncated. If not provided, defaults to "
             "the maximum sequence length allowed by the model (which is "
             "specified by '--pretrained_model_name_or_path')."

    )
    parser.add_argument(
        "--num_train_epochs",
        default=None,
        type=int,
        help="Total number of training epochs to perform. Defaults to None.",  # TODO
    )
    parser.add_argument(
        "--max_steps",
        default=None,
        type=int,
        help="Total number of training steps to perform. One step is "
             "defined as one gradient update (backward pass). If provided, "
             "overrides value of '--num_train_epochs'."
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        default=2,
        type=int,
        help="Batch size used during training (per device). Defaults to 2.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        default=2,
        type=int,
        help="Batch size used during evaluation (per device). Defaults to 2.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing an "
             "update to the model's parameters. Defaults to 1. Useful for "
             "using batch sizes bigger than allowed by the available GPU memory."
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-5,
        type=float,
        help="The initial learning rate used for optimization. Defaults to 1e-5.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        default="linear",
        choices=["linear", "linear_halve", "cosine",
                 "cosine_hard_restart", "constant"],
        type=str,
        help="The learning rate schedule type for Adam. Defaults to linear",
    )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay factor. Defaults to 0 (no weight decay)."
    )
    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help="Amount of steps during which the learning rate will be linearly increased. "
             "Defaults to 0 (no warmup).",

    )
    parser.add_argument(
        "--warmup_ratio",
        default=0,
        type=float,
        help="Proportion of total number of training steps during which the learning rate "
             "will be linearly increased. Useful when you don't know the exact number of steps,"
             "but still want to warmup e.g. for first 10% of steps. Defaults to 0 (no warmup).",
    )
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="Stability factor used in ADAM Optimizer, used to mitigate zero-division errors. "
             "Defaults to 1e-8."
    )
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Maximum value of L2-norm of the gradients during optimization. Gradients "
             "with norm greater than this value will be clipped. Defaults to 1.0."
    )

    # runtime meta args
    parser.add_argument(
        "--gpus",
        type=str,
        nargs="+",
        default=[0],
        help="List of GPUs to use. Defaults to '[0]', which means that "
             "model will be trained on GPU:0 only."
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
        help="Logging interval in steps. Defaults to 50."
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=None,
        help="Evaluation interval in steps. If unset, evaluatiion will not be performed."
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=None,
        help="Saving interval in steps. Defaults to None.",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        required=False,
        default=3,
        help="Maximum number of checkpoints that will be saved.",
    )
    parser.add_argument(
        "--metric_for_best_model",
        type=str,
        required=True,
        choices=["f1_macro", "precision", "recall", "loss"],
        help="Metric used to compare model checkpoints. Checkpoints will be sorted according "
             "to the value of provided metric on the development sets."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of workers to use for dataloading. For best performance, set "
             "this to the number of available CPU cores."
    )

    # directories relevant for IO
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and "
             "checkpoints will be written."
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Path to a directory containing cached models (downloaded from hub)."
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs",
        help="Directory used to store the run logs. Defaults to './logs', relative "
             "to the working directory."
    )

    # misc
    parser.add_argument(  # TODO - make use of this
        "--seed",
        type=int,
        default=42,
        help="Random seed. Will be used to perform dataset splitting, as well as "
             "random parameter initialization within the model. Defaults to 42."
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        required=True,
        help="Project name to use for W&B logging."
    )
    parser.add_argument(
        "--wandb_run",
        type=str,
        required=True,
        help="Run name used to associated the experiment with W&B run."
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If set, logging level is set to INFO. Else, it's set to WARN."
    )

    return parser
