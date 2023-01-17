import logging
import os
from dataclasses import dataclass, field
from itertools import chain, tee
from pprint import pformat
from typing import Any

import evaluate
import numpy as np
import torch
from accelerate import Accelerator
from torch import nn
from torch.optim import Optimizer, AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    PreTrainedModel,
    DataCollator,
    PreTrainedTokenizer,
    BartForSequenceClassification,
    BartForConditionalGeneration,
)

from src.summarization import postprocess_for_rouge
from src.utils import check_shared_weights

logger = logging.getLogger(__name__)


@dataclass
class TrainableTask:
    name: str
    model: PreTrainedModel
    train_dataloader: DataLoader
    eval_dataloader: DataLoader
    accelerator: Accelerator
    optimizer: Optimizer
    # TODO - scheduler
    lr_scheduler: Any = field(init=False)

    def __post_init__(self):
        # acceleration, babyyy
        self.model = self.accelerator.prepare_model(self.model)
        self.optimizer = self.accelerator.prepare_optimizer(self.optimizer)
        self.train_dataloader = self.accelerator.prepare_data_loader(
            self.train_dataloader
        )
        self.eval_dataloader = self.accelerator.prepare_data_loader(
            self.eval_dataloader
        )


def prepare_task(
    name: str,
    model: PreTrainedModel,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    learning_rate: float,  # comes from optimizer
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    gradient_accumulation_steps: int,
    collate_fn: DataCollator,
) -> TrainableTask:
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=per_device_eval_batch_size,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
    return TrainableTask(
        name=name,
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        accelerator=accelerator,
    )


def setup_models(pretrained_model_name_or_path: str):
    classification_model = BartForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path,
        num_labels=59  # number of docee labels
    )
    logger.info(f"===== Classification model =====")
    logger.info(pformat(classification_model))

    summarization_model = BartForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path
    )
    logger.info(f"===== Summarization model =====")
    logger.info(pformat(summarization_model))

    # make models share the embeddings, encoder, and decoder
    logger.info(
        f"Models will share weights of the following layers: 'shared', 'encoder' and 'decoder'"
    )
    summarization_model.model.shared = classification_model.model.shared
    summarization_model.model.encoder = classification_model.model.encoder
    summarization_model.model.decoder = classification_model.model.decoder

    check_shared_weights(
        summarization_model,
        classification_model,
        ["model.shared", "model.encoder", "model.decoder"],
    )

    return {
        "summarization": summarization_model,
        "classification": classification_model,
    }


# we cannot partial on anything except "name"
#   -> collate_fn depends on the tokenizer
#   -> datasets also depend on the tokenizer
#       -> those two things could be done at the same place


def set_train(train_mode: bool, *tasks):
    for task in tasks:
        task.model.train(train_mode)


def evaluate_summarization(
    global_step: int,
    eval_dataloader: DataLoader,
    accelerator: Accelerator,
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    max_gen_length: int
):
    rouge_score = evaluate.load("rouge")

    for step, batch in tqdm(
        enumerate(eval_dataloader),
        desc=f"[GLOBAL_STEP = {global_step}] Evaluating summarization performance",
        total=len(eval_dataloader),
    ):
        with torch.no_grad():
            # generate summaries
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=max_gen_length
            )  # aha! we can plug the generation parameters here

            # pad to max length
            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            labels = batch["labels"]

            # If we did not pad to max length, we need to pad the labels too
            labels = accelerator.pad_across_processes(
                labels, dim=1, pad_index=tokenizer.pad_token_id
            )

            generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
            labels = accelerator.gather(labels).cpu().numpy()

            # Replace -100 in the labels as we can't decode them
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]

            # decode generated tokens into words (predicted summaries)
            decoded_preds = tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )
            # decode labels into summaries
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            # preprocess for rouge
            decoded_preds, decoded_labels = postprocess_for_rouge(
                decoded_preds, decoded_labels
            )

            rouge_score.add_batch(predictions=decoded_preds, references=decoded_labels)

    # Compute metrics
    result = rouge_score.compute()

    # Extract the median ROUGE scores
    result = {key: round(value * 100, 4) for key, value in result.items()}
    logger.info(f"[GLOBAL_STEP={global_step}] ======= Evaluation results =======")
    logger.info(pformat(result))
    logger.info("Evaluation complete.")


def evaluate_classification(
    global_step: int,
    eval_dataloader: DataLoader,
    accelerator: Accelerator,
    model: nn.Module,
):
    metrics = {
        "f1": evaluate.load("f1"),
        "precision": evaluate.load("precision"),
        "recall": evaluate.load("recall"),
    }
    for batch in tqdm(
        eval_dataloader,
        total=len(eval_dataloader),
        desc=f"[GLOBAL_STEP={global_step}] Evaluating classification performance",
    ):
        # extract outputs
        outputs = accelerator.unwrap_model(model)(**batch)

        # decode logits into labels
        predictions = torch.argmax(outputs["logits"], dim=1)
        predictions = accelerator.gather(predictions).cpu().numpy()

        for metric in metrics.values():
            metric.add_batch(
                predictions=predictions,
                references=batch["labels"].cpu().numpy(),
            )

    logger.info(f"[GLOBAL_STEP={global_step}] ======= Evaluation results =======")
    for metric_name, metric in metrics.items():
        result = metric.compute(average="macro")
        logger.info(pformat(result))

    logger.info("Evaluation complete.")


def save_everything(
    tasks: dict[str, TrainableTask],
    output_dir: str,
    global_step: int,
    tokenizer: PreTrainedTokenizer,
):
    save_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
    for task in tasks:
        model_save_dir = os.path.join(save_dir, task)
        logger.warning(f"Saving {task} model to {os.path.abspath(model_save_dir)}")
        save_model(
            model=tasks[task].model,
            accelerator=tasks[task].accelerator,
            output_dir=model_save_dir,
        )
        logger.warning(
            f"{task} model successfully saved to {os.path.abspath(model_save_dir)}"
        )

    if tasks["classification"].accelerator.is_main_process:
        logger.warning(f"Saving tokenizer to {os.path.abspath(save_dir)}")
        tokenizer.save_pretrained(save_dir)
        logger.warning(f"Successfully saved tokenizer to {os.path.abspath(save_dir)}")


def save_model(model: PreTrainedModel, accelerator: Accelerator, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    logger.warning(f"Saved model checkpoint to {os.path.abspath(output_dir)}")


def train(
    tasks: dict[str, TrainableTask],
    num_epochs: int,
    tokenizer: PreTrainedTokenizer,
    output_dir: str,
    cls_eval_steps: int,
    summ_eval_steps: int,
    save_steps: int,
    max_gen_length: int,
):

    classification_task = tasks["classification"]
    summarization_task = tasks["summarization"]

    data_ratio = (
        len(summarization_task.train_dataloader)
        // len(classification_task.train_dataloader)
        + 1
    )
    if len(summarization_task.train_dataloader) != len(
        classification_task.train_dataloader
    ):
        logger.warning("Two tasks contain different amount of training examples.")
        logger.warning(
            f"Classification task contains {len(classification_task.train_dataloader)} batches of data."
        )
        logger.warning(
            f"Summarization task contains {len(summarization_task.train_dataloader)} batches of data."
        )

        # assume that summ > cls because thats the case with CNN / Docee
        logger.warning(
            f"Classification examples will be duplicated {data_ratio} times."
        )
        logger.warning(
            f"Instead of {len(classification_task.train_dataloader)}, classification dataloader will yield "
            f"{data_ratio * len(classification_task.train_dataloader)} examples."
        )

    global_step = 0
    for epoch in tqdm(range(num_epochs), desc="Epoch", total=num_epochs):

        # load training data, step by step
        iters = {
            "summarization": iter(summarization_task.train_dataloader),
            "classification": chain(
                *tee(iter(classification_task.train_dataloader)), data_ratio
            ),
        }
        progress_bars = {
            "summarization": tqdm(
                range(len(summarization_task.train_dataloader)),
                desc=f"Summarization progress in epoch {epoch+1}",
                total=len(summarization_task.train_dataloader),
            ),
            "classification": tqdm(
                range(data_ratio * len(classification_task.train_dataloader)),
                desc=f"Classification progress in epoch {epoch+1}",
                total=min(
                    len(summarization_task.train_dataloader),
                    data_ratio * len(classification_task.train_dataloader),
                ),
            ),
        }

        set_train(True, *tasks.values())
        # tu nesto nece bit dobro zbog kopiranja, idk
        num_epoch_steps = len(summarization_task.train_dataloader) * 2

        for step in range(num_epoch_steps):
            global_step += 1

            if step % 2 == 0:  # train summarization
                task = "summarization"
            else:
                task = "classification"
            batch = next(iters[task])
            with tasks[task].accelerator.accumulate(tasks[task].model):
                outputs = tasks[task].model(**batch)
                loss = outputs.loss
                tasks[task].accelerator.backward(loss)
                tasks[task].optimizer.step()
                # tasks[task].lr_scheduler.step()
                tasks[task].optimizer.zero_grad()
                progress_bars[task].update(1)

            if global_step % (cls_eval_steps * 2) == 0:
                set_train(False, tasks["classification"])
                evaluate_classification(
                    global_step=global_step,
                    eval_dataloader=tasks["classification"].eval_dataloader,
                    accelerator=tasks["classification"].accelerator,
                    model=tasks["classification"].model,
                )
                set_train(True, tasks["classification"])

            if global_step % (summ_eval_steps * 2) == 0:
                set_train(False, tasks["summarization"])
                evaluate_summarization(
                    global_step=global_step,
                    eval_dataloader=tasks["summarization"].eval_dataloader,
                    accelerator=tasks["summarization"].accelerator,
                    model=tasks["summarization"].model,
                    tokenizer=tokenizer,
                    max_gen_length=max_gen_length
                )
                set_train(True, tasks["summarization"])

            if global_step % (save_steps * 2) == 0:
                # * 2 because global step counts both tasks
                save_everything(
                    tasks=tasks,
                    output_dir=output_dir,
                    global_step=global_step,
                    tokenizer=tokenizer,
                )

        # deep learning would be so cool to do with monads, no?

        # TODO - loss weighing

        # idea -> instead of alternating batches, we could scale losses
        # idea2 -> GAN setup?
        #   -> generator tries to generate summaries
        #   -> discriminator predicts event types base on those summaries
        #   -> generator wants to generate such that discriminator is able to predict labels easier
        #
        # this would also be expensive AS FUCK to train
        #
        # would this work, and why not?
        #   where are the real/fake examples?

    logger.info(f"Training complete.")
    # save final checkpoint
    save_everything(
        tasks, output_dir=output_dir, global_step=global_step, tokenizer=tokenizer
    )
