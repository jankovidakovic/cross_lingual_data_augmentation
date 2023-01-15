import logging
import os
from dataclasses import dataclass, field
from itertools import chain, tee
from pprint import pformat
from typing import Sequence, Iterable, Any

import evaluate
import numpy as np
import torch
from accelerate import Accelerator
from torch.optim import Optimizer, AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import PreTrainedModel, DataCollator, PreTrainedTokenizer

from src.summarization import postprocess_for_rouge

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
        self.train_dataloader = self.accelerator.prepare_data_loader(self.train_dataloader)
        self.eval_dataloader = self.accelerator.prepare_data_loader(self.eval_dataloader)


def get_classification_dataloader(
        dataset: Dataset,
        shuffle: bool,
): pass  # this would just be the constructor


def prepare_task(
        name: str,
        model: PreTrainedModel,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        learning_rate: float,  # comes from optimizer
        per_device_train_batch_size: int,
        per_device_eval_batch_size: int,
        gradient_accumulation_steps: int,
        collate_fn: DataCollator
) -> TrainableTask:
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=per_device_eval_batch_size,
        collate_fn=collate_fn,
        pin_memory=True
    )
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
    return TrainableTask(
        name=name,
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        accelerator=accelerator
    )

# we cannot partial on anything except "name"
#   -> collate_fn depends on the tokenizer
#   -> datasets also depend on the tokenizer
#       -> those two things could be done at the same place


def set_train(train_mode: bool, tasks: dict[str, TrainableTask]):
    for name, task in tasks.items():
        logger.info(f"setting {name} task to train={train_mode}")
        task.model.train(train_mode)


def train(
        tasks: dict[str, TrainableTask],
        num_epochs: int,
        tokenizer: PreTrainedTokenizer,
        output_dir: str
):
    rouge_score = evaluate.load("rouge")
    classification_f1 = evaluate.load("f1")

    classification_task = tasks["classification"]
    summarization_task = tasks["summarization"]

    data_ratio = len(summarization_task.train_dataloader) // len(classification_task.train_dataloader) + 1
    if len(summarization_task.train_dataloader) != len(classification_task.train_dataloader):
        logger.warning("Two tasks contain different amount of training examples.")
        logger.warning(f"Classification task contains {len(classification_task.train_dataloader)} batches of data.")
        logger.warning(f"Summarization task contains {len(summarization_task.train_dataloader)} batches of data.")

        # assume that summ > cls because thats the case with CNN / Docee
        logger.warning(f"Classification examples will be duplicated {data_ratio} times.")
        logger.warning(f"Instead of {len(classification_task.train_dataloader)}, classification dataloader will yield "
                       f"{data_ratio * len(classification_task.train_dataloader)} examples.")


    for epoch in tqdm(range(num_epochs), desc="Epoch", total=num_epochs):

        # load training data, step by step
        iters = {
            "summarization": iter(summarization_task.train_dataloader),
            "classification": chain(*tee(iter(classification_task.train_dataloader)), data_ratio)
        }
        progress_bars = {
            "summarization": tqdm(
                range(len(summarization_task.train_dataloader)),
                desc=f"Summarization progress in epoch {epoch+1}",
                total=len(summarization_task.train_dataloader)
            ),
            "classification": tqdm(
                range(data_ratio * len(classification_task.train_dataloader)),
                desc=f"Classification progress in epoch {epoch+1}",
                total=min(len(summarization_task.train_dataloader), data_ratio * len(classification_task.train_dataloader))
            )
        }

        set_train(True, tasks)
        # tu nesto nece bit dobro zbog kopiranja, idk
        num_epoch_steps = len(summarization_task.train_dataloader) * 2

        for step in range(num_epoch_steps):
            if step % 2 == 0: # train summarization
                task = "summarization"
            else:
                task = "classification"
            batch = next(iters[task])
            with tasks[task].accelerator.accumulate(tasks[task].model):
                outputs = tasks[task].model(**batch)
                loss = outputs.loss
                tasks[task].accelerator.backward(loss)
                tasks[task].optimizer.step()
                tasks[task].lr_scheduler.step()
                tasks[task].optimizer.zero_grad()
                progress_bars[task].update(1)

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
        #

        # evaluation at the end of epoch

        ## evaluation
        set_train(False, tasks)

        # evaluate summarization
        # accelerator = tasks["summarization"]["accelerator"]
        eval_dataloader = tasks["summarization"].eval_dataloader
        accelerator = tasks["summarization"].accelerator
        for step, batch in tqdm(
                enumerate(eval_dataloader),
                desc=f"Summarization evaluation in epoch {epoch+1}",
                total=len(eval_dataloader)
        ):
            with torch.no_grad():
                generated_tokens = accelerator.unwrap_model(tasks["summarization"].model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )  # aha! we can plug the generation parameters here

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
                decoded_preds = tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                decoded_preds, decoded_labels = postprocess_for_rouge(
                    decoded_preds, decoded_labels
                )

                rouge_score.add_batch(predictions=decoded_preds, references=decoded_labels)

        # Compute metrics
        result = rouge_score.compute()
        # Extract the median ROUGE scores
        result = {key: value * 100 for key, value in result.items()}
        result = {k: round(v, 4) for k, v in result.items()}
        logger.info(f"[SUMM] Epoch {epoch+1}:", pformat(result))

        epoch_output_dir = os.path.join(output_dir, f"epoch_{epoch}")
        os.makedirs(epoch_output_dir, exist_ok=True)
        summarization_save_dir = os.path.join(epoch_output_dir, "summ")
        os.makedirs(summarization_save_dir, exist_ok=True)
        # Save and upload
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(tasks["summarization"].model)
        unwrapped_model.save_pretrained(
            summarization_save_dir,
            save_function=accelerator.save
        )
        logger.warning(f"Saved summarization checkpoint to {os.path.abspath(summarization_save_dir)}")
        if accelerator.is_main_process:
            tokenizer.save_pretrained(epoch_output_dir)  # ovo treba samo jednom realno
            logger.warning(f"Saved tokenizer config to {os.path.abspath(epoch_output_dir)}")

        # evaluate classification
        eval_dataloader = tasks["classification"].eval_dataloader
        accelerator = tasks["classification"].accelerator
        model = tasks["classification"].model
        for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="[CLS] Evaluation"):
            # extract outputs
            outputs = accelerator.unwrap_model(model)(**batch)

            # decode logits into labels
            predictions = torch.argmax(outputs["logits"], dim=1)
            predictions = accelerator.gather(predictions).cpu().numpy()
            # print(labels)
            classification_f1.add_batch(
                predictions=predictions,
                references=batch["labels"].cpu().numpy(),
            )
        result = classification_f1.compute(average="macro")
        print(f"[CLS] Epoch {epoch+1}: {pformat(result)}")

        classification_output_dir = os.path.join(epoch_output_dir, "cls")
        os.makedirs(classification_output_dir, exist_ok=True)
        # Save and upload
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(classification_output_dir, save_function=accelerator.save)
        logger.warning(f"Saved classification model to {os.path.abspath(classification_output_dir)}")

    logger.info(f"Training complete.")