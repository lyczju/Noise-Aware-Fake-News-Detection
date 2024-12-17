# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import pickle
from tqdm import tqdm
# import logging
# logging.basicConfig(level=logging.DEBUG)

import evaluate
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import random_split

# from torch_geometric.data import DataLoader, DataListLoader
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import UPFD
from torch_geometric.transforms import ToUndirected
from torch_geometric.nn import DataParallel
from torch_geometric.nn import global_mean_pool, GATConv
from torch_geometric.utils import to_dense_adj

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)

from accelerate import Accelerator, DataLoaderConfiguration, DistributedType

from models.GIB_model import GIBModel

########################################################################
# This is a fully working simple example to use Accelerate
#
# This example trains a Bert base model on GLUE MRPC
# in any of the following settings (with the same script):
#   - single CPU or single GPU
#   - multi GPUS (using PyTorch distributed mode)
#   - (multi) TPUs
#   - fp16 (mixed-precision) or fp32 (normal precision)
#
# This example also demonstrates the checkpointing and sharding capabilities
#
# To run it in each of these various modes, follow the instructions
# in the readme for examples:
# https://github.com/huggingface/accelerate/tree/main/examples
#
########################################################################

MAX_GPU_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 16


def training_function(config, args):
    # Initialize accelerator
    dataloader_config = DataLoaderConfiguration()
    if args.with_tracking:
        accelerator = Accelerator(
            cpu=args.cpu,
            mixed_precision=args.mixed_precision,
            dataloader_config=dataloader_config,
            log_with="all",
            project_dir=args.project_dir,
        )
    else:
        accelerator = Accelerator(
            cpu=args.cpu,
            mixed_precision=args.mixed_precision,
            dataloader_config=dataloader_config,
        )

    if hasattr(args.checkpointing_steps, "isdigit"):
        if args.checkpointing_steps == "epoch":
            checkpointing_steps = args.checkpointing_steps
        elif args.checkpointing_steps.isdigit():
            checkpointing_steps = int(args.checkpointing_steps)
        else:
            raise ValueError(
                f"Argument `checkpointing_steps` must be either a number or `epoch`. `{args.checkpointing_steps}` passed."
            )
    else:
        checkpointing_steps = None
    # Sample hyper-parameters for learning rate, batch size, seed and a few other HPs
    lr = config["lr"]
    num_epochs = int(config["num_epochs"])
    seed = int(config["seed"])
    batch_size = int(config["batch_size"])

    # We need to initialize the trackers we use, and also store our configuration
    if args.with_tracking:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run, config)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    
    # dataset = FNNDataset(root='data', feature='bert', empty=False, name=args.dataset, transform=ToUndirected())

    # num_training = int(len(dataset) * 0.75)
    # num_val = int(len(dataset) * 0.1)
    # num_test = len(dataset) - (num_training + num_val)
    # train_dataset, val_dataset, test_dataset = random_split(dataset, [num_training, num_val, num_test])
    # train_loader = DataLoader(
    #     train_dataset, shuffle=True, batch_size=batch_size
    # )
    # val_loader = DataLoader(
    #     val_dataset, shuffle=False, batch_size=EVAL_BATCH_SIZE
    # )
    # test_loader = DataLoader(
    #     test_dataset, shuffle=False, batch_size=EVAL_BATCH_SIZE
    # )
    
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'UPFD')
    val_dataset = UPFD(path, args.dataset, args.feature, 'train', ToUndirected())
    test_dataset = UPFD(path, args.dataset, args.feature, 'val', ToUndirected())
    train_dataset = UPFD(path, args.dataset, args.feature, 'test', ToUndirected())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    accuracy_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")

    # def tokenize_function(examples):
    #     # max_length=None => use the model max length (it's actually the default)
    #     outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=None)
    #     return outputs

    # Apply the method we just defined to all the examples in all the splits of the dataset
    # starting with the main process first:
    # with accelerator.main_process_first():
    #     tokenized_datasets = datasets.map(
    #         tokenize_function,
    #         batched=True,
    #         remove_columns=["idx", "sentence1", "sentence2"],
    #     )

    # We also rename the 'label' column to 'labels' which is the expected name for labels by the models of the
    # transformers library
    # tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    # If the batch size is too big we use gradient accumulation
    gradient_accumulation_steps = 1
    if (
        batch_size > MAX_GPU_BATCH_SIZE
        and accelerator.distributed_type != DistributedType.XLA
    ):
        gradient_accumulation_steps = batch_size // MAX_GPU_BATCH_SIZE
        batch_size = MAX_GPU_BATCH_SIZE

    def collate_fn(examples):
        # On TPU it's best to pad everything to the same length or training will be very slow.
        max_length = (
            128 if accelerator.distributed_type == DistributedType.XLA else None
        )
        # When using mixed precision we want round multiples of 8/16
        if accelerator.mixed_precision == "fp8":
            pad_to_multiple_of = 16
        elif accelerator.mixed_precision != "no":
            pad_to_multiple_of = 8
        else:
            pad_to_multiple_of = None

        return tokenizer.pad(
            examples,
            padding="longest",
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )

    # Instantiate dataloaders.
    # train_loader = DataLoader(
    #     train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=batch_size
    # )
    # val_loader = DataLoader(
    #     val_dataset, shuffle=False, collate_fn=collate_fn, batch_size=EVAL_BATCH_SIZE
    # )
    
    set_seed(seed)

    # Instantiate the model (we build the model here so that the seed also control new weights initialization)
    # model = AutoModelForSequenceClassification.from_pretrained(
    #     "bert-base-cased", return_dict=True
    # )
    model = GIBModel()

    # We could avoid this line since the accelerator is set with `device_placement=True` (default value).
    # Note that if you are placing tensors on devices manually, this line absolutely needs to be before the optimizer
    # creation otherwise training will not work on TPU (`accelerate` will kindly throw an error to make us aware of that).
    model = model.to(accelerator.device)

    # Instantiate optimizer
    optimizer = AdamW(params=model.parameters(), lr=lr)

    # Instantiate scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=(len(train_loader) * num_epochs)
        // gradient_accumulation_steps,
    )

    # Prepare everything
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    model, optimizer, train_loader, val_loader, lr_scheduler = (
        accelerator.prepare(
            model, optimizer, train_loader, val_loader, lr_scheduler
        )
    )

    # We need to keep track of how many total steps we have iterated over
    overall_step = 0
    # We also need to keep track of the stating epoch so files are named properly
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[
                -1
            ]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(train_loader)
            resume_step -= starting_epoch * len(train_loader)

    # Now we train the model
    for epoch in tqdm(range(starting_epoch, num_epochs), desc="Training"):
        model.train()
        if args.with_tracking:
            total_loss = 0
        if (
            args.resume_from_checkpoint
            and epoch == starting_epoch
            and resume_step is not None
        ):
            # We need to skip steps until we reach the resumed step
            if not args.use_stateful_dataloader:
                active_dataloader = accelerator.skip_first_batches(
                    train_loader, resume_step
                )
            else:
                active_dataloader = train_loader
            overall_step += resume_step
        else:
            # After the first iteration though, we need to go back to the original dataloader
            active_dataloader = train_loader
        for step, batch in enumerate(active_dataloader):
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            batch.to(accelerator.device)
            output, ib_loss, detect_loss = model(batch)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, batch.y)
            loss += ib_loss + detect_loss
            loss = loss / gradient_accumulation_steps
            # We keep track of the loss at each epoch
            if args.with_tracking:
                total_loss += loss.detach().float()
            accelerator.backward(loss)
            if step % gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            overall_step += 1

            if isinstance(checkpointing_steps, int):
                output_dir = f"step_{overall_step}"
                if overall_step % checkpointing_steps == 0:
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
                    
        def evaluate_model(model, dataloader, type):
            model.eval()
            for step, batch in enumerate(dataloader):
                # We could avoid this line since we set the accelerator with `device_placement=True`.
                batch.to(accelerator.device)
                with torch.no_grad():
                    output, ib_loss, detect_loss = model(batch)
                predictions = output.argmax(dim=-1)
                predictions, references = accelerator.gather_for_metrics(
                    (predictions, batch.y)
                )
                accuracy_metric.add_batch(predictions=predictions, references=references)
                precision_metric.add_batch(predictions=predictions, references=references)
                recall_metric.add_batch(predictions=predictions, references=references)
                f1_metric.add_batch(predictions=predictions, references=references)

            accuracy = accuracy_metric.compute()
            precision = precision_metric.compute(zero_division=1)
            recall = recall_metric.compute(zero_division=1)
            f1 = f1_metric.compute()
            # Use accelerator.print to print only on the main process.
            
            accelerator.print(f"{type} set:", f"epoch {epoch}:", f"accuracy: {accuracy}", f"precision: {precision}", f"recall: {recall}", f"f1: {f1}")
            if args.with_tracking:
                accelerator.log(
                    {
                        "type": type,
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "train_loss": total_loss.item() / len(train_loader),
                        "epoch": epoch,
                    },
                    step=epoch,
                )
        
        evaluate_model(model, val_loader, "Validation")
        evaluate_model(model, test_loader, "Test")

        if checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    accelerator.end_training()


def main():
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )
    parser.add_argument(
        "--cpu", action="store_true", help="If passed, will train on the CPU."
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--use_stateful_dataloader",
        action="store_true",
        help="If the dataloader should be a resumable stateful dataloader.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to load in all available experiment trackers from the environment and use them for logging.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Optional save directory where all checkpoint folders will be stored. Default is the current working directory.",
    )
    parser.add_argument(
        "--project_dir",
        type=str,
        default="logs",
        help="Location on where to store experiment tracking logs` and relevent project information",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="gossipcop",
        choices=["gossipcop", "politifact"],
        help="The dataset to use for training and evaluation.",
    )
    parser.add_argument(
        "--feature",
        type=str,
        default="bert",
        choices=["profile", "spacy", "bert", "content"],
        help="The feature to use for training and evaluation.",
    )
    args = parser.parse_args()
    print("args:", args)
    config = {"lr": 2e-5, "num_epochs": 100, "seed": 42, "batch_size": 16}
    training_function(config, args)


if __name__ == "__main__":
    main()
