# DanLing
# Copyright (C) 2022-Present  DanLing

# This program is free software: you can redistribute it and/or modify
# it under the terms of the following licenses:
# - The Unlicense
# - GNU Affero General Public License v3.0 or later
# - GNU General Public License v2.0 or later
# - BSD 4-Clause "Original" or "Old" License
# - MIT License
# - Apache License 2.0

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the LICENSE file for more details.

import torch
from chanfig import Registry
from datasets import load_dataset
from torch import nn, optim
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

import danling as dl

OPTIMIZERS = Registry()
OPTIMIZERS.register(optim.AdamW, "adamw")
OPTIMIZERS.register(optim.SGD, "sgd")


class IMDBConfig(dl.Config):
    epoch_end: int = 2
    log: bool = False
    tensorboard: bool = False
    log_interval: int = 1000
    score_split: str = "val"
    score_name: str = "loss"
    debug: bool = False
    patience: int = 1

    def __init__(self):
        super().__init__()
        self.pretrained = "prajjwal1/bert-tiny"
        self.dataset.path = "stanfordnlp/imdb"
        self.dataloader.batch_size = 8
        self.optim.name = "adamw"
        self.optim.lr = 1e-3
        self.optim.weight_decay = 1e-4
        self.sched.strategy = "cosine"

    def post(self):
        super().post()
        self.transformers = AutoConfig.from_pretrained(self.pretrained)
        self.experiment_name = f"{self.network.name}_{self.optim.name}@{self.optim.lr}"


class IMDBRunner(dl.AccelerateRunner):
    def __init__(self, config: dl.Config):
        super().__init__(config)

        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained)
        self.datasets.train = load_dataset(split="train", **self.dataset)
        self.datasets.val = load_dataset(split="train", **self.dataset)
        # only run on a few samples to speed up testing process
        self.datasets.train._data = self.datasets.train._data[:64]
        self.datasets.val._data = self.datasets.val._data[:64]
        self.datasets.train = self.preprocess_data(self.datasets.train)
        self.datasets.val = self.preprocess_data(self.datasets.val)

        self.model = AutoModelForSequenceClassification.from_config(self.config.transformers)
        self.optimizer = OPTIMIZERS.build(params=self.model.parameters(), **self.optim)
        self.scheduler = dl.optim.LRScheduler(self.optimizer, total_steps=self.total_steps, **self.sched)
        self.criterion = nn.CrossEntropyLoss()

        self.metrics = dl.metrics.binary_metrics()
        self.meters.loss.reset()
        self.meters.time.reset()

    def preprocess_data(self, dataset):
        def tokenization(example):
            example["text"] = self.tokenizer(example["text"], truncation=True, max_length=510)["input_ids"]
            return example

        def transform(data):
            text = dl.NestedTensor(data.pop("text"))
            data["input_ids"] = text.tensor
            data["attention_mask"] = text.mask
            data["labels"] = torch.tensor(data.pop("label"))
            return data

        dataset = dataset.map(tokenization, batched=True)
        dataset.set_transform(transform)
        dataset.__getitems__ = dataset.__getitem__
        return dataset

    def train_step(self, data) -> torch.Tensor:
        with self.autocast(), self.accumulate():
            pred = self.model(**data)
            loss = pred["loss"]
            self.advance(loss)
            self.metrics.update(pred["logits"][:, 0], data["labels"])
        return pred, loss

    def evaluate_step(self, data) -> torch.Tensor:
        pred = self.model(**data)
        loss = pred["loss"]
        self.metrics.update(pred["logits"][:, 0], data["labels"])
        return pred, loss

    @staticmethod
    def collate_fn(batch):
        return batch


if __name__ == "__main__":
    config = IMDBConfig()
    config.parse()
    with dl.debug(config.get("debug", False)):
        runner = IMDBRunner(config)
        runner.train()
        runner.evaluate(["val"])
