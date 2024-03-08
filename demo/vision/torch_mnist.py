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

import torchvision
from chanfig import Registry
from torch import nn, optim

import danling as dl

OPTIMIZERS = Registry()
OPTIMIZERS.register(optim.AdamW, "adamw")
OPTIMIZERS.register(optim.SGD, "sgd")


class MNISTConfig(dl.Config):
    epoch_end = 2
    log = False
    tensorboard = False
    log_interval = 1000
    score_split = "val"
    score_name = "loss"
    debug = False
    patience = 1

    def __init__(self):
        super().__init__()
        self.network.name = "resnet18"
        self.dataset.download = True
        self.dataset.root = "data"
        self.dataloader.batch_size = 8
        self.optim.name = "adamw"
        self.optim.lr = 1e-3
        self.optim.weight_decay = 1e-4
        self.sched.strategy = "cosine"

    def post(self):
        self.copy_class_attributes()
        self.experiment_name = f"{self.network.name}_{self.optim.name}@{self.optim.lr}"


class MNISTRunner(dl.TorchRunner):
    def __init__(self, config: dl.Config):
        super().__init__(config)

        self.dataset.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        self.datasets.train = torchvision.datasets.MNIST(train=True, **self.dataset)
        self.datasets.val = torchvision.datasets.MNIST(train=False, **self.dataset)
        # only run on a few samples to speed up testing process
        self.datasets.train.data = self.datasets.train.data[:64]
        self.datasets.val.data = self.datasets.val.data[:64]

        self.model = getattr(torchvision.models, self.network.name)(pretrained=False, num_classes=10)
        self.model.conv1 = nn.Conv2d(1, 64, 1, bias=False)
        self.optimizer = OPTIMIZERS.build(params=self.model.parameters(), **self.optim)
        self.scheduler = dl.optim.LRScheduler(self.optimizer, total_steps=self.trainable_steps, **self.sched)
        self.criterion = nn.CrossEntropyLoss()

        self.metrics = dl.metrics.multiclass_metrics(num_classes=10)
        self.meters.loss.reset()
        self.meters.time.reset()


if __name__ == "__main__":
    config = MNISTConfig()
    config.parse()
    with dl.debug(config.get("debug", False)):
        runner = MNISTRunner(config)
        runner.train()
        runner.evaluate(["val"])
