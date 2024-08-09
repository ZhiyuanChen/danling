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

from chanfig import NestedDict

import danling as dl


class Runner(dl.BaseRunner):
    conflict: bool = True

    def init_distributed(self) -> None:
        pass


class Config(dl.Config):
    __test__ = False

    def __init__(self):
        self.network.name = "resnet18"
        self.dataset.download = True
        self.dataset.root = "data"
        self.dataloader.batch_size = 8
        self.epoch_end = 2
        self.optim.name = "adamw"
        self.optim.lr = 1e-3
        self.optim.weight_decay = 1e-4
        self.log = False
        self.tensorboard = False
        self.gradient_clip = False
        self.log_interval = None
        self.save_interval = None
        self.train_iterations_per_epoch = 64
        self.val_iterations_per_epoch = 16
        self.score_split = "val"
        self.score = "loss"
        self.conflict = 1


class Test:
    config = Config()
    runner = Runner(config)

    def test_results(self):
        runner = self.runner
        runner.results = NestedDict(
            {
                0: {
                    "val": {
                        "loss": 1.0,
                        "acc": 0.0,
                    },
                },
                1: {
                    "val": {
                        "loss": 0.5,
                        "acc": 0.5,
                    },
                },
                2: {
                    "val": {
                        "loss": 0.2,
                        "acc": 0.8,
                    },
                },
                3: {
                    "val": {
                        "loss": 0.6,
                        "acc": 0.4,
                    },
                },
            }
        )
        assert runner.best_result.dict() == {
            "index": 2,
            "val": {
                "loss": 0.2,
                "acc": 0.8,
            },
        }
        assert runner.latest_result.dict() == {
            "index": 3,
            "val": {
                "loss": 0.6,
                "acc": 0.4,
            },
        }
        assert runner.best_score == 0.2
        assert runner.latest_score == 0.6

    def test_conflict(self):
        runner = self.runner
        config = runner.config
        runner.conflict = False
        assert not runner.conflict
        assert config.conflict == 1
