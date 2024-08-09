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

import sys

sys.path.insert(0, "demo")

from torch_mnist import MNISTConfig, MNISTRunner  # noqa: E402


class Test:
    config = MNISTConfig().boot()
    runner = MNISTRunner(config)

    def test_train(self):
        self.runner.train()

    def test_evaluate(self):
        self.runner.evaluate(["val"])
