# Runner

The Runner of DanLing sets up the basic environment for running neural networks.

## Components

For cross-platform compatibilities, DanLing features a two-level Runner + Config system.

### PlatformRunner

PlatformRunner implements platform-specific features like `step` and `prepare`.

The Runner contains all runtime information that is irrelevant to the checkpoint (e.g. `world_size`, `rank`, etc.). All other information should be saved in `Config`.

Currently, only [`AccelerateRunner`][danling.runner.AccelerateRunner] is supported.

### [`BaseRunner`][danling.runner.BaseRunner]

[`BaseRunner`][danling.runner.BaseRunner] defines shared attributes and implements platform-agnostic features, including `init_logging`, `results` and `scores`.

### [`Config`][danling.runner.Config]

[`Config`][danling.runner.Config] stores the state of a run (e.g. `epoch`, `run_id`, `network`, etc.).

With `Config` and corresponding weights, you can resume a run from any point.
Therefore, all members in `Config` will be saved in the checkpoint, and thus should be json serialisable.

## Experiments Management

DanLing Runner is designed for a 3.5-level experiments management system: **Project**, **Group**, **Experiment**, and, **Run**.

### Project

A project corresponds to your project.

Generally speaking, there should be only one project for each repository.

`project_root` is the root directory of all experiments of a certain project, and should be consistent across the project.

### Group

A group groups multiple experiments with similar characteristics.

For example, if you run multiple experiments on learning rate, you may want to group them into a group.

Note that Group is a virtual level (which is why it only counts 0.5) and does not correspond to anything.
There are no attributes/properties for groups.

### Experiment

An experiment is the basic unit of experiments.

Each experiment corresponds to a certain commit, which means the code should be consistent across the experiment.

DanLing will automatically generate `experiment_id` and `experiment_uuid` based on git revision.
They are unique for each commit.

You may also set a catchy custom `experiment_name` to identify each experiment.

### Run

A run is the basic unit of runnings.

Run corresponds to a certain run of an experiment, each run may have different hyperparameters.

DanLing will automatically generate `run_id` and `run_uuid` based on `experiment_uuid` and provided config.
They are unique for each commit and config.

You may also set a catchy custom `run_name` to identify each experiment.

### Identifiers

DanLing has two properties built-in to help you identify each run.

- `id` by default is the join of `experiment_id`, `run_id`, and `uuid`. It is automatically generated hex-strings and is unique for each run.
- `name` by default is `experiment_name-run_name`. It is manually specified and easy to read. Note that `name` is not guaranteed to be unique.

### Directories

To help you manage your experiments, DanLing will automatically generate directories for you.

`dir` is the directory of a certain run, defaults to `{dir/name-id}`.
All run files should be under this directory.

In particular, `checkpoint_dir`, which defaults to `dir/checkpoint_dir_name` contains all checkpoint files.

As a result, your `project_root` should looks like following:

```bash
- {project_root}
-     |- {name}-{id} (equivalents to {experiment_name}-{run_name}-{experiment_id}-{run_id}-{uuid})
-       |
-       |- {checkpoint_dir_name}
-       |    |
-       |    |- best.pth
-       |    |- latest.pth
-       |    |- epoch-10.pth
-       |
-       |- run.log
-       |- runner.yaml
-       |- results.json
-       |- latest.json
-       |- best.json
```
