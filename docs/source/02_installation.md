# Installation guide

## Prerequisites

* a tool to manage Python virtual environments (e.g. venv, conda, virtualenv).
* Docker

## Kedro setup

First, you need to install base Kedro package

```console
$ pip install ">=0.18.3,<0.19"
```

## Plugin installation

### Install from PyPI

You can install ``kedro-sagemaker`` plugin from ``PyPi`` with `pip`:

```console
pip install --upgrade kedro-sagemaker
```

### Install from sources

You may want to install the develop branch which has unreleased features:

```console
pip install git+https://github.com/getindata/kedro-sagemaker.git@develop
```

## Available commands

You can check available commands by going into project directory and running:

```console
Usage: kedro sagemaker [OPTIONS] COMMAND [ARGS]...

Options:
  -e, --env TEXT  Environment to use.
  -h, --help      Show this message and exit.

Commands:
  compile  Compiles the pipeline to a JSON file
  init     Creates basic configuration for Kedro AzureML plugin
  run      Runs the pipeline on SageMaker Pipelines
```
