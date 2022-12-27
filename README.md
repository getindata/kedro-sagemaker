# Kedro SageMaker Pipelines plugin

[![Python Version](https://img.shields.io/pypi/pyversions/kedro-sagemaker)](https://github.com/getindata/kedro-sagemaker)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![SemVer](https://img.shields.io/badge/semver-2.0.0-green)](https://semver.org/)
[![PyPI version](https://badge.fury.io/py/kedro-sagemaker.svg)](https://pypi.org/project/kedro-sagemaker/)
[![Downloads](https://pepy.tech/badge/kedro-sagemaker)](https://pepy.tech/project/kedro-sagemaker)

[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=getindata_kedro-sagemaker&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=getindata_kedro-sagemaker)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=getindata_kedro-sagemaker&metric=coverage)](https://sonarcloud.io/summary/new_code?id=getindata_kedro-sagemaker)
[![Documentation Status](https://readthedocs.org/projects/kedro-sagemaker/badge/?version=latest)](https://kedro-sagemaker.readthedocs.io/en/latest/?badge=latest)

<p align="center">
  <a href="https://getindata.com/solutions/ml-platform-machine-learning-reliable-explainable-feature-engineering"><img height="150" src="https://getindata.com/img/logo.svg"></a>
  <h3 align="center">We help companies turn their data into assets</h3>
</p>

## About
This plugin enables you to run Kedro projects on Amazon SageMaker. Simply install the package and use the provided `kedro sagemaker` commands to build, push, and run your project on SageMaker.

<img src="./docs/images/sagemaker_running_pipeline.gif" alt="Kedro SageMaker plugin" title="Kedro SageMaker plugin" />


## Documentation 

For detailed documentation refer to https://kedro-sagemaker.readthedocs.io/

## Usage guide

```
Usage: kedro sagemaker [OPTIONS] COMMAND [ARGS]...

Options:
  -e, --env TEXT  Environment to use.
  -h, --help      Show this message and exit.

Commands:
  compile  Compiles the pipeline to a JSON file
  init     Creates basic configuration for Kedro SageMaker plugin
  run      Runs the pipeline on SageMaker Pipelines
```

## Quickstart
Follow **quickstart** section on [kedro-sagemaker.readthedocs.io](https://kedro-sagemaker.readthedocs.io/) to see how to run your Kedro project on AWS SageMaker.
