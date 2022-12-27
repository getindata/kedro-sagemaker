# Introduction

## About SageMaker Pipelines
[Amazon SageMaker Pipelines](https://aws.amazon.com/sagemaker/pipelines/) is a fully managed service that enables you to build, test, and deploy machine learning (ML) workflows with ease. It provides an easy-to-use Python Software Development Kit (SDK) that allows you to define and execute ML workflows, as well as visualize and manage them using Amazon SageMaker Studio.

One of the key benefits of SageMaker Pipelines is the ability to store and reuse workflow steps, which can help you be more efficient and scale faster. Additionally, SageMaker Pipelines comes with built-in templates to help you quickly set up CI/CD (continuous integration/continuous delivery) in your ML environment, so you can get started quickly and streamline your ML workflows.

## Why to integrate Kedro project with SageMaker Pipelines?

The Kedro Framework is a tool for building and deploying machine learning (ML) pipelines that follows the best standards and practices for ML model development. Once the code is ready, there are now multiple tools available for automating and scaling the delivery of those ML pipelines.

These tools can be used in conjunction with Kedro to execute ML pipelines code using robust services without altering the underlying business logic. 
The use of these tools can provide resource benefits for handling large training datasets or complex and compute intensive models. 

We currently support:
* Azure ML Pipelines [kedro-azureml](https://github.com/getindata/kedro-azureml)
* GCP Vertex AI Pipelines [kedro-vertexai](https://github.com/getindata/kedro-vertexai)
* Kubeflow Pipelines [kedro-kubeflow](https://github.com/getindata/kedro-kubeflow)
* Airflow on Kubernetes [kedro-airflow-k8s](https://github.com/getindata/kedro-airflow-k8s)

With this **kedro-sagemaker** plugin, you can run your Kedro project on Amazon SageMaker Pipelines in a fully managed fashion.
