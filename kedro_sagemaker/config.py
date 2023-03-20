from dataclasses import dataclass
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel


@dataclass
class SageMakerMetricsTrackingConfig:
    metrics: dict


class ResourceConfig(BaseModel):
    instance_type: str
    instance_count: Optional[int]
    timeout_seconds: Optional[int]
    security_group_ids: Optional[List[str]]
    subnets: Optional[List[str]]


class SageMakerConfig(BaseModel):
    pipeline_names_mapping: Optional[Dict[str, str]] = None


class DockerConfig(BaseModel):
    image: str
    working_directory: str = "/home/kedro"


class AwsConfig(BaseModel):
    execution_role: str
    bucket: str
    sagemaker: SageMakerConfig
    resources: Dict[str, ResourceConfig]


class KedroSageMakerPluginConfig(BaseModel):
    aws: AwsConfig
    docker: DockerConfig


CONFIG_TEMPLATE_YAML = """
aws:
  # Bucket name to use as a temporary storage within the pipeline job
  bucket: "{bucket}"

  # AWS SageMaker Executor role ARN
  execution_role: "{execution_role}"

  # use Kedro node tags (recommended) or node names to assign compute resources
  # use __default__ to specify the default values (for all nodes)
  resources:
    __default__:
      instance_count: 1
      instance_type: ml.t3.medium
      timeout_seconds: 86400
      security_group_ids: null
      subnets: null
  sagemaker:
    # (optional) mapping between kedro pipeline names (keys) and SageMaker pipeline names
    # Note that SageMaker does not support underscores in pipeline names.
    # Here you can map for example add `__default__: "kedro-sagemaker-default-pipeline"`
    # to make the `__default__` Kedro pipeline appear as `kedro-sagemaker-default-pipeline` in SageMaker UI
    pipeline_names_mapping:
      __default__: "kedro-sagemaker-default-pipeline"
docker:
  image: "{docker_image}"
  working_directory: /home/kedro
""".strip()

# This auto-validates the template above during import
_CONFIG_TEMPLATE = KedroSageMakerPluginConfig.parse_obj(
    yaml.safe_load(CONFIG_TEMPLATE_YAML)
)
