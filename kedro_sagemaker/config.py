from dataclasses import dataclass
from typing import Dict, Optional

import yaml
from pydantic import BaseModel


@dataclass
class SageMakerMetricsTrackingConfig:
    metrics: dict


class ResourceConfig(BaseModel):
    instance_type: str
    instance_count: int = 1
    timeout_seconds: int = 24 * 60 * 60


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
      instance_type: ml.m5.large
      timeout_seconds: 86400
  sagemaker:
    # (optional) mapping between kedro pipeline names (keys) and SageMaker pipeline names
    # Note that SageMaker does not support underscores in pipeline names.
    # Here you can map for example add `__default__: "my-pipeline"`
    # to make the `__default__` Kedro pipeline appear as `my-pipeline` in SageMaker UI 
    pipeline_names_mapping:
      kedro_pipeline_name: "sagemaker-pipeline-name"
docker:
  image: "{docker_image}"
  working_directory: /home/kedro
""".strip()

# This auto-validates the template above during import
_CONFIG_TEMPLATE = KedroSageMakerPluginConfig.parse_obj(
    yaml.safe_load(CONFIG_TEMPLATE_YAML)
)
