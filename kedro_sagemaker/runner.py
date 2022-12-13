import logging
import os
from typing import Any, Dict

from kedro.io import AbstractDataSet, DataCatalog
from kedro.pipeline import Pipeline
from kedro.runner import SequentialRunner
from pluggy import PluginManager
from pydantic import BaseModel

from kedro_sagemaker.constants import (
    KEDRO_SAGEMAKER_RUNNER_CONFIG,
)
from kedro_sagemaker.datasets import (
    CloudpickleDataset,
    DistributedCloudpickleDataset,
)
from kedro_sagemaker.utils import is_distributed_environment


logger = logging.getLogger(__name__)


class KedroSageMakerRunnerConfig(BaseModel):
    bucket: str
    run_id: str


class SageMakerPipelinesRunner(SequentialRunner):
    @classmethod
    def runner_name(cls):
        return f"{cls.__module__}.{cls.__qualname__}"

    def __init__(self, is_async: bool = False):
        super().__init__(is_async)
        self.runner_config_raw = os.environ.get(KEDRO_SAGEMAKER_RUNNER_CONFIG)
        self.runner_config = KedroSageMakerRunnerConfig.parse_raw(
            self.runner_config_raw
        )

    def run(
        self,
        pipeline: Pipeline,
        catalog: DataCatalog,
        hook_manager: PluginManager = None,
        session_id: str = None,
    ) -> Dict[str, Any]:
        unsatisfied = pipeline.inputs() - set(catalog.list())
        for ds_name in unsatisfied:
            catalog = catalog.shallow_copy()
            catalog.add(ds_name, self.create_default_data_set(ds_name))

        return super().run(pipeline, catalog, hook_manager, session_id)

    def create_default_data_set(self, ds_name: str) -> AbstractDataSet:
        # TODO: handle credentials better (probably with built-in Kedro credentials
        #  via ConfigLoader (but it's not available here...)
        dataset_cls = CloudpickleDataset
        if is_distributed_environment():
            logger.info("Using distributed dataset class as a default")
            dataset_cls = DistributedCloudpickleDataset

        return dataset_cls(
            self.runner_config.bucket,
            ds_name,
            self.runner_config.run_id,
        )
