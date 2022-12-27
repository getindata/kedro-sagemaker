import logging
from typing import Callable, Optional

from sagemaker.workflow.pipeline import Pipeline as SageMakerPipeline

logger = logging.getLogger(__name__)


class SageMakerClient:
    def __init__(self, sagemaker_pipeline: SageMakerPipeline, execution_role: str):
        self.execution_role = execution_role
        self.sagemaker_pipeline = sagemaker_pipeline

    def run(
        self,
        is_local: bool,
        wait_for_completion: bool = False,
        on_pipeline_started: Optional[Callable[[SageMakerPipeline], None]] = None,
    ):
        self.sagemaker_pipeline.upsert(self.execution_role)
        smp = self.sagemaker_pipeline.start()
        if on_pipeline_started:
            on_pipeline_started(smp)

        if not is_local and wait_for_completion:
            try:
                smp.wait()
                return True
            except Exception:
                logger.exception("Error while running the pipeline", exc_info=True)
                return False
        else:
            return True

    def update(self):
        self.sagemaker_pipeline.upsert(self.execution_role)
