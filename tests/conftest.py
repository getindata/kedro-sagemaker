from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from uuid import uuid4

import pytest

from kedro_sagemaker.datasets import CloudpickleDataset, DistributedCloudpickleDataset


@pytest.fixture(params=[CloudpickleDataset, DistributedCloudpickleDataset])
def patched_cloudpickle_dataset(request):
    with TemporaryDirectory() as tmp_dir:
        target_path = Path(tmp_dir) / (uuid4().hex + ".bin")
    with patch.object(
        CloudpickleDataset,
        "_get_target_path",
        return_value=str(target_path.absolute()),
    ):
        yield request.param("bucket", "unit_tests_ds", uuid4().hex)
