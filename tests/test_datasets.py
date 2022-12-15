import os
import pickle
import tarfile
from io import StringIO
from pathlib import Path
from typing import Type
from unittest.mock import patch
from uuid import uuid4

import cloudpickle
import numpy as np
import pandas as pd
import pytest
from kedro.io import DataSetError

from kedro_sagemaker.constants import KEDRO_SAGEMAKER_S3_TEMP_DIR_NAME
from kedro_sagemaker.datasets import (
    SageMakerModelDataset,
    CloudpickleDataset,
    DistributedCloudpickleDataset,
)


@pytest.mark.parametrize("store_method", SageMakerModelDataset.STORE_METHODS)
def test_can_load_from_sagemaker_model_dataset(store_method, tmp_path):
    data_to_save = pd.DataFrame(
        {"data": [uuid4().hex for _ in range(100)]}
    )  # contents are not important

    with tarfile.open(tmp_path / "model.tar.gz", "w:gz") as tf:
        file = Path(tmp_path / "data.bin")
        {
            "blob": lambda: file.write_bytes(
                data_to_save.to_csv(index=False).encode("UTF-8")
            ),
            "pickle": lambda: file.write_bytes(pickle.dumps(data_to_save)),
            "cloudpickle": lambda: file.write_bytes(cloudpickle.dumps(data_to_save)),
        }[store_method]()
        tf.add(file)

    ds = SageMakerModelDataset(
        store_method, sagemaker_path=tmp_path, load_kwargs={"path": tmp_path}
    )
    loaded_data = ds.load()

    if store_method == "blob":
        assert isinstance(loaded_data, bytes)
        loaded_data = pd.read_csv(StringIO(loaded_data.decode("UTF-8")))

    assert data_to_save.equals(
        loaded_data
    ), "Data after loading is not the same as saved data"


@pytest.mark.xfail(raises=DataSetError)
def test_sagemaker_dataset_no_data_to_load(tmp_path):
    ds = SageMakerModelDataset(sagemaker_path=str(tmp_path))
    ds.load()


@pytest.mark.parametrize("store_method", SageMakerModelDataset.STORE_METHODS)
@pytest.mark.parametrize("save_kwargs", [{}, None, {"model_file_name": "file.bin"}])
def test_can_save_sagemaker_model_dataset(store_method, save_kwargs, tmp_path):
    data_to_save = (
        uuid4().hex.encode()
    )  # because bytes can be pickled, cloudpickled and saved
    ds = SageMakerModelDataset(
        store_method, sagemaker_path=str(tmp_path), save_kwargs=save_kwargs
    )

    ds.save(data_to_save)

    file_name_to_load = (save_kwargs or {}).get("model_file_name", "model.pkl")
    file_path = tmp_path / file_name_to_load
    assert file_path.exists() and file_path.is_file()

    with file_path.open("rb") as f:
        saved_data = {
            "blob": lambda: f.read(),
            "pickle": lambda: pickle.load(f),
            "cloudpickle": lambda: cloudpickle.load(f),
        }[store_method]()
        assert data_to_save == saved_data


@pytest.mark.parametrize(
    "dataset_class", (CloudpickleDataset, DistributedCloudpickleDataset)
)
def test_azure_dataset_config(dataset_class: Type):
    run_id = uuid4().hex
    bucket = f"bucket_{uuid4().hex}"
    ds = dataset_class(bucket, "unit_tests_dataset", run_id)
    target_path = ds._get_target_path()
    cfg = ds._get_storage_options()
    assert (
        target_path.startswith("s3://")
        and target_path.endswith(".bin")
        and all(
            part in target_path
            for part in (
                bucket,
                "unit_tests_dataset",
                KEDRO_SAGEMAKER_S3_TEMP_DIR_NAME,
                run_id,
            )
        )
    ), "Invalid target path"

    assert isinstance(cfg, dict)


@pytest.mark.parametrize(
    "obj,comparer",
    [
        (
            pd.DataFrame(np.random.rand(1000, 3), columns=["a", "b", "c"]),
            lambda a, b: a.equals(b),
        ),
        (np.random.rand(100, 100), lambda a, b: np.equal(a, b).all()),
        (["just", "a", "list"], lambda a, b: all(a[i] == b[i] for i in range(len(a)))),
        ({"some": "dictionary"}, lambda a, b: all(a[k] == b[k] for k in a.keys())),
        (set(["python", "set"]), lambda a, b: len(a - b) == 0),
        ("this is a string", lambda a, b: a == b),
        (1235, lambda a, b: a == b),
        ((1234, 5678), lambda a, b: all(a[i] == b[i] for i in range(len(a)))),
    ],
)
@pytest.mark.parametrize(
    "is_distributed", (False, pytest.param(True, marks=pytest.mark.xfail))
)
def test_can_save_python_objects_using_fspec(
    obj, comparer, patched_cloudpickle_dataset, is_distributed
):
    with patch.dict(os.environ, {"RANK": "1" if is_distributed else "0"}, clear=False):
        ds = patched_cloudpickle_dataset
        ds.save(obj)
        assert (
            Path(ds._get_target_path()).stat().st_size > 0
        ), "File does not seem to be saved"
        assert comparer(
            obj, ds.load()
        ), "Objects are not the same after deserialization"
