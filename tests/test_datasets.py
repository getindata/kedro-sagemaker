import pickle
import tarfile
from io import StringIO
from pathlib import Path
from uuid import uuid4

import cloudpickle
import pandas as pd
import pytest

from kedro_sagemaker.datasets import SageMakerModelDataset


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


@pytest.mark.parametrize("store_method", SageMakerModelDataset.STORE_METHODS)
@pytest.mark.parametrize("save_kwargs", [{}, None, {"model_file_name": "file.bin"}])
def test_can_save_sagemaker_model_dataset(store_method, save_kwargs, tmp_path):
    data_to_save = (
        uuid4().hex.encode()
    )  # because bytes can be pickled, cloudpickled and saved
    ds = SageMakerModelDataset(
        store_method, sagemaker_path=tmp_path, save_kwargs=save_kwargs
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
