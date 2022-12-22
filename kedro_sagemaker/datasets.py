import logging
import tarfile
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from sys import version_info
from typing import Any, Dict, Union

import backoff
import cloudpickle
import fsspec
import zstandard as zstd
from kedro.io import AbstractDataSet, DataSetError

from kedro_sagemaker.constants import KEDRO_SAGEMAKER_S3_TEMP_DIR_NAME
from kedro_sagemaker.utils import is_distributed_master_node

logger = logging.getLogger(__name__)


class SageMakerModelDataset(AbstractDataSet):
    STORE_METHODS = ("pickle", "blob", "cloudpickle")
    SAGEMAKER_DEFAULT_PATH = "/opt/ml"
    MODEL_TAR_GZ = "model.tar.gz"

    def __init__(
        self,
        store_method: str = "pickle",
        sagemaker_path: str = "/opt/ml/model",
        load_kwargs: dict = None,
        save_kwargs: dict = None,
        model_registry_kwargs: dict = None,
    ):
        super().__init__()
        assert (
            store_method in self.STORE_METHODS
        ), f"Unsupported store method {store_method}, possible: {','.join(self.STORE_METHODS)}"
        assert load_kwargs is None or isinstance(load_kwargs, dict)
        assert save_kwargs is None or isinstance(save_kwargs, dict)
        self.store_method = store_method
        self.load_kwargs = load_kwargs or {}
        self.save_kwargs = save_kwargs or {}
        self.model_registry_kwargs = model_registry_kwargs or {}
        self.sagemaker_path = sagemaker_path

    def _load(self) -> Union[Any, dict]:
        load_kwargs = deepcopy(self.load_kwargs)
        search_path = load_kwargs.pop("path", None) or self.SAGEMAKER_DEFAULT_PATH
        for p in Path(search_path).rglob(self.MODEL_TAR_GZ):
            if p.is_file():
                with tarfile.open(p, "r:gz") as tf:
                    tf.extractall(self.sagemaker_path)
                    break
        else:
            raise DataSetError(f"No model.tar.gz found in {search_path}")

        to_load = {}
        for p in Path(self.sagemaker_path).iterdir():
            if (
                p.is_file()
                and not p.is_symlink()
                and not p.name.startswith(".")
                and p.name != self.MODEL_TAR_GZ
            ):
                logger.info(f"File to load: {p.absolute()}")
                to_load[p.name] = p

        def _pickle(fo):
            import pickle

            return pickle.load(fo, **(load_kwargs or {}))

        def _cloudpickle(fo):
            import cloudpickle

            return cloudpickle.load(fo, **(load_kwargs or {}))

        methods = {
            "blob": lambda fo: fo.read(),
            "pickle": _pickle,
            "cloudpickle": _cloudpickle,
        }

        results = {}
        for name, path in to_load.items():
            with path.open("rb") as f:
                results[name] = methods[self.store_method](f)

        if len(results) == 1:
            return next(iter(results.values()))
        else:
            return results

    def _save(self, data: Union[Any, dict]) -> None:
        save_kwargs = deepcopy(self.save_kwargs)
        to_save = (
            data
            if isinstance(data, dict)
            else {save_kwargs.pop("model_file_name", "model.pkl"): data}
        )

        def _pickle(f, obj):
            import pickle

            return pickle.dump(obj, f, **(save_kwargs or {}))

        def _cloudpickle(f, obj):
            import cloudpickle

            return cloudpickle.dump(obj, f, **(save_kwargs or {}))

        methods = {
            "blob": lambda f, obj: f.write(obj),
            "pickle": _pickle,
            "cloudpickle": _cloudpickle,
        }

        for name, data in to_save.items():
            with (Path(self.sagemaker_path) / name).open("wb") as f:
                methods[self.store_method](f, data)

    def _describe(self) -> Dict[str, Any]:
        return {}


class CloudpickleDataset(AbstractDataSet):
    def __init__(
        self,
        bucket,
        dataset_name,
        run_id,
    ):
        self.bucket = bucket
        self.run_id = run_id
        self.dataset_name = dataset_name
        self.pickle_protocol = None if version_info[:2] > (3, 8) else 4

    @lru_cache()
    def _get_target_path(self):
        # This fails with:
        # 2022-11-29T13:19:22.871+01:00	DataSetError: Failed while saving data to data set
        #
        # 2022-11-29T13:19:22.871+01:00	CloudpickleDataset(dataset_name=data_processing.preprocessed_shuttles, info=for
        #
        # 2022-11-29T13:19:22.871+01:00	use only within SageMaker Pipelines,
        #
        # 2022-11-29T13:19:22.871+01:00	path=/opt/ml/processing/data_processing.preprocessed_shuttles/data.bin).
        #
        # 2022-11-29T13:19:22.871+01:00	[Errno 13] Permission denied:
        #
        # 2022-11-29T13:19:22.871+01:00	'/opt/ml/processing/data_processing.preprocessed_shuttles/data.bin'
        # p = Path(
        #     f"/opt/ml/processing/{self.dataset_name}"  # TODO - move to config
        # )
        # p.mkdir(parents=True, exist_ok=True)
        # return str(p / "data.bin")
        return f"s3://{self.bucket}/{KEDRO_SAGEMAKER_S3_TEMP_DIR_NAME}/{self.run_id}/{self.dataset_name}.bin"

    @lru_cache()
    def _get_storage_options(self):
        return {}

    def _load(self):
        with fsspec.open(
            self._get_target_path(), "rb", **self._get_storage_options()
        ) as f:
            with zstd.open(f, "rb") as stream:
                return cloudpickle.load(stream)

    def _save(self, data: Any) -> None:
        with fsspec.open(
            self._get_target_path(), "wb", **self._get_storage_options()
        ) as f:
            with zstd.open(f, "wb") as stream:
                cloudpickle.dump(data, stream, protocol=self.pickle_protocol)

    def _describe(self) -> Dict[str, Any]:
        return {
            "info": "for use only within SageMaker Pipelines",
            "dataset_name": self.dataset_name,
            "path": self._get_target_path(),
        }


class DistributedCloudpickleDataset(CloudpickleDataset):
    @backoff.on_exception(
        backoff.fibo,
        Exception,
        max_time=lambda: 300,
        raise_on_giveup=False,
    )
    def _load(self):
        return super()._load()

    def _save(self, data: Any) -> None:
        if is_distributed_master_node():
            super()._save(data)
        else:
            logger.warning(
                f"DataSet {self.dataset_name} will not be saved on a distributed node"
            )
