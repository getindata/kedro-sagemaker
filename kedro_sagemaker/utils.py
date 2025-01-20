import json
import logging
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from typing import Any, List, MutableMapping, Optional, Union

import kedro
from kedro.framework.session import KedroSession
from packaging import version

from kedro_sagemaker.config import KedroSageMakerPluginConfig
from kedro_sagemaker.constants import (
    KEDRO_SAGEMAKER_PARAM_KEY_PREFIX,
    KEDRO_SAGEMAKER_PARAM_VALUE_PREFIX,
)

logger = logging.getLogger()


@dataclass
class CliContext:
    env: str
    metadata: Any


def is_distributed_master_node() -> bool:
    is_rank_0 = True
    try:
        if "TF_CONFIG" in os.environ:
            # TensorFlow
            tf_config = json.loads(os.environ["TF_CONFIG"])
            worker_type = tf_config["task"]["type"].lower()
            is_rank_0 = (worker_type == "chief" or worker_type == "master") or (
                worker_type == "worker" and tf_config["task"]["index"] == 0
            )
        else:
            # MPI + PyTorch
            for e in ("OMPI_COMM_WORLD_RANK", "RANK"):
                if e in os.environ:
                    is_rank_0 = int(os.environ[e]) == 0
                    break
    except:  # noqa
        logger.error(
            "Could not parse environment variables related to distributed computing. "
            "Set appropriate values for one of: RANK, OMPI_COMM_WORLD_RANK or TF_CONFIG",
            exc_info=True,
        )
        logger.warning("Assuming this node is not a master node, due to error.")
        return False
    return is_rank_0


def is_distributed_environment() -> bool:
    return any(e in os.environ for e in ("OMPI_COMM_WORLD_RANK", "RANK", "TF_CONFIG"))


class KedroContextManager:
    def __init__(
        self, package_name: str, env: str, extra_params: Optional[dict] = None
    ):
        self.extra_params = extra_params
        self.env = env
        self.package_name = package_name
        self.session: Optional[KedroSession] = None

    @cached_property
    def plugin_config(self) -> KedroSageMakerPluginConfig:
        # from this version onwards (not inclusive) config_loader uses OmegaConfigLoader which requires a different syntax
        required_version = version.parse("0.18.4")
        current_version = version.parse(kedro.__version__)

        if current_version > required_version:
            return KedroSageMakerPluginConfig.parse_obj(
                self.context.config_loader["parameters"]
            )
        else:
            return KedroSageMakerPluginConfig.parse_obj(
                self.context.config_loader.get("parameters*")
            )

    @cached_property
    def context(self):
        assert self.session is not None, "Session not initialized yet"
        return self.session.load_context()

    def __enter__(self):
        self.session = KedroSession.create(
            self.package_name, env=self.env, extra_params=self.extra_params
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.__exit__(exc_type, exc_val, exc_tb)


def flatten_dict(d: MutableMapping, parent_key: str = "", sep: str = ".") -> dict:
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def parse_flat_parameters(
    param_source: Union[dict, MutableMapping],
    key_prefix: str = KEDRO_SAGEMAKER_PARAM_KEY_PREFIX,
    value_prefix: str = KEDRO_SAGEMAKER_PARAM_VALUE_PREFIX,
) -> dict:
    """
    This function takes `param_source` (most likely os.environ)
    searches for the keys starting with `key_prefix`/`value_prefix`
    and parses them from the form:
    "key_prefix_0": "name.of.the.param"
    "value_prefix_0": "666"
    into dictionary: {"name": {"of": {"the": {"param": 666}}}}

    :param param_source:
    :param key_prefix:
    :param value_prefix:
    :return:
    """

    def sorted_keys_with_prefix(prefix) -> List[str]:
        return sorted(
            (x for x in param_source.keys() if x.startswith(prefix)),
            key=lambda x: int(x.replace(prefix, "")),
        )

    def nested_defaultdict():
        return defaultdict(nested_defaultdict)

    params = nested_defaultdict()

    keys = sorted_keys_with_prefix(key_prefix)
    values = sorted_keys_with_prefix(value_prefix)
    assert len(keys) == len(values), "Invalid parameter/key pairs"

    for k, v in zip(keys, values):
        p = params
        key = param_source[k]
        value = param_source[v]
        while "." in key:
            level, reminder = key.split(".", maxsplit=1)
            p = p[level]
            key = reminder
        else:
            try:
                p[key] = json.loads(value)
            except json.JSONDecodeError:
                p[key] = value
    return params


def docker_build(path: str, image: str, platforms: str) -> int:
    rv = subprocess.run(
        [
            "docker",
            "buildx",
            "build",
            "--platform",
            platforms,
            path,
            "-t",
            image,
            "--push"
        ],
        stdout=sys.stdout,
        stderr=subprocess.STDOUT,
    ).returncode
    if rv:
        logger.error("Docker build has failed.")
    return rv


def docker_push(image: str) -> int:
    rv = subprocess.run(
        ["docker", "push", image], stdout=sys.stdout, stderr=subprocess.STDOUT
    ).returncode
    if rv:
        logger.error("Docker push has failed.")
    return rv


def is_mlflow_enabled() -> bool:
    try:
        import kedro_mlflow  # NOQA
        import mlflow  # NOQA

        return True
    except ImportError:
        return False
