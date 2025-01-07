import json
import os
from unittest.mock import Mock, patch

import pytest

from kedro_sagemaker.constants import (
    KEDRO_SAGEMAKER_PARAM_KEY_PREFIX,
    KEDRO_SAGEMAKER_PARAM_VALUE_PREFIX,
)
from kedro_sagemaker.utils import (
    docker_build,
    docker_push,
    is_distributed_master_node,
    parse_flat_parameters,
)


@pytest.mark.parametrize(
    "params",
    [
        (
            {
                KEDRO_SAGEMAKER_PARAM_VALUE_PREFIX + "0": "xD",
                KEDRO_SAGEMAKER_PARAM_KEY_PREFIX + "0": "my.param.asd",
                KEDRO_SAGEMAKER_PARAM_VALUE_PREFIX + "1": "3.0",
                KEDRO_SAGEMAKER_PARAM_KEY_PREFIX + "1": "top",
            },
            {"my": {"param": {"asd": "xD"}}, "top": 3.0},
        ),
        (
            {
                KEDRO_SAGEMAKER_PARAM_KEY_PREFIX + "0": "root_key",
                KEDRO_SAGEMAKER_PARAM_VALUE_PREFIX
                + "0": json.dumps([1.0, 2.0, 3.0, "xyz"]),
            },
            {"root_key": [1.0, 2.0, 3.0, "xyz"]},
        ),
        ({}, {}),
        (
            {
                KEDRO_SAGEMAKER_PARAM_VALUE_PREFIX + "0": "xD",
                KEDRO_SAGEMAKER_PARAM_KEY_PREFIX + "0": "my.param.first",
                KEDRO_SAGEMAKER_PARAM_VALUE_PREFIX + "1": "3.0",
                KEDRO_SAGEMAKER_PARAM_KEY_PREFIX + "1": "my.param.second",
            },
            {"my": {"param": {"first": "xD", "second": 3.0}}},
        ),
        ({"totally not used key": "not used"}, {}),
        (
            {
                KEDRO_SAGEMAKER_PARAM_KEY_PREFIX + "0": "a.b.c.d.e.f.g.h.i.j.k.l.m",
                KEDRO_SAGEMAKER_PARAM_VALUE_PREFIX + "0": "1234",
            },
            {
                "a": {
                    "b": {
                        "c": {
                            "d": {
                                "e": {
                                    "f": {
                                        "g": {
                                            "h": {"i": {"j": {"k": {"l": {"m": 1234}}}}}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
        ),
        ({KEDRO_SAGEMAKER_PARAM_KEY_PREFIX + "0": "k"}, AssertionError),
    ],
)
def test_can_parse_flat_kedro_params(params):
    inputs, expected = params

    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            parse_flat_parameters(inputs)
    else:
        v = parse_flat_parameters(inputs)
        assert v == expected
        assert isinstance(v, dict)
        assert (
            json.loads(json.dumps(v)) == expected
        ), "Cannot serialize/deserialize returned object"


@pytest.mark.parametrize(
    "environment,expected_master",
    [
        ({"TF_CONFIG": "ASD"}, False),
        ({"TF_CONFIG": json.dumps({"my_config": "not valid"})}, False),
        ({"RANK": "0"}, True),
        ({"RANK": "1"}, False),
        ({"RANK": "666"}, False),
        ({"OMPI_COMM_WORLD_RANK": "0"}, True),
        ({"OMPI_COMM_WORLD_RANK": "1"}, False),
        ({"TF_CONFIG": json.dumps({"task": {"type": "master"}})}, True),
        ({"TF_CONFIG": json.dumps({"task": {"type": "chief"}})}, True),
        ({"TF_CONFIG": json.dumps({"task": {"type": "worker"}})}, False),
        ({"TF_CONFIG": json.dumps({"task": {"type": "worker", "index": 1}})}, False),
        ({"TF_CONFIG": json.dumps({"task": {"type": "worker", "index": 0}})}, True),
        ({}, True),
    ],
)
def test_can_detect_distributed_master_node(environment, expected_master):
    with patch.dict(os.environ, environment, clear=True):
        assert (
            status := is_distributed_master_node()
        ) == expected_master, f"Invalid master node status detected, should be {expected_master} but was {status}"


@pytest.mark.parametrize("exit_code", range(10))
def test_docker_build(exit_code):
    with patch(
        "subprocess.run", return_value=Mock(returncode=exit_code)
    ) as subprocess_run:
        result = docker_build(".", "my_image:latest","platforms")
        assert exit_code == result, "Invalid exit code"
        subprocess_run.assert_called_once()


@pytest.mark.parametrize("exit_code", range(10))
def test_docker_push(exit_code):
    with patch(
        "subprocess.run", return_value=Mock(returncode=exit_code)
    ) as subprocess_run:
        result = docker_push("my_image:latest")
        assert exit_code == result, "Invalid exit code"
        subprocess_run.assert_called_once()
