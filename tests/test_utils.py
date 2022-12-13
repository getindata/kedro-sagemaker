import json

import pytest

from kedro_sagemaker.constants import (
    KEDRO_SAGEMAKER_PARAM_KEY_PREFIX,
    KEDRO_SAGEMAKER_PARAM_VALUE_PREFIX,
)
from kedro_sagemaker.utils import parse_flat_parameters


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
