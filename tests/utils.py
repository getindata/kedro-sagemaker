from unittest.mock import MagicMock


def identity(x):
    return x


def assert_has_any_call_with_args(magic: MagicMock, args_list: list):
    for call in list(magic.mock_calls):  # list() copies on first call
        if call.args[0] == args_list:
            return True
    else:
        raise AssertionError
