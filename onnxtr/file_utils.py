# Copyright (C) 2021-2024, Mindee | Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import importlib.metadata
import importlib.util
import logging
from typing import Optional

__all__ = ["requires_package"]

ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({"AUTO"})


def requires_package(name: str, extra_message: Optional[str] = None) -> None:  # pragma: no cover
    """
    package requirement helper

    Args:
    ----
        name: name of the package
        extra_message: additional message to display if the package is not found
    """
    try:
        _pkg_version = importlib.metadata.version(name)
        logging.info(f"{name} version {_pkg_version} available.")
    except importlib.metadata.PackageNotFoundError:
        raise ImportError(
            f"\n\n{extra_message if extra_message is not None else ''} "
            f"\nPlease install it with the following command: pip install {name}\n"
        )
