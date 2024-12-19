"""This module defines the ClusteringPartitioner pass."""
from __future__ import annotations

import logging

import numpy as np

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.utils.typing import is_integer

_logger = logging.getLogger(__name__)


class MergePartitions(BasePass):
    """
    The merge partitions Pass. Merges 2 adjacent partitions together.
    """

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        pass
        
