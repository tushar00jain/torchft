# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import types
import unittest
from datetime import timedelta
from typing import Callable, List, Optional, TypeVar, cast

# Define a type variable for the Future's value type
T = TypeVar("T")

import parameterized
import torch
from torch.distributed.distributed_c10d import Work
from torch.futures import Future

from torchft.manager import Manager, _ManagedWork


class SimpleWork(Work):
    """A simple implementation of torch.distributed.Work for testing."""

    def __init__(self, tensors: List[torch.Tensor]) -> None:
        super().__init__()
        self._tensors = tensors
        self._future: Future[List[torch.Tensor]] = torch.futures.Future()
        self._is_completed: bool = False

    def wait(self, timeout: Optional[timedelta] = None) -> bool:
        self._is_completed = True
        self._future.set_result(self._tensors)
        return True

    def get_future(self) -> Future[List[torch.Tensor]]:
        return self._future


class TestManagedWork(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("cpu", torch.device("cpu")),
            ("cuda", torch.device("cuda:0")),
        ]
    )
    def test_callbacks_execute_after_wait(
        self, name: str, device: torch.device
    ) -> None:
        """Test that callbacks are only executed after wait() is called."""
        # Skip if CUDA is requested but not available
        if device.type == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        # Create a tensor to work with
        tensor = torch.ones(1, dtype=torch.float32, device=device)

        # Create a simple work object
        work = SimpleWork([tensor])

        # Create a minimal manager object with just the wrap_future method
        manager = Manager.__new__(Manager)  # Create instance without calling __init__
        # We're using types.MethodType to attach a method to the manager instance
        # This is just for testing purposes
        manager.wrap_future = types.MethodType(  # type: ignore
            lambda self, fut, default, timeout=None: fut, manager
        )

        # Create the managed work
        managed_work = _ManagedWork(manager, work, [tensor])

        # Track callback execution
        callback_executed: bool = False

        def callback(fut: Future[object]) -> List[torch.Tensor]:
            # Cast to the expected type
            tensor_fut = cast(Future[List[torch.Tensor]], fut)
            nonlocal callback_executed
            callback_executed = True
            # Multiply tensor by 2 to verify the callback ran
            value = tensor_fut.value()
            value[0].mul_(2)
            return value

        # Add the callback
        fut = managed_work.get_future()
        fut = fut.then(callback)

        # Verify callback hasn't executed yet
        self.assertFalse(callback_executed)
        self.assertEqual(tensor.item(), 1.0)

        # Call wait() which should trigger the callback
        managed_work.wait()

        # Verify callback has executed
        self.assertTrue(callback_executed)
        self.assertEqual(tensor.item(), 2.0)

    @parameterized.parameterized.expand(
        [
            ("cpu", torch.device("cpu")),
            ("cuda", torch.device("cuda:0")),
        ]
    )
    def test_multiple_callbacks_execute_in_order(
        self, name: str, device: torch.device
    ) -> None:
        """Test that multiple callbacks are executed in the order they were added."""
        # Skip if CUDA is requested but not available
        if device.type == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        # Create a tensor to work with
        tensor = torch.ones(1, dtype=torch.float32, device=device)

        # Create a simple work object
        work = SimpleWork([tensor])

        # Create a minimal manager object with just the wrap_future method
        manager = Manager.__new__(Manager)  # Create instance without calling __init__
        manager.wrap_future = types.MethodType(  # type: ignore
            lambda self, fut, default, timeout=None: fut, manager
        )

        # Create the managed work
        managed_work = _ManagedWork(manager, work, [tensor])

        # Track execution order
        execution_order: List[int] = []

        def callback1(fut: Future[T]) -> List[torch.Tensor]:
            # Cast to the expected type
            tensor_fut = cast(Future[List[torch.Tensor]], fut)
            execution_order.append(1)
            value = tensor_fut.value()
            value[0].add_(1)
            return value

        def callback2(fut: Future[T]) -> List[torch.Tensor]:
            # Cast to the expected type
            tensor_fut = cast(Future[List[torch.Tensor]], fut)
            execution_order.append(2)
            value = tensor_fut.value()
            value[0].add_(2)
            return value

        def callback3(fut: Future[T]) -> List[torch.Tensor]:
            # Cast to the expected type
            tensor_fut = cast(Future[List[torch.Tensor]], fut)
            execution_order.append(3)
            value = tensor_fut.value()
            value[0].add_(3)
            return value

        # Add callbacks
        fut = managed_work.get_future()
        fut = fut.then(callback1)
        fut = fut.then(callback2)
        fut = fut.then(callback3)

        # Verify no callbacks have executed yet
        self.assertEqual(len(execution_order), 0)
        self.assertEqual(tensor.item(), 1.0)

        # Call wait() which should trigger the callbacks
        managed_work.wait()

        # Verify callbacks executed in order
        self.assertEqual(execution_order, [1, 2, 3])

        # Each callback adds to the tensor, so final value should be 1 + 1 + 2 + 3 = 7
        self.assertEqual(tensor.item(), 7.0)

    @parameterized.parameterized.expand(
        [
            ("cpu", torch.device("cpu")),
            ("cuda", torch.device("cuda:0")),
        ]
    )
    def test_future_then_api(self, name: str, device: torch.device) -> None:
        """Test that the future's then API works correctly with ManagedWork."""
        # Skip if CUDA is requested but not available
        if device.type == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        # Create a tensor to work with
        tensor = torch.ones(1, dtype=torch.float32, device=device)

        # Create a simple work object
        work = SimpleWork([tensor])

        # Create a minimal manager object with just the wrap_future method
        manager = Manager.__new__(Manager)  # Create instance without calling __init__
        manager.wrap_future = types.MethodType(  # type: ignore
            lambda self, fut, default, timeout=None: fut, manager
        )

        # Create the managed work
        managed_work = _ManagedWork(manager, work, [tensor])

        # Get the future
        future = managed_work.get_future()

        # Track callback execution
        callback_executed: bool = False

        def callback(fut: Future[object]) -> List[torch.Tensor]:
            # Cast to the expected type
            tensor_fut = cast(Future[List[torch.Tensor]], fut)
            nonlocal callback_executed
            callback_executed = True
            # Multiply tensor by 3 to verify the callback ran
            value = tensor_fut.value()
            value[0].mul_(3)
            return value

        # Use the then API
        future = future.then(callback)

        # Verify callback hasn't executed yet
        self.assertFalse(callback_executed)
        self.assertEqual(tensor.item(), 1.0)

        # Call wait() on the managed_work first to set up the future properly
        managed_work.wait()

        # Now wait on the future
        future.wait()

        # Verify callback has executed
        self.assertTrue(callback_executed)
        self.assertEqual(tensor.item(), 3.0)


if __name__ == "__main__":
    unittest.main()
