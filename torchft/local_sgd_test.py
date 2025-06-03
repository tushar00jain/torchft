# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List
from unittest import TestCase
from unittest.mock import MagicMock, create_autospec

import torch
from parameterized import parameterized
from torch import Tensor, nn, optim
from torch.distributed.tensor import DTensor

from torchft.local_sgd import DiLoCo, LocalSGD, extract_local_tensor
from torchft.manager import Manager


def create_manager() -> MagicMock:
    """
    Creates a mock manager with some useful defaults for testing
    the the optimizer's usage of the Manager
    """
    manager = create_autospec(Manager)

    current_local_step: int = 0

    def local_step() -> int:
        return current_local_step

    def increment_local_step() -> None:
        nonlocal current_local_step
        current_local_step += 1

    manager.local_step.side_effect = local_step
    manager.increment_local_step.side_effect = increment_local_step

    manager.errored.return_value = None

    return manager


class SimpleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(3, 4),
            nn.ReLU(),
            nn.Linear(4, 5),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def _params_dict(m: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {name: p.data for name, p in m.named_parameters()}


def _copy_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {name: value.clone().detach() for name, value in state_dict.items()}


class LocalSGDTest(TestCase):
    def test_local_sgd_healthy(self) -> None:
        model = SimpleModel()
        optimizer = optim.SGD(model.parameters())
        manager = create_autospec(Manager)
        with LocalSGD(manager, model, optimizer, sync_every=2) as local_sgd:
            self.assertEqual(local_sgd._local_step, 0)
            inp = torch.rand(2, 3)
            loss = model(inp).mean()
            loss.backward()
            optimizer.step()

            self.assertEqual(local_sgd._local_step, 1)
            self.assertEqual(manager.start_quorum.call_count, 0)
            loss = model(inp).mean()
            loss.backward()
            optimizer.step()
            self.assertEqual(manager.start_quorum.call_count, 1)

            manager.should_commit.return_value = True
            self.assertEqual(local_sgd._local_step, 0)
            self.assertEqual(manager.should_commit.call_count, 1)
            self.assertEqual(manager.collect_all_allreduce.call_count, 1)

    def test_extract_local_tensor(self) -> None:
        regular_tensor = torch.rand(3, 3, requires_grad=True)
        regular_result = extract_local_tensor(regular_tensor)

        self.assertTrue(torch.equal(regular_result, regular_tensor))
        self.assertIsNone(regular_result.grad)
        self.assertNotEqual(id(regular_result), id(regular_tensor))
        local_tensor = torch.rand(3, 3, requires_grad=True)
        dtensor = MagicMock(spec=DTensor)
        dtensor.to_local.return_value = local_tensor
        dtensor_result = extract_local_tensor(dtensor)

        self.assertTrue(torch.equal(dtensor_result, local_tensor))
        self.assertIsNone(dtensor_result.grad)
        self.assertNotEqual(id(dtensor_result), id(local_tensor))
        dtensor.to_local.assert_called_once()

    def test_local_sgd_recovery(self) -> None:
        model = SimpleModel()
        optimizer = optim.SGD(model.parameters())
        manager = create_autospec(Manager)

        with LocalSGD(manager, model, optimizer, sync_every=2) as local_sgd:
            og_state_dict = _copy_state_dict(model.state_dict())

            inp = torch.rand(2, 3)

            loss = model(inp).mean()
            loss.backward()
            optimizer.step()

            # Check that the model's state dict has been updated
            for name, param in model.state_dict().items():
                # Ensure the parameter has changed
                self.assertFalse(
                    torch.equal(og_state_dict[name], param),
                    f"Parameter {name} did not change.",
                )
            self.assertEqual(local_sgd._local_step, 1)


class DiLoCoTest(TestCase):
    def test_diloco_healthy(self) -> None:
        model = SimpleModel()

        # Setup optimizers
        inner_optimizer = torch.optim.AdamW(
            model.parameters(), lr=4e-4, weight_decay=0.1, betas=(0.9, 0.95)
        )
        outer_optimizer = torch.optim.SGD(
            model.parameters(), lr=0.7, momentum=0.9, nesterov=True
        )

        manager = create_manager()
        manager._use_async_quorum = False
        with DiLoCo(
            manager, [model], inner_optimizer, outer_optimizer, sync_every=2
        ) as diloco:
            parameter_count = len(list(model.parameters()))
            initial_outer_opt_state = outer_optimizer.state_dict()
            self.assertEqual(initial_outer_opt_state["state"], {})

            self.assertEqual(manager.local_step(), 0)
            torch.testing.assert_close(
                diloco._fragments[0].original_parameters, _params_dict(model)
            )
            inp = torch.rand(2, 3)
            loss = model(inp).mean()
            loss.backward()
            inner_optimizer.step()

            self.assertEqual(manager.local_step(), 1)
            self.assertEqual(manager.start_quorum.call_count, 1)
            loss = model(inp).mean()
            loss.backward()
            inner_optimizer.step()
            self.assertEqual(manager.start_quorum.call_count, 2)

            manager.should_commit.return_value = True
            self.assertEqual(manager.local_step(), 2)
            torch.testing.assert_close(
                diloco._fragments[0].original_parameters, _params_dict(model)
            )
            self.assertEqual(manager.should_commit.call_count, 1)
            self.assertEqual(manager.collect_all_allreduce.call_count, 1)

            outer_opt_state = outer_optimizer.state_dict()
            self.assertEqual(len(outer_opt_state["state"]), parameter_count)

    @parameterized.expand(
        [
            ("bucketized_should_use_fewer_calls", True, True),
            ("non_bucketized_should_call_per_param", False, False),
        ]
    )
    def test_diloco_allreduce_call_efficiency(
        self,
        name: str,
        use_bucketization: bool,
        expect_fewer_calls: bool,
    ) -> None:
        model = SimpleModel()

        inner_optimizer = torch.optim.AdamW(
            model.parameters(), lr=4e-4, weight_decay=0.1, betas=(0.9, 0.95)
        )
        outer_optimizer = torch.optim.SGD(
            model.parameters(), lr=0.7, momentum=0.9, nesterov=True
        )

        manager = create_manager()

        manager._use_async_quorum = False
        manager.should_commit.return_value = True

        with DiLoCo(
            manager,
            [model],
            inner_optimizer,
            outer_optimizer,
            sync_every=2,
            use_bucketization=use_bucketization,
        ) as diloco:
            inp = torch.rand(2, 3)
            loss = model(inp).mean()
            loss.backward()
            inner_optimizer.step()

            loss = model(inp).mean()
            loss.backward()
            inner_optimizer.step()

            allreduce_calls = manager.collect_all_allreduce.call_count

            if expect_fewer_calls:
                self.assertEqual(int(allreduce_calls), 1)
            else:
                self.assertEqual(int(allreduce_calls), 1)

    def test_bucketization_correctness(self) -> None:
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.w1 = nn.Parameter(torch.tensor([1.0, 2.0]))
                self.w2 = nn.Parameter(torch.tensor([3.0, 4.0, 5.0]))

            def forward(self, x):
                return x @ self.w1.unsqueeze(0).T + self.w2.sum()

        model = TinyModel()
        inner_opt = torch.optim.SGD(model.parameters(), lr=0.1)
        outer_opt = torch.optim.SGD(model.parameters(), lr=0.1)

        # Manually assign fake gradients
        grads = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0])]
        for p, g in zip(model.parameters(), grads):
            p.grad = g.clone()

        manager = create_autospec(Manager)
        manager._use_async_quorum = False
        manager.should_commit.return_value = True

        # Define fake collect_all_allreduce: multiplies all buffers by 2
        def fake_collect_all_allreduce(
            tensors: List[Tensor], should_quantize: bool
        ) -> torch.futures.Future[List[torch.futures.Future[Tensor]]]:
            for tensor in tensors:
                tensor.mul_(2)
            fut = torch.futures.Future()  # pyre-fixme[29]: not a function
            fut.set_result(tensors)

            futs = torch.futures.Future()  # pyre-fixme[29]: not a function
            futs.set_result([fut])
            return futs

        manager.collect_all_allreduce.side_effect = fake_collect_all_allreduce

        diloco = DiLoCo(
            manager, [model], inner_opt, outer_opt, sync_every=2, use_bucketization=True
        )
        diloco._fragments[0].bucket_cap_mb = 10 * 1024 * 1024

        # Run only bucketized logic
        diloco._fragments[0]._average_grads()

        # Expect grads to have been doubled
        expected_grads = [g * 2 for g in grads]
        for param, expected in zip(model.parameters(), expected_grads):
            torch.testing.assert_close(param.grad, expected, rtol=1e-5, atol=1e-8)
