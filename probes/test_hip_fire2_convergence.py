# SPDX-License-Identifier: Apache-2.0
"""CPU fixtures for the acceptance probe's physical and budget classification."""

import unittest
from types import SimpleNamespace

import torch

from hip_fire2_convergence import CountUpdates, FMAX, Limits, STRESS, joint_convergence, physically_converged
from nvalchemi.data import AtomicData, Batch


def batch(dtype=torch.float64):
    systems = []
    for source in (0, 2):
        data = AtomicData(positions=torch.zeros(2, 3, dtype=dtype), atomic_numbers=torch.ones(2, dtype=torch.long),
                          forces=torch.zeros(2, 3, dtype=dtype), energy=torch.zeros(1, 1, dtype=dtype),
                          cell=torch.eye(3, dtype=dtype)[None], stress=torch.zeros(1, 3, 3, dtype=dtype))
        data.add_system_property('status', torch.zeros(1, 1, dtype=torch.long))
        data.add_system_property('source_index', torch.tensor([[source]], dtype=torch.long))
        systems.append(data)
    return Batch.from_data_list(systems)


class ClassificationTests(unittest.TestCase):
    def test_absolute_stress_and_force_both_required(self):
        for dtype in (torch.float64, torch.float32):
            data = batch(dtype)
            hook = joint_convergence()
            data.stress[0, 0, 0] = -2 * STRESS
            data.forces[2, 0] = 2 * FMAX
            self.assertIsNone(hook.evaluate(data))
            data.stress[0, 0, 0] = -STRESS / 2
            self.assertEqual(hook.evaluate(data).tolist(), [0])
            hook(SimpleNamespace(batch=data), hook.stage)
            self.assertEqual(data.status.flatten().tolist(), [1, 0])

    def test_terminal_physical_metrics_use_joint_threshold(self):
        values = {'energy': torch.tensor([1.0]), 'forces': torch.zeros(2, 3), 'stress': torch.zeros(1, 3, 3)}
        self.assertTrue(physically_converged(values))
        values['stress'][0, 0, 0] = -2 * STRESS
        self.assertFalse(physically_converged(values))
        values['stress'].zero_()
        values['forces'][0, 0] = 2 * FMAX
        self.assertFalse(physically_converged(values))

    def test_step_budget_preserves_physical_exit(self):
        data = batch()
        count = CountUpdates()
        ctx = SimpleNamespace(batch=data)
        count(ctx, count.stage)
        data.status[0] = 1
        limits = Limits(count, max_steps=1, wall_seconds=60)
        limits(ctx, limits.stage)
        self.assertEqual(data.status.flatten().tolist(), [1, 2])
        self.assertEqual({k: v['status_code'] for k, v in limits.exits.items()}, {0: 1, 2: 2})
        count(ctx, count.stage)
        self.assertEqual(count.updates, {0: 1, 2: 1})

    def test_wall_timeout_has_distinct_status(self):
        data = batch()
        count = CountUpdates()
        ctx = SimpleNamespace(batch=data)
        count(ctx, count.stage)
        limits = Limits(count, max_steps=500, wall_seconds=60)
        limits.started -= 61
        limits(ctx, limits.stage)
        self.assertEqual(data.status.flatten().tolist(), [3, 3])


if __name__ == '__main__':
    unittest.main()
