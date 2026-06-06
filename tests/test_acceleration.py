#!/usr/bin/env python
# -*- coding: utf-8 -*-

from contextlib import contextmanager

import numpy as np

from properimage.acceleration import (
    AccelerationConfig,
    clear_acceleration_cache,
    configure_acceleration,
    get_acceleration_config,
    subtract_batch,
)


def _prepared(index, ref, new, smooth_psf, fitted_psf, persistent):
    return index, ref, new, ref, new, False


def _fake_subtract(ref, new, **kwargs):
    value = float(ref) + float(new)
    data = np.full((2, 2), value)
    return data, data, data, np.zeros((2, 2), dtype=bool)


@contextmanager
def _null_cuda_context(device_id, use_stream=True):
    yield


def test_configure_acceleration_updates_defaults():
    original = get_acceleration_config()
    try:
        updated = configure_acceleration(devices="cpu", cpu_workers=3)
        assert updated.devices == "cpu"
        assert get_acceleration_config().cpu_workers == 3
    finally:
        configure_acceleration(**original.__dict__)


def test_subtract_batch_cpu(monkeypatch):
    import properimage.acceleration as acceleration
    import properimage.operations as operations

    monkeypatch.setattr(acceleration, "_prepare_pair", _prepared)
    monkeypatch.setattr(operations, "subtract", _fake_subtract)

    summary, results = subtract_batch(
        [(1, 2), (3, 4)],
        return_results=True,
        acceleration=AccelerationConfig(devices="cpu", cpu_workers=2),
        use_gpu=False,
    )

    assert summary.ok
    assert summary.devices == ()
    assert [task.device_id for task in summary.tasks] == [None, None]
    assert len(results) == 2
    assert np.all(results[0][0] == 3.0)


def test_subtract_batch_distributes_virtual_cuda_devices(monkeypatch):
    import properimage.acceleration as acceleration
    import properimage.operations as operations

    monkeypatch.setattr(acceleration, "_prepare_pair", _prepared)
    monkeypatch.setattr(operations, "subtract", _fake_subtract)
    monkeypatch.setattr(acceleration, "resolve_cuda_devices", lambda *a, **k: (0, 1))
    monkeypatch.setattr(acceleration, "cuda_device_context", _null_cuda_context)

    summary = subtract_batch(
        [(1, 10), (2, 20), (3, 30), (4, 40)],
        acceleration=AccelerationConfig(devices="auto", prefetch=4),
        use_gpu="auto",
    )

    assert summary.ok
    assert summary.devices == (0, 1)
    assert {task.device_id for task in summary.tasks} == {0, 1}


def test_subtract_batch_keeps_running_after_task_failure(monkeypatch):
    import properimage.acceleration as acceleration
    import properimage.operations as operations

    def subtract_with_failure(ref, new, **kwargs):
        if ref == "bad":
            raise ValueError("broken image")
        return _fake_subtract(1, 1, **kwargs)

    monkeypatch.setattr(acceleration, "_prepare_pair", _prepared)
    monkeypatch.setattr(operations, "subtract", subtract_with_failure)

    summary = subtract_batch(
        [("ok", "a"), ("bad", "b"), ("ok", "c")],
        acceleration=AccelerationConfig(devices="cpu", cpu_workers=2),
        use_gpu=False,
    )

    assert not summary.ok
    assert len(summary.failed) == 1
    assert summary.failed[0].index == 1
    assert "broken image" in summary.failed[0].error
    assert sum(task.ok for task in summary.tasks) == 2


def test_clear_acceleration_cache_is_idempotent():
    clear_acceleration_cache()
    clear_acceleration_cache()
