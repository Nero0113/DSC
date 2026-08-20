#!/usr/bin/env python3
"""Dependency-light tests for deployment preprocessing and INT8 helpers."""

from __future__ import annotations

import unittest

import numpy as np

import export_int8_tflite as exporter
import runtime_imx93 as runtime


class DeploymentHelperTests(unittest.TestCase):
    def test_encoder_layout_is_nhwc(self):
        windows = np.arange(2 * 8 * 200, dtype=np.float32).reshape(2, 8, 200)
        converted = exporter._to_encoder_nhwc(windows)
        self.assertEqual(converted.shape, (2, 1, 200, 8))
        np.testing.assert_array_equal(converted[1, 0, :, 3], windows[1, 3, :])

    def test_statistics_match_training_contract(self):
        window = np.arange(8 * 200, dtype=np.float32).reshape(8, 200) / 100.0
        delta = window[:, 1:] - window[:, :-1]
        expected = np.concatenate(
            (
                window.mean(-1),
                window.std(-1),
                np.abs(delta).mean(-1),
                delta.std(-1),
            )
        )
        actual = runtime.StreamingClassifier._statistics(window)
        self.assertEqual(actual.shape, (32,))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)

    def test_int8_quantization_saturates(self):
        detail = {"dtype": np.int8, "quantization": (0.1, -3)}
        quantized = runtime._quantize(
            np.asarray([-100.0, 0.0, 100.0], dtype=np.float32), detail
        )
        np.testing.assert_array_equal(quantized, [-128, -3, 127])
        np.testing.assert_allclose(
            runtime._dequantize(quantized, detail), [-12.5, 0.0, 13.0]
        )


if __name__ == "__main__":
    unittest.main()
