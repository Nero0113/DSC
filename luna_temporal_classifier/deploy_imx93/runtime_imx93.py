#!/usr/bin/env python3
"""Streaming INT8 inference for InceptionTemporalNet + context TCN on i.MX93."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np


def _tflite_api():
    try:
        from tflite_runtime.interpreter import Interpreter, load_delegate
    except ImportError:
        try:
            import tensorflow as tf

            Interpreter = tf.lite.Interpreter
            load_delegate = tf.lite.experimental.load_delegate
        except ImportError as error:
            raise SystemExit(
                "TensorFlow Lite runtime is missing. Use the NXP BSP Python image "
                "or install tflite-runtime."
            ) from error
    return Interpreter, load_delegate


def _quantize(value: np.ndarray, detail: dict) -> np.ndarray:
    dtype = detail["dtype"]
    if dtype == np.float32:
        return value.astype(np.float32)
    scale, zero_point = detail["quantization"]
    if scale <= 0:
        raise RuntimeError(f"Invalid input quantization: {detail['quantization']}")
    limits = np.iinfo(dtype)
    return np.clip(np.rint(value / scale + zero_point), limits.min, limits.max).astype(dtype)


def _dequantize(value: np.ndarray, detail: dict) -> np.ndarray:
    if detail["dtype"] == np.float32:
        return value.astype(np.float32)
    scale, zero_point = detail["quantization"]
    return (value.astype(np.float32) - zero_point) * scale


class TFLiteModel:
    def __init__(
        self,
        model_path: Path,
        delegate_path: Optional[str],
        cache_path: Optional[Path],
        threads: int,
    ) -> None:
        Interpreter, load_delegate = _tflite_api()
        delegates = None
        if delegate_path:
            options = {}
            if cache_path is not None:
                options["cache_file_path"] = str(cache_path)
            delegates = [load_delegate(delegate_path, options)]
        self.interpreter = Interpreter(
            model_path=str(model_path),
            num_threads=threads,
            experimental_delegates=delegates,
        )
        self.interpreter.allocate_tensors()
        inputs = self.interpreter.get_input_details()
        outputs = self.interpreter.get_output_details()
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError("Deployment model must have exactly one input and one output")
        self.input = inputs[0]
        self.output = outputs[0]

    def __call__(self, value: np.ndarray) -> np.ndarray:
        expected = tuple(int(item) for item in self.input["shape"])
        if value.shape != expected:
            raise ValueError(f"Model expects {expected}, got {value.shape}")
        self.interpreter.set_tensor(self.input["index"], _quantize(value, self.input))
        self.interpreter.invoke()
        return _dequantize(
            self.interpreter.get_tensor(self.output["index"]), self.output
        )


class StreamingClassifier:
    """Stateful classifier. Call reset() at every trip/session boundary."""

    def __init__(
        self,
        model_dir: Path,
        encoder_name: str,
        context_name: str,
        delegate_path: Optional[str],
        cache_dir: Optional[Path],
        threads: int,
    ) -> None:
        manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
        preprocessing = np.load(model_dir / manifest["preprocessing"], allow_pickle=False)
        self.mean = preprocessing["feature_mean"].astype(np.float32).reshape(-1)
        self.std = preprocessing["feature_std"].astype(np.float32).reshape(-1)
        if np.any(self.std <= 0):
            raise ValueError("preprocessing.npz contains a non-positive feature_std")
        self.clip = float(manifest["clip"])
        self.context_windows = int(manifest["context_windows"])
        self.alpha = float(manifest["smoothing_alpha"])
        self.labels = list(manifest["labels"])
        self.history: deque[np.ndarray] = deque(maxlen=self.context_windows)
        self.smooth_logits: Optional[np.ndarray] = None

        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
        encoder_cache = None if cache_dir is None else cache_dir / "encoder.cache"
        context_cache = None if cache_dir is None else cache_dir / "context.cache"
        self.encoder = TFLiteModel(
            model_dir / encoder_name, delegate_path, encoder_cache, threads
        )
        self.context = TFLiteModel(
            model_dir / context_name, delegate_path, context_cache, threads
        )

    def reset(self) -> None:
        self.history.clear()
        self.smooth_logits = None

    @staticmethod
    def _statistics(window: np.ndarray) -> np.ndarray:
        delta = window[:, 1:] - window[:, :-1]
        return np.concatenate(
            (
                window.mean(axis=-1),
                window.std(axis=-1),
                np.abs(delta).mean(axis=-1),
                delta.std(axis=-1),
            )
        ).astype(np.float32)

    def push(self, window_chw: np.ndarray) -> dict:
        window = np.asarray(window_chw, dtype=np.float32)
        if window.shape != (8, 200):
            raise ValueError(f"Expected one [8, 200] window, got {window.shape}")
        window = np.clip(window, -self.clip, self.clip)

        encoder_input = np.transpose(window, (1, 0))[None, None, :, :]
        embedding = self.encoder(encoder_input)[0].reshape(-1)
        feature = np.concatenate((embedding, self._statistics(window)))
        if feature.shape != self.mean.shape:
            raise RuntimeError(
                f"Feature shape {feature.shape} does not match normalizer {self.mean.shape}"
            )
        feature = ((feature - self.mean) / self.std).astype(np.float32)
        self.history.append(feature)

        values = list(self.history)
        padded = [values[0]] * (self.context_windows - len(values)) + values
        context_input = np.stack(padded)[None, None, :, :]
        logits = self.context(context_input)[0].reshape(-1)
        if self.smooth_logits is None:
            self.smooth_logits = logits.copy()
        else:
            self.smooth_logits = (
                self.alpha * logits + (1.0 - self.alpha) * self.smooth_logits
            )
        prediction = int(np.argmax(self.smooth_logits))
        return {
            "prediction": prediction,
            "label": self.labels[prediction],
            "logits": logits.tolist(),
            "smoothed_logits": self.smooth_logits.tolist(),
            "context_valid": len(self.history),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True,
                        help=".npy [N,8,200] or .npz containing X")
    parser.add_argument("--encoder-model", default="inception_encoder_int8_vela.tflite")
    parser.add_argument("--context-model", default="context_tcn_int8_vela.tflite")
    parser.add_argument("--delegate", choices=("ethosu", "cpu"), default="ethosu")
    parser.add_argument("--delegate-path", default="/usr/lib/libethosu_delegate.so")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="Useful with uncompiled TFLite; omit for Vela models")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    delegate_path = args.delegate_path if args.delegate == "ethosu" else None
    classifier = StreamingClassifier(
        args.model_dir,
        args.encoder_model,
        args.context_model,
        delegate_path,
        args.cache_dir,
        args.threads,
    )
    loaded = np.load(args.input, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        windows = loaded["X"]
        trips = loaded["trip_id"] if "trip_id" in loaded.files else None
    else:
        windows = loaded
        trips = None
    if windows.ndim != 3 or windows.shape[1:] != (8, 200):
        raise ValueError(f"Input must be [N,8,200], got {windows.shape}")

    previous_trip = None
    for index, window in enumerate(windows):
        trip = None if trips is None else str(trips[index])
        if index == 0 or (trips is not None and trip != previous_trip):
            classifier.reset()
        result = classifier.push(window)
        result.update({"index": index, "trip_id": trip})
        print(json.dumps(result, ensure_ascii=False), flush=True)
        previous_trip = trip


if __name__ == "__main__":
    main()
