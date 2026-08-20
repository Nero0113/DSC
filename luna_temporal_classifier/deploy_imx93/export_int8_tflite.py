#!/usr/bin/env python3
"""Export InceptionTemporalNet + context TCN as full-INT8 i.MX93 models.

The deployment graph uses NHWC Conv2D with a 1 x K kernel.  This is exactly
equivalent to the training-time Conv1d graph, while matching the operators
that Vela can compile for the i.MX93 Ethos-U65 NPU.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _dependencies():
    try:
        import tensorflow as tf
        import torch
    except ImportError as error:
        raise SystemExit(
            "Missing conversion dependency. Install requirements-convert.txt "
            "in a host-side virtual environment."
        ) from error
    return tf, torch


def _numpy(tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _conv1d_as_conv2d(tf, x, module, name: str, padding: str = "same"):
    """Create a weight-identical NHWC Conv2D from a PyTorch Conv1d."""
    layer = tf.keras.layers.Conv2D(
        filters=module.out_channels,
        kernel_size=(1, module.kernel_size[0]),
        strides=(1, module.stride[0]),
        padding=padding,
        dilation_rate=(1, module.dilation[0]),
        groups=module.groups,
        use_bias=module.bias is not None,
        name=name,
    )
    y = layer(x)
    # Torch: [out, in/groups, k]. Keras: [1, k, in/groups, out].
    weights = [np.transpose(_numpy(module.weight), (2, 1, 0))[None, ...]]
    if module.bias is not None:
        weights.append(_numpy(module.bias))
    layer.set_weights(weights)
    return y


def _batch_norm(tf, x, module, name: str):
    layer = tf.keras.layers.BatchNormalization(
        axis=-1, epsilon=module.eps, center=True, scale=True, name=name
    )
    y = layer(x, training=False)
    layer.set_weights(
        [
            _numpy(module.weight),
            _numpy(module.bias),
            _numpy(module.running_mean),
            _numpy(module.running_var),
        ]
    )
    return y


def _dense(tf, x, module, name: str):
    layer = tf.keras.layers.Dense(
        module.out_features, use_bias=module.bias is not None, name=name
    )
    y = layer(x)
    weights = [_numpy(module.weight).T]
    if module.bias is not None:
        weights.append(_numpy(module.bias))
    layer.set_weights(weights)
    return y


def build_encoder_keras(tf, encoder):
    """Build the window encoder, ending at the 96-value pooled embedding."""
    inputs = tf.keras.Input(batch_shape=(1, 1, 200, 8), name="sensor_window")
    x = _conv1d_as_conv2d(tf, inputs, encoder.stem[0], "stem_conv")
    x = _batch_norm(tf, x, encoder.stem[1], "stem_bn")
    x = tf.keras.layers.ReLU(name="stem_relu")(x)

    block_number = 0
    for module in encoder.features:
        if module.__class__.__name__ == "InceptionBlock1D":
            block_number += 1
            prefix = f"inception_{block_number}"
            reduced = _conv1d_as_conv2d(
                tf, x, module.reduce, f"{prefix}_reduce"
            )
            branches = [
                _conv1d_as_conv2d(
                    tf, reduced, conv, f"{prefix}_conv_{conv.kernel_size[0]}"
                )
                for conv in module.convs
            ]
            pooled = tf.keras.layers.MaxPool2D(
                pool_size=(1, 3), strides=(1, 1), padding="same",
                name=f"{prefix}_pool",
            )(x)
            pooled = _conv1d_as_conv2d(
                tf, pooled, module.pool_branch[1], f"{prefix}_pool_project"
            )
            merged = tf.keras.layers.Concatenate(axis=-1, name=f"{prefix}_concat")(
                branches + [pooled]
            )
            merged = _batch_norm(tf, merged, module.bn, f"{prefix}_bn")
            x = tf.keras.layers.Add(name=f"{prefix}_residual")([merged, x])
            x = tf.keras.layers.ReLU(name=f"{prefix}_relu")(x)
        else:
            # The only other modules in encoder.features are MaxPool1d(2).
            x = tf.keras.layers.MaxPool2D(
                pool_size=(1, 2), strides=(1, 2), padding="valid",
                name=f"downsample_{block_number}",
            )(x)

    average = tf.keras.layers.GlobalAveragePooling2D(name="average_pool")(x)
    maximum = tf.keras.layers.GlobalMaxPooling2D(name="maximum_pool")(x)
    embedding = tf.keras.layers.Concatenate(name="embedding")([average, maximum])
    return tf.keras.Model(inputs, embedding, name="inception_temporal_encoder")


def build_context_keras(tf, context_model, context_windows: int):
    """Build the causal context head with [N, 1, time, feature] NHWC input."""
    feature_dim = context_model.stem[0].in_channels
    inputs = tf.keras.Input(
        batch_shape=(1, 1, context_windows, feature_dim), name="context_features"
    )
    x = _conv1d_as_conv2d(tf, inputs, context_model.stem[0], "stem_conv")
    x = _batch_norm(tf, x, context_model.stem[1], "stem_bn")
    x = tf.keras.layers.ReLU(name="stem_relu")(x)

    for number, block in enumerate(context_model.blocks, start=1):
        residual = x
        dilation = int(block.dilation)
        x = tf.keras.layers.ZeroPadding2D(
            padding=((0, 0), (2 * dilation, 0)), name=f"causal_{number}_pad"
        )(x)
        x = _conv1d_as_conv2d(
            tf, x, block.conv, f"causal_{number}_conv", padding="valid"
        )
        x = _batch_norm(tf, x, block.bn, f"causal_{number}_bn")
        # Dropout is intentionally absent: PyTorch eval() also makes it an identity.
        x = tf.keras.layers.Add(name=f"causal_{number}_residual")([x, residual])
        x = tf.keras.layers.ReLU(name=f"causal_{number}_relu")(x)

    latest = tf.keras.layers.Lambda(
        lambda value: value[:, 0, -1, :], name="latest_timestep"
    )(x)
    average = tf.keras.layers.GlobalAveragePooling2D(name="context_average")(x)
    summary = tf.keras.layers.Concatenate(name="context_summary")([latest, average])
    hidden = _dense(tf, summary, context_model.head[0], "head_dense")
    hidden = tf.keras.layers.ReLU(name="head_relu")(hidden)
    logits = _dense(tf, hidden, context_model.head[3], "logits")
    return tf.keras.Model(inputs, logits, name="hierarchical_causal_tcn")


def _sample_indices(length: int, count: int, seed: int) -> np.ndarray:
    if length <= 0:
        raise ValueError("Calibration dataset is empty")
    count = min(length, count)
    generator = np.random.default_rng(seed)
    return np.sort(generator.choice(length, size=count, replace=False))


def _to_encoder_nhwc(windows: np.ndarray) -> np.ndarray:
    windows = np.asarray(windows, dtype=np.float32)
    if windows.ndim != 3 or windows.shape[1:] != (8, 200):
        raise ValueError(f"Expected [N, 8, 200], got {windows.shape}")
    return np.transpose(windows, (0, 2, 1))[:, None, :, :]


def _make_context_inputs(features: np.ndarray, dataset, indices: Iterable[int]):
    output = []
    for index in indices:
        _, source = dataset.indices[int(index)]
        output.append(features[source][None, :, :])
    return np.stack(output).astype(np.float32)


def _convert_full_int8(tf, model, representative_inputs: np.ndarray) -> bytes:
    def representative_dataset():
        for sample in representative_inputs:
            yield [sample[None, ...].astype(np.float32, copy=False)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def _quantize(value: np.ndarray, detail: dict) -> np.ndarray:
    dtype = detail["dtype"]
    if dtype == np.float32:
        return value.astype(np.float32)
    scale, zero_point = detail["quantization"]
    if scale <= 0:
        raise ValueError(f"Invalid quantization parameters: {detail['quantization']}")
    limits = np.iinfo(dtype)
    return np.clip(np.rint(value / scale + zero_point), limits.min, limits.max).astype(dtype)


def _dequantize(value: np.ndarray, detail: dict) -> np.ndarray:
    if detail["dtype"] == np.float32:
        return value.astype(np.float32)
    scale, zero_point = detail["quantization"]
    return (value.astype(np.float32) - zero_point) * scale


def _tflite_infer(tf, model_path: Path, inputs: np.ndarray) -> np.ndarray:
    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=2)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    outputs = []
    for sample in inputs:
        interpreter.set_tensor(
            input_detail["index"], _quantize(sample[None, ...], input_detail)
        )
        interpreter.invoke()
        outputs.append(_dequantize(interpreter.get_tensor(output_detail["index"]), output_detail)[0])
    return np.stack(outputs)


def _torch_encoder_embeddings(torch, encoder, windows: np.ndarray) -> np.ndarray:
    values = torch.from_numpy(windows.astype(np.float32, copy=False))
    with torch.no_grad():
        temporal = encoder.features(encoder.stem(values))
        embedding = torch.cat(
            (encoder.avg_pool(temporal), encoder.max_pool(temporal)), dim=1
        ).squeeze(-1)
    return embedding.numpy()


def _checkpoint_width(checkpoint: dict) -> int:
    if "width" in checkpoint:
        return int(checkpoint["width"])
    return int(checkpoint.get("config", {}).get("width", 48))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--context-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--calibration-split", choices=("train", "val"), default="train")
    parser.add_argument("--calibration-samples", type=int, default=512)
    parser.add_argument("--verify-split", choices=("train", "val"), default="val")
    parser.add_argument("--verify-samples", type=int, default=64)
    parser.add_argument("--context", type=int, default=25)
    parser.add_argument("--clip", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--float-parity-atol", type=float, default=3e-4)
    args = parser.parse_args()

    if args.context != 25:
        raise ValueError("The trained deployment contract requires context=25")
    if args.calibration_samples < 1 or args.verify_samples < 1:
        raise ValueError("Sample counts must be positive")

    tf, torch = _dependencies()
    from context_tcn import HierarchicalCausalTCN
    from model import InceptionTemporalNet
    from train import LABELS
    from train_context_tcn import ContextDataset, extract_features

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    encoder_checkpoint = torch.load(
        args.encoder_checkpoint, map_location="cpu", weights_only=False
    )
    context_checkpoint = torch.load(
        args.context_checkpoint, map_location="cpu", weights_only=False
    )
    encoder = InceptionTemporalNet(
        width=_checkpoint_width(encoder_checkpoint), dropout=0.15
    ).eval()
    encoder.load_state_dict(encoder_checkpoint["model_state"])

    feature_mean = np.asarray(context_checkpoint["feature_mean"], dtype=np.float32).reshape(-1)
    feature_std = np.asarray(context_checkpoint["feature_std"], dtype=np.float32).reshape(-1)
    if np.any(feature_std <= 0):
        raise ValueError("Context checkpoint contains a non-positive feature_std")
    context_width = int(
        context_checkpoint["model_state"]["stem.0.weight"].shape[0]
    )
    context_model = HierarchicalCausalTCN(
        feature_dim=len(feature_mean), width=context_width
    ).eval()
    context_model.load_state_dict(context_checkpoint["model_state"])

    encoder_keras = build_encoder_keras(tf, encoder)
    context_keras = build_context_keras(tf, context_model, args.context)

    calibration_path = args.data / f"{args.calibration_split}.npz"
    calibration_data = np.load(calibration_path, allow_pickle=False)
    calibration_windows = np.clip(
        calibration_data["X"], -args.clip, args.clip
    ).astype(np.float32)
    encoder_indices = _sample_indices(
        len(calibration_windows), args.calibration_samples, args.seed
    )
    encoder_representatives = _to_encoder_nhwc(calibration_windows[encoder_indices])

    # Context calibration uses the exact training-time feature extraction and
    # normalization. The deployed encoder's output error is checked below.
    calibration_features, calibration_metadata = extract_features(
        calibration_path, encoder, torch.device("cpu")
    )
    calibration_features = (
        calibration_features - feature_mean[None, :]
    ) / feature_std[None, :]
    calibration_context = ContextDataset(
        calibration_features,
        calibration_metadata["y"],
        calibration_metadata["trip_id"],
        calibration_metadata["window_start_s"],
        args.context,
    )
    context_indices = _sample_indices(
        len(calibration_context), args.calibration_samples, args.seed + 1
    )
    context_representatives = _make_context_inputs(
        calibration_features, calibration_context, context_indices
    )

    # Check the Conv1d -> Conv2D weight mapping before quantization.
    verify_path = args.data / f"{args.verify_split}.npz"
    verify_data = np.load(verify_path, allow_pickle=False)
    verify_windows_all = np.clip(verify_data["X"], -args.clip, args.clip).astype(np.float32)
    verify_indices = _sample_indices(len(verify_windows_all), args.verify_samples, args.seed + 2)
    verify_windows = verify_windows_all[verify_indices]
    torch_embeddings = _torch_encoder_embeddings(torch, encoder, verify_windows)
    keras_embeddings = encoder_keras.predict(
        _to_encoder_nhwc(verify_windows), verbose=0
    )
    encoder_float_error = float(np.max(np.abs(torch_embeddings - keras_embeddings)))

    verify_features_raw, verify_metadata = extract_features(
        verify_path, encoder, torch.device("cpu")
    )
    verify_features = (
        verify_features_raw - feature_mean[None, :]
    ) / feature_std[None, :]
    verify_context = ContextDataset(
        verify_features,
        verify_metadata["y"],
        verify_metadata["trip_id"],
        verify_metadata["window_start_s"],
        args.context,
    )
    verify_context_indices = _sample_indices(
        len(verify_context), args.verify_samples, args.seed + 3
    )
    verify_context_inputs = _make_context_inputs(
        verify_features, verify_context, verify_context_indices
    )
    torch_context_inputs = torch.from_numpy(
        verify_context_inputs[:, 0].transpose(0, 2, 1).copy()
    )
    with torch.no_grad():
        torch_logits = context_model(torch_context_inputs).numpy()
    keras_logits = context_keras.predict(verify_context_inputs, verbose=0)
    context_float_error = float(np.max(np.abs(torch_logits - keras_logits)))
    if max(encoder_float_error, context_float_error) > args.float_parity_atol:
        raise RuntimeError(
            "Conv1d -> Conv2D parity check failed: "
            f"encoder={encoder_float_error:.6g}, context={context_float_error:.6g}"
        )

    encoder_path = args.out / "inception_encoder_int8.tflite"
    context_path = args.out / "context_tcn_int8.tflite"
    encoder_path.write_bytes(
        _convert_full_int8(tf, encoder_keras, encoder_representatives)
    )
    context_path.write_bytes(
        _convert_full_int8(tf, context_keras, context_representatives)
    )

    quantized_embeddings = _tflite_infer(
        tf, encoder_path, _to_encoder_nhwc(verify_windows)
    )
    quantized_logits = _tflite_infer(tf, context_path, verify_context_inputs)

    # End-to-end check: replace every encoder embedding required by the chosen
    # contexts with the INT8 encoder output, then run those contexts through
    # the INT8 head. This catches error propagation across the model boundary.
    source_indices = np.unique(
        np.concatenate(
            [verify_context.indices[int(index)][1] for index in verify_context_indices]
        )
    )
    deployed_features_raw = verify_features_raw.copy()
    deployed_features_raw[source_indices, : torch_embeddings.shape[1]] = _tflite_infer(
        tf,
        encoder_path,
        _to_encoder_nhwc(verify_windows_all[source_indices]),
    )
    deployed_features = (
        deployed_features_raw - feature_mean[None, :]
    ) / feature_std[None, :]
    deployed_context_inputs = _make_context_inputs(
        deployed_features, verify_context, verify_context_indices
    )
    deployed_logits = _tflite_infer(tf, context_path, deployed_context_inputs)
    report = {
        "float_mapping": {
            "encoder_max_abs_error": encoder_float_error,
            "context_max_abs_error": context_float_error,
        },
        "int8": {
            "encoder_mae": float(np.mean(np.abs(torch_embeddings - quantized_embeddings))),
            "encoder_max_abs_error": float(np.max(np.abs(torch_embeddings - quantized_embeddings))),
            "context_logits_mae": float(np.mean(np.abs(torch_logits - quantized_logits))),
            "context_logits_max_abs_error": float(np.max(np.abs(torch_logits - quantized_logits))),
            "context_top1_agreement": float(
                np.mean(torch_logits.argmax(1) == quantized_logits.argmax(1))
            ),
            "end_to_end_logits_mae": float(
                np.mean(np.abs(torch_logits - deployed_logits))
            ),
            "end_to_end_logits_max_abs_error": float(
                np.max(np.abs(torch_logits - deployed_logits))
            ),
            "end_to_end_top1_agreement": float(
                np.mean(torch_logits.argmax(1) == deployed_logits.argmax(1))
            ),
        },
        "samples": {
            "calibration": int(len(encoder_representatives)),
            "verification": int(len(verify_windows)),
        },
    }

    np.savez(
        args.out / "preprocessing.npz",
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    manifest = {
        "target": "NXP i.MX93 Ethos-U65-256",
        "format": "full-integer INT8 TFLite; compile with NXP Vela",
        "encoder_model": encoder_path.name,
        "context_model": context_path.name,
        "preprocessing": "preprocessing.npz",
        "labels": list(LABELS),
        "clip": args.clip,
        "window_shape_chw": [8, 200],
        "encoder_input_nhwc": [1, 1, 200, 8],
        "embedding_dim": int(torch_embeddings.shape[1]),
        "statistics_dim": 32,
        "context_feature_dim": int(len(feature_mean)),
        "context_windows": args.context,
        "context_input_nhwc": [1, 1, args.context, int(len(feature_mean))],
        "window_seconds": 20,
        "stride_seconds": 5,
        "smoothing_alpha": 0.05,
        "delegate": "/usr/lib/libethosu_delegate.so",
        "quantization_report": report,
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
