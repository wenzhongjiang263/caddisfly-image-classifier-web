"""Reusable inference helpers for the packaged caddisfly classifier."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torch import nn
from torchvision import models, transforms


PACKAGE_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PACKAGE_ROOT / "model"
CHECKPOINT_PATH = MODEL_DIR / "best_model.pth"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"
MODEL_CONFIG_PATH = MODEL_DIR / "model_config.json"

DEFAULT_IMAGE_SIZE = 224
DEFAULT_RESIZE_SIZE = 256
DEFAULT_IMAGENET_MEAN = [0.485, 0.456, 0.406]
DEFAULT_IMAGENET_STD = [0.229, 0.224, 0.225]
SUPPORTED_MODEL_NAME = "mobilenet_v2"


class InferenceError(RuntimeError):
    """Raised when model loading or prediction fails with a clear message."""


@dataclass(frozen=True)
class ModelArtifacts:
    model: nn.Module
    preprocess: transforms.Compose
    class_names: List[str]
    config: Dict[str, Any]
    device: torch.device


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise InferenceError(f"Required file is missing: {path}")
    if not path.is_file():
        raise InferenceError(f"Required path is not a file: {path}")
    try:
        with path.open("r", encoding="utf-8") as json_file:
            return json.load(json_file)
    except json.JSONDecodeError as exc:
        raise InferenceError(f"Could not parse JSON file: {path}") from exc


def _build_mobilenet_v2(num_classes: int) -> nn.Module:
    try:
        model = models.mobilenet_v2(weights=None)
    except TypeError:
        model = models.mobilenet_v2(pretrained=False)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def _build_preprocess(model_config: Dict[str, Any]) -> transforms.Compose:
    image_size = int(model_config.get("image_size", DEFAULT_IMAGE_SIZE))
    resize_size = int(model_config.get("resize_size", DEFAULT_RESIZE_SIZE))
    mean = model_config.get("normalization_mean", DEFAULT_IMAGENET_MEAN)
    std = model_config.get("normalization_std", DEFAULT_IMAGENET_STD)

    if not isinstance(mean, list) or not isinstance(std, list) or len(mean) != 3 or len(std) != 3:
        raise InferenceError("Model configuration must contain three-channel normalization mean and std.")

    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def _load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    if not path.exists():
        raise InferenceError(f"Model checkpoint is missing: {path}")
    if not path.is_file():
        raise InferenceError(f"Model checkpoint path is not a file: {path}")
    try:
        # The packaged checkpoint is a trusted local full checkpoint, not a
        # weights-only file. PyTorch 2.6+ defaults can reject it otherwise.
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    except Exception as exc:
        raise InferenceError(f"Failed to load model checkpoint: {path}") from exc

    if not isinstance(checkpoint, dict):
        raise InferenceError("Model checkpoint must be a dictionary checkpoint.")
    return checkpoint


def load_model_artifacts(
    checkpoint_path: Path = CHECKPOINT_PATH,
    class_names_path: Path = CLASS_NAMES_PATH,
    model_config_path: Path = MODEL_CONFIG_PATH,
    device_name: str = "cpu",
) -> ModelArtifacts:
    """Load model, preprocessing, class names, and configuration for inference."""
    device = torch.device(device_name)
    class_names = _read_json(class_names_path)
    model_config = _read_json(model_config_path)

    if not isinstance(class_names, list) or not all(isinstance(name, str) for name in class_names):
        raise InferenceError("class_names.json must contain a JSON list of class-name strings.")
    if not isinstance(model_config, dict):
        raise InferenceError("model_config.json must contain a JSON object.")
    if len(class_names) != int(model_config.get("num_classes", len(class_names))):
        raise InferenceError("Class-name count does not match model_config.json num_classes.")

    model_name = model_config.get("model_name", SUPPORTED_MODEL_NAME)
    if model_name != SUPPORTED_MODEL_NAME:
        raise InferenceError(f"Unsupported model architecture in config: {model_name}")

    checkpoint = _load_checkpoint(checkpoint_path, device)
    checkpoint_model_name = checkpoint.get("model_name", SUPPORTED_MODEL_NAME)
    if checkpoint_model_name != SUPPORTED_MODEL_NAME:
        raise InferenceError(f"Unsupported model architecture in checkpoint: {checkpoint_model_name}")

    checkpoint_class_names = checkpoint.get("class_names")
    if isinstance(checkpoint_class_names, list) and checkpoint_class_names != class_names:
        raise InferenceError("Checkpoint class order does not match class_names.json.")

    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise InferenceError("Checkpoint does not contain model_state_dict.")

    classifier_weight = state_dict.get("classifier.1.weight")
    if classifier_weight is None or classifier_weight.shape[0] != len(class_names):
        raise InferenceError("Checkpoint output layer does not match the loaded class count.")

    model = _build_mobilenet_v2(num_classes=len(class_names)).to(device)
    try:
        model.load_state_dict(state_dict)
    except Exception as exc:
        raise InferenceError("Failed to load model weights into MobileNetV2.") from exc
    model.eval()

    return ModelArtifacts(
        model=model,
        preprocess=_build_preprocess(model_config),
        class_names=list(class_names),
        config=model_config,
        device=device,
    )


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """Decode uploaded image bytes and return an RGB Pillow image."""
    if not image_bytes:
        raise InferenceError("The uploaded image is empty.")
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image = ImageOps.exif_transpose(image)
            return image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise InferenceError("The uploaded file is not a valid readable image.") from exc
    except Exception as exc:
        raise InferenceError("Could not open the uploaded image.") from exc


def top_k_predictions(
    probabilities: Sequence[float],
    class_names: Sequence[str],
    k: int = 5,
) -> List[Dict[str, Any]]:
    """Return top-k class probabilities in descending order."""
    probabilities_array = np.asarray(probabilities, dtype=np.float64)
    if probabilities_array.ndim != 1 or probabilities_array.size != len(class_names):
        raise InferenceError("Probability vector length does not match class names.")

    top_count = min(k, len(class_names))
    top_indices = np.argsort(probabilities_array)[::-1][:top_count]
    return [
        {
            "rank": rank,
            "class_name": class_names[int(class_index)],
            "class_index": int(class_index),
            "probability": float(probabilities_array[int(class_index)]),
        }
        for rank, class_index in enumerate(top_indices, start=1)
    ]


def summarize_probabilities(
    probabilities: Sequence[float],
    class_names: Sequence[str],
    k: int = 5,
) -> Dict[str, Any]:
    """Build a prediction summary from a probability vector."""
    probabilities_array = np.asarray(probabilities, dtype=np.float64)
    top_predictions = top_k_predictions(probabilities_array, class_names=class_names, k=k)
    best = top_predictions[0]
    return {
        "prediction": best["class_name"],
        "class_index": best["class_index"],
        "confidence": best["probability"],
        "probabilities": probabilities_array,
        "top_k": top_predictions,
    }


def predict_image_bytes(
    image_bytes: bytes,
    artifacts: ModelArtifacts,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Run model inference for uploaded image bytes."""
    image = load_image_from_bytes(image_bytes)
    try:
        image_tensor = artifacts.preprocess(image).unsqueeze(0).to(artifacts.device)
        with torch.inference_mode():
            outputs = artifacts.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()
    except Exception as exc:
        raise InferenceError("Prediction failed while running the model.") from exc

    if probabilities.shape[0] != len(artifacts.class_names):
        raise InferenceError("Model output length does not match the loaded class list.")

    return summarize_probabilities(probabilities, artifacts.class_names, k=top_k)


def combine_probabilities(
    probabilities_image_1: Sequence[float],
    probabilities_image_2: Sequence[float],
) -> np.ndarray:
    """Average two probability vectors for the same specimen."""
    first = np.asarray(probabilities_image_1, dtype=np.float64)
    second = np.asarray(probabilities_image_2, dtype=np.float64)
    if first.shape != second.shape:
        raise InferenceError("Cannot combine probability vectors with different shapes.")
    if first.ndim != 1:
        raise InferenceError("Probability vectors must be one-dimensional.")
    return (first + second) / 2.0
