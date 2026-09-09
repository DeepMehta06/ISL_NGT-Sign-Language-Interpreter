"""Model factory for SignBridge — builds ISL or NGT model from a config dict."""

from __future__ import annotations

from backend.core.logger import get_logger
from backend.ml.models.sign_model import SignBridgeModel

logger = get_logger(__name__)


def build_model(config: dict, language: str) -> SignBridgeModel:
    """Build a SignBridgeModel from a loaded YAML config dict.

    Args:
        config: Parsed YAML config dict containing a 'model' key.
        language: "ISL" or "NGT" — used for logging only.

    Returns:
        Initialised SignBridgeModel instance (not yet moved to device).

    Raises:
        ValueError: If num_classes is 0 or missing (NGT stub not ready).
        KeyError: If required keys are missing from config.
    """
    model_cfg = config["model"]
    num_classes: int = model_cfg["num_classes"]
    dropout: float = model_cfg.get("dropout", 0.4)
    hidden_size: int = model_cfg.get("hidden_size", 256)

    if num_classes < 1:
        raise ValueError(
            f"num_classes={num_classes} for {language}. "
            "NGT training requires Phase 3 annotation parsing first."
        )

    model = SignBridgeModel(
        num_classes=num_classes,
        dropout=dropout,
        hidden_size=hidden_size,
    )

    logger.info(
        f"Built SignBridgeModel | language={language} | "
        f"classes={num_classes} | params={model.get_num_parameters():,}"
    )
    return model
