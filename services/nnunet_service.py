from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from plugins.hook_types import HookType
from plugins.segmentation_hooks.segmentation_plugin_manager import (
    PipelineConfig,
    PipelineInput,
    PipelinePluginManager,
)


@dataclass
class NnUnetResult:
    mask: np.ndarray
    labels_mapping: dict
    output_path: Path
    raw_output: Any


class NnUnetService:
    """Wraps the nnUNet plugin pipeline to run inference on the current volume."""

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    def run_inference(
        self,
        *,
        volume: np.ndarray,
        raw_volume: np.ndarray | None,
        model_path: str | Path,
        output_path: str | Path,
        dataset_id: str = "current",
        on_success: Callable[[NnUnetResult], None],
        on_error: Callable[[Exception], None],
    ) -> None:
        """Execute nnUNet pipeline asynchronously and persist the NPZ result."""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Modèle nnUNet introuvable: {model_path}")

        output_path = Path(output_path)
        if output_path.suffix.lower() != ".npz":
            output_path = output_path.with_suffix(".npz")

        # Heavy imports (nnunetv2) are deferred to call time and also register plugins.
        try:
            from plugins.segmentation_hooks.segmentation_plugins import (  # noqa: F401
                nnunetv2_plugin,
            )
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "La dépendance nnunetv2 est requise pour lancer l'inférence."
            ) from exc

        cfg = PipelineConfig(
            workflow=[
                HookType.SEGMENTATION_PREPROCESS,
                HookType.SEGMENTATION_INFERENCE,
                HookType.SEGMENTATION_POSTPROCESS,
            ],
            registry_id="nnunetv2",
        )

        manager = PipelinePluginManager(cfg)

        seg_inp = PipelineInput(
            config={},
            pipeline_id="nnunet-inference",
            group_index=0,
            data_array=volume,
            raw_data_array=raw_volume if raw_volume is not None else volume,
            pipeline_context={"exported_model_folder": str(model_path)},
            dataset_id=dataset_id,
        )

        def _cleanup():
            try:
                manager.cleanup()
            except Exception:
                pass

        def _handle_result(res: Any) -> None:
            try:
                if isinstance(res, Exception):
                    raise res
                if res is None or not hasattr(res, "segmentation_mask"):
                    raise RuntimeError("Résultat nnUNet invalide ou vide.")

                mask = np.asarray(res.segmentation_mask.get("mask"), dtype=np.uint8)
                labels_mapping = res.labels_mapping if isinstance(res.labels_mapping, dict) else {}

                output_path.parent.mkdir(parents=True, exist_ok=True)
                # stocker le mapping complet (dict) dans une case objet pour le recharger aisément
                np.savez_compressed(
                    output_path,
                    mask=mask,
                    labels_mapping=np.array([labels_mapping], dtype=object),
                )

                result = NnUnetResult(
                    mask=mask,
                    labels_mapping=labels_mapping,
                    output_path=output_path,
                    raw_output=res,
                )
                self.logger.info("nnUNet terminé | masque=%s | labels=%s", mask.shape, list(labels_mapping.get("labels", {}).keys()))
                self.logger.info("Résultat enregistré dans : %s", output_path)
                on_success(result)
            except Exception as exc:
                self.logger.exception("Erreur lors de la finalisation nnUNet")
                on_error(exc)
            finally:
                _cleanup()

        manager.execute(seg_inp, metadata=None, completion_callback=_handle_result)
