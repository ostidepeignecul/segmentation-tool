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
from services.overlay_export import OverlayExport


@dataclass
class Mask2FormerResult:
    mask: np.ndarray
    labels_mapping: dict
    output_path: Path
    raw_output: Any


class Mask2FormerService:
    """Wraps the Mask2Former plugin pipeline to run inference on the current volume."""

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger
        self.label_manifest_export = OverlayExport()

    def run_inference(
        self,
        *,
        volume: np.ndarray,
        raw_volume: np.ndarray | None,
        model_path: str | Path,
        output_path: str | Path,
        dataset_id: str = "current",
        batch_size: int = 8,
        on_success: Callable[[Mask2FormerResult], None],
        on_error: Callable[[Exception], None],
    ) -> None:
        """Execute Mask2Former pipeline asynchronously and persist the NPZ result."""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Modèle Mask2Former introuvable: {model_path}")

        if batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")

        output_path = Path(output_path)
        if output_path.suffix.lower() != ".npz":
            output_path = output_path.with_suffix(".npz")

        # Heavy imports are deferred to call time and register plugins.
        try:
            from plugins.segmentation_hooks.segmentation_plugins import (  # noqa: F401
                mask2former_plugin,
            )
        except ImportError as exc:
            raise RuntimeError(
                "La dépendance transformers est requise pour lancer l'inférence Mask2Former."
            ) from exc

        cfg = PipelineConfig(
            workflow=[
                HookType.SEGMENTATION_PREPROCESS,
                HookType.SEGMENTATION_INFERENCE,
                HookType.SEGMENTATION_POSTPROCESS,
            ],
            registry_id="mask2former",
        )

        manager = PipelinePluginManager(cfg)

        seg_inp = PipelineInput(
            config={
                "batch_size": batch_size,
            },
            pipeline_id="mask2former-inference",
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
                    raise RuntimeError("Résultat Mask2Former invalide ou vide.")

                mask = np.asarray(res.segmentation_mask.get("mask"), dtype=np.uint8)
                labels_mapping = res.labels_mapping if isinstance(res.labels_mapping, dict) else {}

                output_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    output_path,
                    mask=mask,
                    labels_mapping=np.array([labels_mapping], dtype=object),
                )
                self.label_manifest_export.save_label_manifest(
                    output_path,
                    mask,
                    export_target="mask2former",
                    primary_axis_name=None,
                    label_palette=None,
                    label_visibility=None,
                    label_names=OverlayExport.label_names_from_mapping(labels_mapping),
                )

                result = Mask2FormerResult(
                    mask=mask,
                    labels_mapping=labels_mapping,
                    output_path=output_path,
                    raw_output=res,
                )
                self.logger.info("Mask2Former terminé | masque=%s | labels=%s", mask.shape, list(labels_mapping.get("labels", {}).keys()))
                self.logger.info("Résultat enregistré dans : %s", output_path)
                on_success(result)
            except Exception as exc:
                self.logger.exception("Erreur lors de la finalisation Mask2Former")
                on_error(exc)
            finally:
                _cleanup()

        manager.execute(seg_inp, metadata=None, completion_callback=_handle_result)
