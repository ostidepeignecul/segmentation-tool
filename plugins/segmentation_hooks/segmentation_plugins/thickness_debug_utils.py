from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def log_thickness_details_to_file(
    mask: np.ndarray,
    volume_data: np.ndarray,
    labels: dict[str, int],
    ascan_peaks: dict[str, dict[tuple[int, int], int]],
    thickness_mask: np.ndarray | None = None,
    output_dir: str = ".",
    log_endview: int = 0,
    log_endview_start: int = 0,
    log_endview_end: int = 10,
) -> None:
    """
    Écrit des logs EXHAUSTIFS sur l'extraction des A-scans et le calcul des distances.
    Crée deux fichiers:
    1. thickness_debug.txt: Logs structurés (JSON) pour l'intervalle [log_endview_start, log_endview_end)
    2. thickness_debug_endview{log_endview}.json: Toutes les données pour l'endview spécifiée

    Args:
        mask: Masque de segmentation (Z, X, Y)
        volume_data: Volume NDE brut (Z, X, Y)
        labels: Dict des labels {'frontwall': 1, 'backwall': 2, ...}
        ascan_peaks: Dict des pics extraits {class_name: {(z,x): y, ...}}
        thickness_mask: Masque optionnel des lignes de thickness
        output_dir: Répertoire de sortie pour les fichiers
        log_endview: Index Z de l'endview pour le JSON détaillé (défaut: 0)
        log_endview_start: Début de l'intervalle pour logs textuels (défaut: 0)
        log_endview_end: Fin de l'intervalle pour logs textuels (défaut: 10)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_file = output_path / f"thickness_debug_{timestamp}.txt"
    json_file = output_path / f"thickness_debug_endview{log_endview}_{timestamp}.json"

    # ================================================================
    # PARTIE 1: Logs structurés (un seul write)
    # ================================================================
    summary_data: dict[str, Any] = {
        "metadata": {
            "timestamp": timestamp,
            "mask_shape": list(mask.shape),
            "volume_shape": list(volume_data.shape),
            "labels": labels,
            "num_classes_with_peaks": len(ascan_peaks),
            "log_interval": {
                "start": log_endview_start,
                "end": log_endview_end,
            },
        },
        "class_statistics": [],
        "thickness_mask": None,
        "endview_analysis": [],
    }

    for class_name, peaks_dict in ascan_peaks.items():
        class_stats: dict[str, Any] = {
            "class_name": class_name,
            "class_value": labels.get(class_name, -1),
            "num_positions": len(peaks_dict),
        }

        if peaks_dict:
            y_values = np.array(list(peaks_dict.values()))
            class_stats["y_stats"] = {
                "min": int(y_values.min()),
                "max": int(y_values.max()),
                "mean": float(y_values.mean()),
                "median": float(np.median(y_values)),
                "std": float(y_values.std()),
            }

            z_coords = [z for z, _ in peaks_dict.keys()]
            unique_z = sorted(set(z_coords))
            class_stats["z_coverage"] = {
                "num_endviews": len(unique_z),
                "total_endviews": mask.shape[0],
                "min_z": int(min(unique_z)),
                "max_z": int(max(unique_z)),
            }

        summary_data["class_statistics"].append(class_stats)

    if thickness_mask is not None:
        thickness_pixels = int(np.sum(thickness_mask > 0))
        summary_data["thickness_mask"] = {
            "shape": list(thickness_mask.shape),
            "non_zero_pixels": thickness_pixels,
            "min_value": float(thickness_mask.min()),
            "max_value": float(thickness_mask.max()),
        }

    def _serialize_peak(sample: tuple[tuple[int, int], int]) -> dict[str, Any]:
        (zz, x), y = sample
        amplitude = float(volume_data[zz, x, y])
        return {
            "x": int(x),
            "y": int(y),
            "amplitude": amplitude,
        }

    max_endview = min(log_endview_end, mask.shape[0])
    for z in range(log_endview_start, max_endview):
        classes_for_z = []

        for class_name, peaks_dict in ascan_peaks.items():
            z_peaks = [item for item in peaks_dict.items() if item[0][0] == z]
            if not z_peaks:
                continue

            sorted_peaks = sorted(z_peaks, key=lambda item: item[0][1])
            num_positions = len(sorted_peaks)
            classes_for_z.append(
                {
                    "class_name": class_name,
                    "num_positions": num_positions,
                    "first_positions": [_serialize_peak(sample) for sample in sorted_peaks[:5]],
                    "last_positions": [_serialize_peak(sample) for sample in sorted_peaks[-5:]],
                    "omitted_positions": max(num_positions - 10, 0),
                }
            )

        if classes_for_z:
            summary_data["endview_analysis"].append(
                {
                    "z_index": z,
                    "classes": classes_for_z,
                }
            )

    with open(txt_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Logs textuels écrits dans: {txt_file}")

    # ================================================================
    # PARTIE 2: JSON complet pour l'endview spécifiée
    # ================================================================
    endview_data: dict[str, Any] = {
        "metadata": {
            "timestamp": timestamp,
            "mask_shape": list(mask.shape),
            "volume_shape": list(volume_data.shape),
            "labels": labels,
            "endview": log_endview,
        },
        "peaks_by_class": {},
        "amplitudes": {},
        "mask_coverage": {},
        "thickness_lines": {},
    }

    # Extraire toutes les données pour Z=log_endview
    z = log_endview

    for class_name, peaks_dict in ascan_peaks.items():
        class_value = labels.get(class_name, -1)

        # Filtrer pour Z=log_endview
        z_peaks = {(zz, x): y for (zz, x), y in peaks_dict.items() if zz == z}

        if len(z_peaks) > 0:
            # Convertir en format JSON-friendly
            peaks_list = []
            amplitudes_list = []

            for (zz, x), y in sorted(z_peaks.items(), key=lambda item: item[0][1]):
                amp = float(volume_data[zz, x, y])

                peaks_list.append(
                    {
                        "x": int(x),
                        "y": int(y),
                        "amplitude": amp,
                    }
                )

                # Extraire tout l'A-scan pour cette position
                ascan = volume_data[zz, x, :].astype(float).tolist()
                amplitudes_list.append(
                    {
                        "x": int(x),
                        "y_peak": int(y),
                        "ascan_full": ascan,
                        "ascan_length": len(ascan),
                    }
                )

            endview_data["peaks_by_class"][class_name] = {
                "class_value": class_value,
                "num_positions": len(z_peaks),
                "positions": peaks_list,
            }

            endview_data["amplitudes"][class_name] = amplitudes_list

        # Coverage du masque pour cette classe
        class_mask_z = mask[z] == class_value
        if class_mask_z.any():
            x_coords, y_coords = np.where(class_mask_z)
            endview_data["mask_coverage"][class_name] = {
                "x_min": int(x_coords.min()),
                "x_max": int(x_coords.max()),
                "y_min": int(y_coords.min()),
                "y_max": int(y_coords.max()),
                "num_pixels": int(class_mask_z.sum()),
            }

    # Thickness lines pour Z=log_endview
    if thickness_mask is not None:
        thickness_z = thickness_mask[z]
        if thickness_z.any():
            x_coords, y_coords = np.where(thickness_z > 0)
            thickness_lines_list = []

            # Grouper par X
            for x in sorted(set(x_coords)):
                y_line = y_coords[x_coords == x]
                thickness_lines_list.append(
                    {
                        "x": int(x),
                        "y_min": int(y_line.min()),
                        "y_max": int(y_line.max()),
                        "length": int(y_line.max() - y_line.min() + 1),
                    }
                )

            endview_data["thickness_lines"] = {
                "num_lines": len(thickness_lines_list),
                "total_pixels": int(thickness_z.sum()),
                "lines": thickness_lines_list,
            }

    # Écrire le JSON
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(endview_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Données JSON pour endview {log_endview} écrites dans: {json_file}")
    logger.info(f"  -> {len(endview_data['peaks_by_class'])} classes avec des pics")
    logger.info("  -> JSON contient les A-scans complets pour vérification")


def log_thickness_lines_and_missing_interfaces(
    volume_data: np.ndarray,
    frontwall_peaks: dict[tuple[int, int], int],
    backwall_peaks: dict[tuple[int, int], int],
    mask_shape: tuple[int, int, int],
    output_dir: str = ".",
    log_endview_start: int = 0,
    log_endview_end: int = 10,
) -> None:
    """
    Log toutes les lignes de thickness créées et les interfaces manquantes.

    Crée deux fichiers JSON:
    1. thickness_lines_complete_{timestamp}.json - Toutes les lignes créées avec détails
    2. missing_interfaces_{timestamp}.json - Positions où une interface manque

    Args:
        volume_data: Volume NDE brut (Z, X, Y)
        frontwall_peaks: Dict {(z, x): y_frontwall, ...}
        backwall_peaks: Dict {(z, x): y_backwall, ...}
        mask_shape: Shape du masque (Z, X, Y)
        output_dir: Répertoire de sortie
        log_endview_start: Début de l'intervalle d'endviews à logger (défaut: 0)
        log_endview_end: Fin de l'intervalle d'endviews à logger (défaut: 10)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    lines_file = output_path / f"thickness_lines_complete_{timestamp}.json"
    missing_file = output_path / f"missing_interfaces_{timestamp}.json"

    # Déterminer les endviews à logger (intervalle)
    total_endviews = mask_shape[0]
    endviews_to_log = range(log_endview_start, min(log_endview_end, total_endviews))

    # ================================================================
    # PARTIE 1: Logger toutes les lignes complètes
    # ================================================================
    lines_data = {
        "metadata": {
            "timestamp": timestamp,
            "mask_shape": list(mask_shape),
            "volume_shape": list(volume_data.shape),
            "log_endview_start": log_endview_start,
            "log_endview_end": log_endview_end,
            "num_endviews_logged": len(endviews_to_log),
            "total_endviews": total_endviews,
            "total_frontwall_peaks": len(frontwall_peaks),
            "total_backwall_peaks": len(backwall_peaks),
        },
        "statistics": {"lines_created": 0, "by_endview": {}},
        "lines": [],
    }

    for z in endviews_to_log:
        endview_lines = []

        for x in range(mask_shape[1]):
            key = (z, x)
            if key not in frontwall_peaks or key not in backwall_peaks:
                continue

            y_front = frontwall_peaks[key]
            y_back = backwall_peaks[key]

            y_min = min(y_front, y_back)
            y_max = max(y_front, y_back)
            distance = y_max - y_min

            # Extraire l'A-scan complet pour ce (z, x)
            ascan = volume_data[z, x, :].astype(float).tolist()

            line_info = {
                "z": int(z),
                "x": int(x),
                "y_front": int(y_front),
                "y_back": int(y_back),
                "y_min": int(y_min),
                "y_max": int(y_max),
                "length_pixels": int(distance + 1),
                "front_amplitude": float(volume_data[z, x, y_front]),
                "back_amplitude": float(volume_data[z, x, y_back]),
                "ascan_length": len(ascan),
                "ascan_values": ascan,
            }
            lines_data["lines"].append(line_info)
            endview_lines.append(line_info)

        if endview_lines:
            lines_data["statistics"]["by_endview"][f"z_{z}"] = {
                "num_lines": len(endview_lines),
                "mean_length": float(
                    np.mean([line["length_pixels"] for line in endview_lines])
                ),
                "max_length": int(
                    np.max([line["length_pixels"] for line in endview_lines])
                ),
                "min_length": int(
                    np.min([line["length_pixels"] for line in endview_lines])
                ),
            }
            lines_data["statistics"]["lines_created"] += len(endview_lines)

    with open(lines_file, "w", encoding="utf-8") as f:
        json.dump(lines_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Lignes de thickness complètes écrites dans: {lines_file}")
    logger.info(f"  -> {lines_data['statistics']['lines_created']} lignes au total")

    # ================================================================
    # PARTIE 2: Logger les interfaces manquantes
    # ================================================================
    missing_data = {
        "metadata": {
            "timestamp": timestamp,
            "mask_shape": list(mask_shape),
            "volume_shape": list(volume_data.shape),
            "log_endview_start": log_endview_start,
            "log_endview_end": log_endview_end,
        },
        "missing": [],
        "statistics": {
            "frontwall_only": 0,
            "backwall_only": 0,
            "total_missing": 0,
            "by_endview": {},
        },
    }

    for z in endviews_to_log:
        frontwall_only_count = 0
        backwall_only_count = 0

        for x in range(mask_shape[1]):
            key = (z, x)
            has_front = key in frontwall_peaks
            has_back = key in backwall_peaks

            if has_front and has_back:
                continue

            amp_front = float(volume_data[z, x, frontwall_peaks[key]]) if has_front else None
            amp_back = float(volume_data[z, x, backwall_peaks[key]]) if has_back else None

            if has_front and not has_back:
                missing_data["missing"].append(
                    {
                        "z": int(z),
                        "x": int(x),
                        "issue": "backwall_missing",
                        "reason": "Backwall peak not detected by nnUNet mask at this (z,x) position",
                        "frontwall": {
                            "y": int(frontwall_peaks[key]),
                            "amplitude": amp_front,
                        },
                        "backwall": None,
                    }
                )
                frontwall_only_count += 1
            elif has_back and not has_front:
                missing_data["missing"].append(
                    {
                        "z": int(z),
                        "x": int(x),
                        "issue": "frontwall_missing",
                        "reason": "Frontwall peak not detected by nnUNet mask at this (z,x) position",
                        "frontwall": None,
                        "backwall": {
                            "y": int(backwall_peaks[key]),
                            "amplitude": amp_back,
                        },
                    }
                )
                backwall_only_count += 1

        missing_data["statistics"]["by_endview"][f"z_{z}"] = {
            "frontwall_only": frontwall_only_count,
            "backwall_only": backwall_only_count,
            "total": frontwall_only_count + backwall_only_count,
        }

    missing_data["statistics"]["total_missing"] = len(missing_data["missing"])
    missing_data["statistics"]["frontwall_only"] = sum(
        1 for item in missing_data["missing"] if item["issue"] == "backwall_missing"
    )
    missing_data["statistics"]["backwall_only"] = sum(
        1 for item in missing_data["missing"] if item["issue"] == "frontwall_missing"
    )

    with open(missing_file, "w", encoding="utf-8") as f:
        json.dump(missing_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Interfaces manquantes écrites dans: {missing_file}")
    logger.info(
        "  -> %d positions incomplètes pour l'intervalle Z=[%d, %d)",
        missing_data["statistics"]["total_missing"],
        log_endview_start,
        log_endview_end,
    )
    logger.info(
        "  -> %d avec frontwall seulement",
        missing_data["statistics"]["frontwall_only"],
    )
    logger.info(
        "  -> %d avec backwall seulement",
        missing_data["statistics"]["backwall_only"],
    )
