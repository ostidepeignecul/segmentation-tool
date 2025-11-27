#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Service pour séparer les images et masques en flaw/noflaw.
Combine la logique de view_labels.py et split_flaw_noflaw.py.
"""

import os
import re
import shutil
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Set, Tuple
import logging
from services.endview_export import export_endviews_gui


class SplitFlawNoflawService:
    """Service pour analyser les labels et séparer flaw/noflaw."""
    
    LINE_RE = re.compile(r'^(?P<name>[^:]+):\s*\[(?P<classes>[^\]]*)\]\s*$')
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_labels(self, masks_dir: Path) -> Tuple[bool, str, Path]:
        """
        Analyse les masques et génère un fichier de résumé des labels.
        
        Args:
            masks_dir: Dossier contenant les masques binaires
            
        Returns:
            Tuple[bool, str, Path]: (succès, message, chemin du fichier de résumé)
        """
        try:
            # Lister tous les fichiers .png dans le dossier
            mask_files = list(masks_dir.glob("*.png"))
            
            if not mask_files:
                return False, "❌ Aucun fichier .png trouvé dans masks_binary", None
            
            total_files = len(mask_files)
            self.logger.info(f"📁 {total_files} fichiers trouvés dans {masks_dir}")
            
            # Créer le fichier de sortie avec timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"view_labels_results_{timestamp}.txt"
            output_path = masks_dir / output_filename
            
            self.logger.info(f"💾 Sauvegarde des résultats dans: {output_path}")
            
            successful_analyses = 0
            failed_analyses = 0
            
            # Ouvrir le fichier de sortie
            with open(output_path, 'w', encoding='utf-8') as output_file:
                # Écrire l'en-tête
                output_file.write(f"Analyse des labels - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                output_file.write(f"Dossier analysé: {masks_dir}\n")
                output_file.write(f"Nombre total de fichiers: {total_files}\n")
                output_file.write("=" * 70 + "\n\n")
                
                # Analyser chaque fichier
                for i, mask_file in enumerate(mask_files, 1):
                    try:
                        # Charger l'image
                        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                        if mask is None:
                            error_msg = f"❌ Impossible de charger l'image: {mask_file.name}"
                            output_file.write(f"{error_msg}\n")
                            failed_analyses += 1
                            continue
                        
                        # Trouver les valeurs uniques
                        unique_values = np.unique(mask)
                        result_msg = f"{mask_file.name}: {unique_values.tolist()}"
                        
                        # Écrire dans le fichier
                        output_file.write(f"{result_msg}\n")
                        successful_analyses += 1
                        
                    except Exception as e:
                        error_msg = f"❌ Erreur lors de l'analyse de {mask_file.name}: {str(e)}"
                        output_file.write(f"{error_msg}\n")
                        failed_analyses += 1
                    
                    # Afficher un résumé périodique
                    if i % 1000 == 0:
                        self.logger.info(f"📊 [{i:6d}/{total_files:6d}] ({i/total_files*100:5.1f}%)")
                        output_file.flush()
            
            message = f"✅ Analyse terminée: {successful_analyses} réussis, {failed_analyses} échecs\n{output_path}"
            self.logger.info(message)
            return True, message, output_path
            
        except Exception as e:
            error_msg = f"Erreur lors de l'analyse des labels: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg, None
    
    def parse_label_file(self, txt_path: Path) -> Dict[str, Set[int]]:
        """
        Parse le fichier de résumé des labels.
        
        Args:
            txt_path: Chemin vers le fichier view_labels_results*.txt
            
        Returns:
            Dict[str, Set[int]]: Mapping nom_fichier -> ensemble de classes
        """
        mapping = {}
        with txt_path.open('r', encoding='utf-8', errors='ignore') as f:
            for raw in f:
                line = raw.strip()
                m = self.LINE_RE.match(line)
                if not m:
                    continue
                name = m.group('name').strip()
                classes_str = m.group('classes').strip()
                cls_set = set()
                if classes_str:
                    for tok in classes_str.split(','):
                        tok = tok.strip()
                        if tok.lstrip('-').isdigit():
                            cls_set.add(int(tok))
                mapping[name] = cls_set
        return mapping
    
    def decide_bucket(self, cls_set: Set[int]) -> str:
        """
        Décide si un masque appartient à flaw ou noflaw.
        
        Args:
            cls_set: Ensemble des classes présentes dans le masque
            
        Returns:
            str: "flaw" ou "noflaw"
        """
        if cls_set == {0}:
            return "noflaw"
        if 1 in cls_set or len(cls_set) >= 2:
            return "flaw"
        return "noflaw"
    
    def safe_copy(self, src: Path, dst_dir: Path) -> bool:
        """
        Copie un fichier de manière sécurisée.
        
        Args:
            src: Fichier source
            dst_dir: Dossier de destination
            
        Returns:
            bool: True si la copie a réussi
        """
        if src.is_file():
            shutil.copy2(src, dst_dir / src.name)
            return True
        else:
            self.logger.warning(f"[WARN] fichier manquant: {src}")
            return False
    
    def split_flaw_noflaw(
        self,
        base_dir: Path,
        masks_dir: Path,
        complete_rgb_dir: Path,
        complete_uint8_dir: Path,
        labels_file: Path = None
    ) -> Tuple[bool, str]:
        """
        Sépare les images et masques en flaw/noflaw selon la structure demandée.
        
        Args:
            base_dir: Dossier racine (où se trouve le .nde)
            masks_dir: Dossier masks_binary
            complete_rgb_dir: Dossier endviews_rgb24/complete
            complete_uint8_dir: Dossier endviews_uint8/complete
            labels_file: Fichier de labels (si None, cherche le plus récent)
            
        Returns:
            Tuple[bool, str]: (succès, message)
        """
        try:
            # Vérifications
            if not masks_dir.exists():
                return False, f"❌ Le dossier masks_binary n'existe pas: {masks_dir}"
            
            # Chercher le fichier de labels
            if labels_file is None:
                txt_files = list(masks_dir.glob("view_labels_results*.txt"))
                if not txt_files:
                    return False, f"❌ Aucun fichier view_labels_results*.txt trouvé dans: {masks_dir}"
                labels_file = max(txt_files, key=lambda p: p.stat().st_mtime)
            
            self.logger.info(f"📄 Fichier de labels: {labels_file}")
            
            # Créer les dossiers de sortie
            rgb_flaw = base_dir / "endviews_rgb24" / "flaw"
            rgb_noflaw = base_dir / "endviews_rgb24" / "noflaw"
            rgb_gtmask_flaw = base_dir / "endviews_rgb24" / "gtmask" / "flaw"
            rgb_gtmask_noflaw = base_dir / "endviews_rgb24" / "gtmask" / "noflaw"
            
            uint8_flaw = base_dir / "endviews_uint8" / "flaw"
            uint8_noflaw = base_dir / "endviews_uint8" / "noflaw"
            uint8_gtmask_flaw = base_dir / "endviews_uint8" / "gtmask" / "flaw"
            uint8_gtmask_noflaw = base_dir / "endviews_uint8" / "gtmask" / "noflaw"
            
            for d in [rgb_flaw, rgb_noflaw, rgb_gtmask_flaw, rgb_gtmask_noflaw,
                      uint8_flaw, uint8_noflaw, uint8_gtmask_flaw, uint8_gtmask_noflaw]:
                d.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"📁 Créé: {d}")
            
            # Parser le fichier de labels
            self.logger.info("🔍 Analyse du fichier de labels...")
            mapping = self.parse_label_file(labels_file)
            
            stats = {
                "flaw_masks": 0,
                "noflaw_masks": 0,
                "flaw_rgb_images": 0,
                "noflaw_rgb_images": 0,
                "flaw_uint8_images": 0,
                "noflaw_uint8_images": 0
            }
            
            total_files = len(mapping)
            self.logger.info(f"📊 {total_files} fichiers à traiter")
            
            processed = 0
            for fname, cls_set in mapping.items():
                bucket = self.decide_bucket(cls_set)
                
                # Chemins des fichiers
                mask_path = masks_dir / fname
                rgb_path = complete_rgb_dir / fname if complete_rgb_dir.exists() else None
                uint8_path = complete_uint8_dir / fname if complete_uint8_dir.exists() else None
                
                if bucket == "flaw":
                    # Copier les masques
                    if self.safe_copy(mask_path, rgb_gtmask_flaw):
                        stats["flaw_masks"] += 1
                    if self.safe_copy(mask_path, uint8_gtmask_flaw):
                        pass  # Déjà compté
                    
                    # Copier les images RGB
                    if rgb_path and self.safe_copy(rgb_path, rgb_flaw):
                        stats["flaw_rgb_images"] += 1
                    
                    # Copier les images UINT8
                    if uint8_path and self.safe_copy(uint8_path, uint8_flaw):
                        stats["flaw_uint8_images"] += 1
                else:
                    # Copier les masques
                    if self.safe_copy(mask_path, rgb_gtmask_noflaw):
                        stats["noflaw_masks"] += 1
                    if self.safe_copy(mask_path, uint8_gtmask_noflaw):
                        pass  # Déjà compté
                    
                    # Copier les images RGB
                    if rgb_path and self.safe_copy(rgb_path, rgb_noflaw):
                        stats["noflaw_rgb_images"] += 1
                    
                    # Copier les images UINT8
                    if uint8_path and self.safe_copy(uint8_path, uint8_noflaw):
                        stats["noflaw_uint8_images"] += 1
                
                processed += 1
                if processed % 100 == 0:
                    self.logger.info(f"⏳ Traité: {processed}/{total_files}")
            
            # Résumé
            summary = "\n=== Résumé ===\n"
            for k, v in stats.items():
                summary += f"{k}: {v}\n"
            
            self.logger.info(summary)
            
            message = f"✅ Traitement terminé!\n{summary}"
            return True, message
            
        except Exception as e:
            error_msg = f"Erreur lors du split flaw/noflaw: {str(e)}"
            self.logger.error(error_msg)
            import traceback
            traceback.print_exc()
            return False, error_msg


def split_flaw_noflaw_gui(
    base_dir: str,
    masks_dir: str,
    complete_rgb_dir: str = None,
    complete_uint8_dir: str = None,
    nde_file: str = None,
    nde_loader_service=None
) -> Tuple[bool, str]:
    """
    Fonction wrapper pour l'interface graphique.
    Exporte automatiquement les endviews avant de faire le split.

    Args:
        base_dir: Dossier racine (où se trouve le .nde)
        masks_dir: Dossier masks_binary
        complete_rgb_dir: Dossier endviews_rgb24/complete (optionnel)
        complete_uint8_dir: Dossier endviews_uint8/complete (optionnel)
        nde_file: Chemin vers le fichier NDE (pour export automatique)
        nde_loader_service: Service NDE loader (pour export automatique)

    Returns:
        Tuple[bool, str]: (succès, message)
    """
    service = SplitFlawNoflawService()

    base_path = Path(base_dir)
    masks_path = Path(masks_dir)

    # Déterminer les chemins des dossiers complete
    if complete_rgb_dir is None:
        complete_rgb_dir = base_path / "endviews_rgb24" / "complete"
    else:
        complete_rgb_dir = Path(complete_rgb_dir)

    if complete_uint8_dir is None:
        complete_uint8_dir = base_path / "endviews_uint8" / "complete"
    else:
        complete_uint8_dir = Path(complete_uint8_dir)

    # ÉTAPE 0: Exporter les endviews automatiquement si NDE fourni
    if nde_file and nde_loader_service:
        service.logger.info("=== ÉTAPE 0: Export automatique des endviews ===")

        # Vérifier si les endviews sont déjà présentes
        rgb_has_images = complete_rgb_dir.exists() and len(list(complete_rgb_dir.glob("*.png"))) > 0
        uint8_has_images = complete_uint8_dir.exists() and len(list(complete_uint8_dir.glob("*.png"))) > 0

        # Export RGB24
        if rgb_has_images:
            num_rgb_images = len(list(complete_rgb_dir.glob("*.png")))
            service.logger.info(f"⏭️  Skip export RGB24: {num_rgb_images} images déjà présentes dans {complete_rgb_dir}")
        else:
            service.logger.info("Export RGB24...")
            success_rgb, message_rgb = export_endviews_gui(
                nde_file=nde_file,
                output_folder=str(complete_rgb_dir),
                group_idx=1,
                nde_loader_service=nde_loader_service,
                export_format='rgb',
                flip_horizontal=False,
                flip_vertical=False,
                rotation_angle=0,
                custom_transpose=False
            )

            if not success_rgb:
                return False, f"Échec de l'export RGB24:\n{message_rgb}"

            service.logger.info(f"✅ {message_rgb}")

        # Export UINT8
        if uint8_has_images:
            num_uint8_images = len(list(complete_uint8_dir.glob("*.png")))
            service.logger.info(f"⏭️  Skip export UINT8: {num_uint8_images} images déjà présentes dans {complete_uint8_dir}")
        else:
            service.logger.info("Export UINT8...")
            success_uint8, message_uint8 = export_endviews_gui(
                nde_file=nde_file,
                output_folder=str(complete_uint8_dir),
                group_idx=1,
                nde_loader_service=nde_loader_service,
                export_format='uint8',
                flip_horizontal=False,
                flip_vertical=False,
                rotation_angle=0,
                custom_transpose=False
            )

            if not success_uint8:
                return False, f"Échec de l'export UINT8:\n{message_uint8}"

            service.logger.info(f"✅ {message_uint8}")
    else:
        service.logger.info("=== ÉTAPE 0: Vérification des dossiers complete ===")
        if not complete_rgb_dir.exists() and not complete_uint8_dir.exists():
            return False, (
                "Aucun dossier 'complete' trouvé et aucun fichier NDE fourni pour l'export automatique.\n\n"
                f"Dossiers manquants:\n"
                f"- {complete_rgb_dir}\n"
                f"- {complete_uint8_dir}\n\n"
                f"Veuillez d'abord exporter les endviews."
            )

    # Étape 1: Analyser les labels
    service.logger.info("=== ÉTAPE 1: Analyse des labels ===")
    success, message, labels_file = service.analyze_labels(masks_path)

    if not success:
        return False, message

    # Étape 2: Séparer flaw/noflaw
    service.logger.info("=== ÉTAPE 2: Séparation flaw/noflaw ===")
    success, message = service.split_flaw_noflaw(
        base_path,
        masks_path,
        complete_rgb_dir,
        complete_uint8_dir,
        labels_file
    )

    return success, message


def split_from_npz_file(
    nde_file: str,
    npz_file: str,
    nde_loader_service=None
) -> Tuple[bool, str]:
    """
    Nouvelle fonction pour split flaw/noflaw en utilisant directement un fichier NPZ.
    
    Workflow:
    1. Charger le NPZ fourni
    2. Créer un dossier temporaire
    3. Exporter le NPZ en PNG dans ce dossier temp
    4. Générer le fichier d'analyse (view_labels)
    5. Exporter les endviews (RGB24 + UINT8)
    6. Faire le split flaw/noflaw
    7. Nettoyer le dossier temporaire
    
    Args:
        nde_file: Chemin vers le fichier NDE
        npz_file: Chemin vers le fichier NPZ à utiliser (le plus récent)
        nde_loader_service: Service NDE loader
        
    Returns:
        Tuple[bool, str]: (succès, message)
    """
    import shutil
    import tempfile
    
    service = SplitFlawNoflawService()
    
    try:
        base_dir = Path(os.path.dirname(nde_file))
        
        # Vérifier que le fichier NPZ existe
        if not os.path.exists(npz_file):
            return False, f"Le fichier NPZ n'existe pas: {npz_file}"
        
        service.logger.info("=== NOUVELLE FONCTION: Split depuis NPZ ===")
        service.logger.info(f"NDE: {nde_file}")
        service.logger.info(f"NPZ: {npz_file}")
        
        # Créer un dossier temporaire
        temp_dir = base_dir / "temp_masks_split"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        service.logger.info(f"📁 Dossier temporaire créé: {temp_dir}")
        
        try:
            # Charger le NPZ ou NPY
            service.logger.info("=== ÉTAPE 1: Chargement du fichier NumPy ===")
            file_ext = Path(npz_file).suffix.lower()
            
            if file_ext == '.npy':
                # Format NPY : charge directement un array
                service.logger.info(f"Chargement d'un fichier NPY: {npz_file}")
                volume = np.load(npz_file)
            elif file_ext == '.npz':
                # Format NPZ : charge un dictionnaire avec des clés
                service.logger.info(f"Chargement d'un fichier NPZ: {npz_file}")
                npz_data = np.load(npz_file)
                
                # Trouver la clé du volume (généralement 'arr_0')
                if 'arr_0' in npz_data:
                    volume = npz_data['arr_0']
                else:
                    # Prendre la première clé disponible
                    keys = list(npz_data.keys())
                    if not keys:
                        return False, "Le fichier NPZ ne contient aucune donnée"
                    volume = npz_data[keys[0]]
            else:
                return False, f"Format non supporté: {file_ext}. Formats acceptés: .npz, .npy"
            
            service.logger.info(f"Volume chargé: shape={volume.shape}, dtype={volume.dtype}")
            
            # Le volume est en format ZXY (après transformations), il faut inverser
            # pour obtenir le format original (Z, H, W)
            service.logger.info("=== ÉTAPE 2: Inversion des transformations ===")
            
            # Inverser transpose ZXY: (Z, X, Y) -> (Z, Y, X)
            volume_transposed = volume.transpose((0, 2, 1))
            
            # Inverser flip horizontal
            volume_restored = np.array([np.fliplr(slice_img) for slice_img in volume_transposed])
            
            service.logger.info(f"Volume restauré: shape={volume_restored.shape}")
            
            # Exporter chaque slice en PNG dans le dossier temporaire
            service.logger.info("=== ÉTAPE 3: Export des masques en PNG (temporaire) ===")
            num_slices = volume_restored.shape[0]
            
            for i in range(num_slices):
                mask = volume_restored[i]
                
                # Utiliser le même format de nommage que les endviews
                position_filename = i * 1500
                filename = f"endview_{position_filename:012d}.png"
                png_path = temp_dir / filename
                
                cv2.imwrite(str(png_path), mask.astype(np.uint8))
                
                if (i + 1) % 100 == 0:
                    service.logger.info(f"  Exporté {i + 1}/{num_slices} masques...")
            
            service.logger.info(f"✅ {num_slices} masques exportés dans {temp_dir}")
            
            # Analyser les labels
            service.logger.info("=== ÉTAPE 4: Analyse des labels ===")
            success, message, labels_file = service.analyze_labels(temp_dir)
            
            if not success:
                return False, f"Erreur lors de l'analyse des labels:\n{message}"
            
            service.logger.info(f"✅ Fichier d'analyse créé: {labels_file}")
            
            # Définir les dossiers complete
            complete_rgb_dir = base_dir / "endviews_rgb24" / "complete"
            complete_uint8_dir = base_dir / "endviews_uint8" / "complete"
            
            # Exporter les endviews (RGB24 + UINT8)
            service.logger.info("=== ÉTAPE 5: Export des endviews ===")
            
            # Vérifier si les endviews sont déjà présentes
            rgb_has_images = complete_rgb_dir.exists() and len(list(complete_rgb_dir.glob("*.png"))) > 0
            uint8_has_images = complete_uint8_dir.exists() and len(list(complete_uint8_dir.glob("*.png"))) > 0
            
            # Export RGB24
            if rgb_has_images:
                num_rgb_images = len(list(complete_rgb_dir.glob("*.png")))
                service.logger.info(f"⏭️  Skip export RGB24: {num_rgb_images} images déjà présentes")
            else:
                service.logger.info("Export RGB24...")
                success_rgb, message_rgb = export_endviews_gui(
                    nde_file=nde_file,
                    output_folder=str(complete_rgb_dir),
                    group_idx=1,
                    nde_loader_service=nde_loader_service,
                    export_format='rgb',
                    flip_horizontal=False,
                    flip_vertical=False,
                    rotation_angle=0,
                    custom_transpose=False
                )
                
                if not success_rgb:
                    return False, f"Échec de l'export RGB24:\n{message_rgb}"
                
                service.logger.info(f"✅ {message_rgb}")
            
            # Export UINT8
            if uint8_has_images:
                num_uint8_images = len(list(complete_uint8_dir.glob("*.png")))
                service.logger.info(f"⏭️  Skip export UINT8: {num_uint8_images} images déjà présentes")
            else:
                service.logger.info("Export UINT8...")
                success_uint8, message_uint8 = export_endviews_gui(
                    nde_file=nde_file,
                    output_folder=str(complete_uint8_dir),
                    group_idx=1,
                    nde_loader_service=nde_loader_service,
                    export_format='uint8',
                    flip_horizontal=False,
                    flip_vertical=False,
                    rotation_angle=0,
                    custom_transpose=False
                )
                
                if not success_uint8:
                    return False, f"Échec de l'export UINT8:\n{message_uint8}"
                
                service.logger.info(f"✅ {message_uint8}")
            
            # Faire le split flaw/noflaw
            service.logger.info("=== ÉTAPE 6: Séparation flaw/noflaw ===")
            success, message = service.split_flaw_noflaw(
                base_dir,
                temp_dir,
                complete_rgb_dir,
                complete_uint8_dir,
                labels_file
            )
            
            if not success:
                return False, message
            
            service.logger.info("✅ Split flaw/noflaw terminé avec succès")
            
            return True, message
            
        finally:
            # Nettoyer le dossier temporaire
            service.logger.info("=== ÉTAPE 7: Nettoyage ===")
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                service.logger.info(f"🧹 Dossier temporaire supprimé: {temp_dir}")
    
    except Exception as e:
        error_msg = f"Erreur lors du split depuis NPZ: {str(e)}"
        service.logger.error(error_msg)
        import traceback
        traceback.print_exc()
        return False, error_msg
