#!/usr/bin/env python3
"""
Module de logging pour le debug du chargement et des interactions avec les fichiers NPZ.
Trace toutes les opérations liées aux masques NPZ : chargement, affichage, interactions 3D, slider.

Singleton pattern pour accès global depuis n'importe quel module.
"""

import time
import numpy as np
from typing import Any, Dict, Tuple, Optional


class NPZDebugLogger:
    """
    Logger singleton pour tracer les opérations NPZ.
    Écrit dans npz_debug_log.txt.
    """
    
    _instance = None
    _enabled = True  # Mettre à False pour désactiver complètement le logging
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._log_file = None
        self._start_time = None
        self._session_active = False
        self._first_occurrence_tracker = {}  # Pour éviter de logger les mêmes événements répétitifs
        self._initialized = True
    
    def start_session(self, npz_filename: str):
        """
        Démarre une nouvelle session de logging pour un fichier NPZ.
        
        Args:
            npz_filename: Nom du fichier NPZ chargé
        """
        if not self._enabled:
            return
        
        try:
            self._log_file = open("npz_debug_log.txt", "w", encoding="utf-8")
            self._start_time = time.time()
            self._session_active = True
            self._first_occurrence_tracker.clear()
            
            self.log_section(f"NPZ DEBUG SESSION: {npz_filename}")
            self.log_separator()
            
        except Exception as e:
            print(f"[NPZDebugLogger] Erreur lors du démarrage de session: {e}")
            self._enabled = False
    
    def end_session(self):
        """Termine la session de logging."""
        if not self._enabled or not self._session_active:
            return
        
        try:
            elapsed = time.time() - self._start_time
            self.log_separator()
            self.log_section(f"SESSION TERMINÉE - Durée totale: {elapsed:.2f}s")
            
            if self._log_file:
                self._log_file.close()
                self._log_file = None
            
            self._session_active = False
            
        except Exception as e:
            print(f"[NPZDebugLogger] Erreur lors de la fin de session: {e}")
    
    def log_section(self, title: str):
        """Log un titre de section."""
        if not self._enabled or not self._session_active:
            return
        
        timestamp = self._get_timestamp()
        self._write(f"\n{'='*80}\n")
        self._write(f"[{timestamp}] {title}\n")
        self._write(f"{'='*80}\n")
    
    def log_separator(self):
        """Log un séparateur visuel."""
        if not self._enabled or not self._session_active:
            return
        self._write(f"\n{'-'*80}\n")
    
    def log_npz_loading(self, npz_path: str, masks_shape: Tuple, num_slices: int):
        """
        Log le chargement d'un fichier NPZ.
        
        Args:
            npz_path: Chemin du fichier NPZ
            masks_shape: Shape du tableau de masques (num_slices, height, width)
            num_slices: Nombre de slices chargées
        """
        if not self._enabled or not self._session_active:
            return
        
        self.log_section("CHARGEMENT NPZ")
        self.log_variable("npz_path", npz_path)
        self.log_variable("masks_shape", masks_shape)
        self.log_variable("num_slices", num_slices)
        self.log_variable("dtype", "uint8 (classes 0-9)")
    
    def log_3d_view_initialization(self, volume_shape: Tuple, orientation: str, 
                                   transpose: bool, rotation_applied: bool):
        """
        Log l'initialisation de la vue 3D vispy.
        
        Args:
            volume_shape: Shape du volume 3D
            orientation: Orientation du slicing
            transpose: Si transpose appliqué
            rotation_applied: Si rotation appliquée
        """
        if not self._enabled or not self._session_active:
            return
        
        self.log_section("INITIALISATION VUE 3D VISPY")
        self.log_variable("volume_shape", volume_shape)
        self.log_variable("orientation", orientation)
        self.log_variable("transpose", transpose)
        self.log_variable("rotation_applied", rotation_applied)
    
    def log_volume_transformation(self, operation: str, input_shape: Tuple, 
                                 output_shape: Tuple, params: Optional[Dict] = None):
        """
        Log une transformation appliquée au volume 3D.
        
        Args:
            operation: Nom de l'opération (transpose, flip, rotation, colormap)
            input_shape: Shape avant transformation
            output_shape: Shape après transformation
            params: Paramètres additionnels de la transformation
        """
        if not self._enabled or not self._session_active:
            return
        
        # Éviter de logger les mêmes transformations répétitives
        key = f"transform_{operation}_{input_shape}_{output_shape}"
        if key in self._first_occurrence_tracker:
            return
        self._first_occurrence_tracker[key] = True
        
        self.log_subsection(f"TRANSFORMATION: {operation}")
        self.log_variable("input_shape", input_shape, indent=1)
        self.log_variable("output_shape", output_shape, indent=1)
        if params:
            for param_name, param_value in params.items():
                self.log_variable(param_name, param_value, indent=1)
    
    def log_endview_update(self, slice_idx: int, endview_shape: Tuple, 
                          num_annotations: int, classes_present: list):
        """
        Log la mise à jour d'une endview (changement de slice via slider).
        
        Args:
            slice_idx: Index du slice affiché
            endview_shape: Shape de l'endview (height, width)
            num_annotations: Nombre de pixels annotés
            classes_present: Liste des classes présentes dans ce slice
        """
        if not self._enabled or not self._session_active:
            return
        
        # AUCUNE LIMITE - Logger tous les événements
        self.log_subsection(f"ENDVIEW UPDATE - Slice {slice_idx}")
        self.log_variable("slice_idx", slice_idx, indent=1)
        self.log_variable("endview_shape", endview_shape, indent=1)
        self.log_variable("num_annotations", num_annotations, indent=1)
        self.log_variable("classes_present", classes_present, indent=1)
    
    def log_3d_interaction(self, interaction_type: str, params: Dict):
        """
        Log une interaction 3D (rotation, zoom, pan).

        Args:
            interaction_type: Type d'interaction (mouse_drag, mouse_wheel, camera_update)
            params: Paramètres de l'interaction (position, delta, camera state)
        """
        if not self._enabled or not self._session_active:
            return

        # AUCUNE LIMITE - Logger tous les événements
        self.log_subsection(f"INTERACTION 3D: {interaction_type}")
        for param_name, param_value in params.items():
            self.log_variable(param_name, param_value, indent=1)
    
    def log_colormap_application(self, num_classes: int, colormap_size: int):
        """
        Log l'application du colormap au volume.

        Args:
            num_classes: Nombre de classes dans le volume
            colormap_size: Taille du colormap (10 pour classes 0-9)
        """
        if not self._enabled or not self._session_active:
            return

        # AUCUNE LIMITE - Logger tous les événements
        self.log_subsection("APPLICATION COLORMAP")
        self.log_variable("num_classes", num_classes, indent=1)
        self.log_variable("colormap_size", colormap_size, indent=1)
    
    def log_subsection(self, title: str):
        """Log un sous-titre."""
        if not self._enabled or not self._session_active:
            return
        
        timestamp = self._get_timestamp()
        self._write(f"\n[{timestamp}] {title}\n")
    
    def log_variable(self, name: str, value: Any, indent: int = 0):
        """
        Log une variable avec formatage intelligent.
        
        Args:
            name: Nom de la variable
            value: Valeur de la variable
            indent: Niveau d'indentation (0, 1, 2)
        """
        if not self._enabled or not self._session_active:
            return
        
        indent_str = "  " * indent
        
        # Formatage selon le type
        if isinstance(value, np.ndarray):
            value_str = f"array(shape={value.shape}, dtype={value.dtype})"
        elif isinstance(value, (list, tuple)) and len(value) > 10:
            value_str = f"{type(value).__name__}(len={len(value)}, first={value[:3]}...)"
        elif isinstance(value, dict):
            # Afficher les valeurs complètes du dictionnaire
            dict_items = ", ".join([f"{k}={v}" for k, v in value.items()])
            value_str = f"{{{dict_items}}}"
        else:
            value_str = str(value)
        
        self._write(f"{indent_str}{name}: {value_str}\n")
    
    def log_error(self, error_msg: str, exception: Optional[Exception] = None):
        """
        Log une erreur.

        Args:
            error_msg: Message d'erreur
            exception: Exception optionnelle
        """
        if not self._enabled or not self._session_active:
            return

        timestamp = self._get_timestamp()
        self._write(f"\n[{timestamp}] ❌ ERROR: {error_msg}\n")
        if exception:
            self._write(f"  Exception: {type(exception).__name__}: {str(exception)}\n")

    def log_nde_volume_loading(self, volume_shape: Tuple, orientation: str,
                               transpose: bool, rotation_applied: bool):
        """Log le chargement du volume NDE principal dans la vue 3D."""
        if not self._enabled or not self._session_active:
            return

        # AUCUNE LIMITE - Logger tous les événements
        self.log_subsection("CHARGEMENT VOLUME NDE PRINCIPAL")
        self.log_variable("volume_shape", volume_shape, indent=1)
        self.log_variable("orientation", orientation, indent=1)
        self.log_variable("transpose", transpose, indent=1)
        self.log_variable("rotation_applied", rotation_applied, indent=1)

    def log_slice_plane_creation(self, slice_idx: int, plane_position: Dict,
                                 plane_dimensions: Dict, rendering_params: Dict):
        """Log la création/mise à jour du plan de coupe jaune."""
        if not self._enabled or not self._session_active:
            return

        # AUCUNE LIMITE - Logger tous les événements
        self.log_subsection(f"CRÉATION PLAN DE COUPE (SLICE {slice_idx})")
        self.log_variable("slice_idx", slice_idx, indent=1)
        self.log_variable("plane_position", plane_position, indent=1)
        self.log_variable("plane_dimensions", plane_dimensions, indent=1)
        self.log_variable("rendering_params", rendering_params, indent=1)

    def log_rendering_order(self, objects: list):
        """Log l'ordre de rendu des objets dans la scène 3D."""
        if not self._enabled or not self._session_active:
            return

        # AUCUNE LIMITE - Logger tous les événements
        self.log_subsection("ORDRE DE RENDU SCÈNE 3D")
        for i, obj in enumerate(objects):
            self.log_variable(f"objet_{i+1}", obj, indent=1)

    def log_visual_properties(self, visual_name: str, properties: Dict):
        """Log les propriétés d'un objet visuel (volume, overlay, slice plane)."""
        if not self._enabled or not self._session_active:
            return

        # AUCUNE LIMITE - Logger tous les événements
        self.log_subsection(f"PROPRIÉTÉS VISUELLES: {visual_name}")
        for prop_key, prop_value in properties.items():
            self.log_variable(prop_key, prop_value, indent=1)

    def _get_timestamp(self) -> str:
        """Retourne le timestamp relatif au début de session."""
        if self._start_time is None:
            return "0.000s"
        elapsed = time.time() - self._start_time
        return f"{elapsed:.3f}s"
    
    def _write(self, text: str):
        """Écrit dans le fichier de log."""
        if self._log_file and not self._log_file.closed:
            self._log_file.write(text)
            self._log_file.flush()  # Flush immédiat pour voir les logs en temps réel


# Instance globale singleton
npz_debug_logger = NPZDebugLogger()

