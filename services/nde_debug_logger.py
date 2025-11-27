#!/usr/bin/env python3
"""
Module de logging détaillé pour le débogage du pipeline de chargement NDE.
Enregistre toutes les opérations dans nde_debug_log.txt.
"""

import os
import datetime
import numpy as np
from typing import Any, Dict, List, Optional, Tuple


class NDEDebugLogger:
    """
    Logger dédié pour tracer toutes les opérations lors du chargement d'un fichier NDE.
    """
    
    _instance = None
    _log_file_path = "nde_debug_log.txt"
    _enabled = True
    
    def __new__(cls):
        """Singleton pattern pour avoir un seul logger global."""
        if cls._instance is None:
            cls._instance = super(NDEDebugLogger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialise le logger (une seule fois grâce au singleton)."""
        if self._initialized:
            return
        self._initialized = True
        self._session_start = None
    
    @classmethod
    def enable(cls):
        """Active le logging."""
        cls._enabled = True
    
    @classmethod
    def disable(cls):
        """Désactive le logging."""
        cls._enabled = False
    
    def start_session(self, nde_file: str):
        """
        Démarre une nouvelle session de logging pour un fichier NDE.
        Efface le fichier de log précédent.
        """
        if not self._enabled:
            return
        
        self._session_start = datetime.datetime.now()
        
        with open(self._log_file_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("NDE DEBUG LOG - SESSION DE CHARGEMENT\n")
            f.write("="*80 + "\n")
            f.write(f"Fichier NDE: {nde_file}\n")
            f.write(f"Date/Heure: {self._session_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
    
    def log_section(self, title: str):
        """Écrit un titre de section dans le log."""
        if not self._enabled:
            return
        
        timestamp = self._get_timestamp()
        with open(self._log_file_path, 'a', encoding='utf-8') as f:
            f.write("\n" + "="*80 + "\n")
            f.write(f"[{timestamp}] {title}\n")
            f.write("="*80 + "\n")
    
    def log(self, message: str, indent: int = 0):
        """
        Écrit un message dans le log.
        
        Args:
            message: Message à écrire
            indent: Niveau d'indentation (0, 1, 2, ...)
        """
        if not self._enabled:
            return
        
        timestamp = self._get_timestamp()
        indent_str = "  " * indent
        
        with open(self._log_file_path, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {indent_str}{message}\n")
    
    def log_variable(self, name: str, value: Any, indent: int = 0):
        """
        Écrit une variable et sa valeur dans le log.
        Gère les types spéciaux (numpy arrays, dicts, etc.).
        """
        if not self._enabled:
            return
        
        timestamp = self._get_timestamp()
        indent_str = "  " * indent
        
        with open(self._log_file_path, 'a', encoding='utf-8') as f:
            if isinstance(value, np.ndarray):
                f.write(f"[{timestamp}] {indent_str}{name}:\n")
                f.write(f"[{timestamp}] {indent_str}  - Type: numpy.ndarray\n")
                f.write(f"[{timestamp}] {indent_str}  - Shape: {value.shape}\n")
                f.write(f"[{timestamp}] {indent_str}  - Dtype: {value.dtype}\n")
                f.write(f"[{timestamp}] {indent_str}  - Min: {np.min(value)}\n")
                f.write(f"[{timestamp}] {indent_str}  - Max: {np.max(value)}\n")
                f.write(f"[{timestamp}] {indent_str}  - Mean: {np.mean(value):.2f}\n")
            elif isinstance(value, dict):
                f.write(f"[{timestamp}] {indent_str}{name}:\n")
                for k, v in value.items():
                    if isinstance(v, (int, float, str, bool)):
                        f.write(f"[{timestamp}] {indent_str}  - {k}: {v}\n")
                    elif isinstance(v, np.ndarray):
                        f.write(f"[{timestamp}] {indent_str}  - {k}: array{v.shape}\n")
                    elif isinstance(v, (list, tuple)):
                        f.write(f"[{timestamp}] {indent_str}  - {k}: {type(v).__name__}{list(v)}\n")
                    else:
                        f.write(f"[{timestamp}] {indent_str}  - {k}: {type(v).__name__}\n")
            elif isinstance(value, (list, tuple)):
                f.write(f"[{timestamp}] {indent_str}{name}: {type(value).__name__} (len={len(value)})\n")
                if len(value) > 0 and len(value) <= 10:
                    f.write(f"[{timestamp}] {indent_str}  - Values: {list(value)}\n")
            else:
                f.write(f"[{timestamp}] {indent_str}{name}: {value}\n")
    
    def log_array_slice(self, name: str, array: np.ndarray, slice_idx: int, indent: int = 0):
        """
        Écrit des informations sur une slice d'un array 3D.
        """
        if not self._enabled:
            return
        
        timestamp = self._get_timestamp()
        indent_str = "  " * indent
        
        with open(self._log_file_path, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {indent_str}{name}[{slice_idx}]:\n")
            f.write(f"[{timestamp}] {indent_str}  - Shape: {array.shape}\n")
            f.write(f"[{timestamp}] {indent_str}  - Dtype: {array.dtype}\n")
            f.write(f"[{timestamp}] {indent_str}  - Min: {np.min(array)}\n")
            f.write(f"[{timestamp}] {indent_str}  - Max: {np.max(array)}\n")
    
    def log_transformation(self, operation: str, input_shape: Tuple, output_shape: Tuple, indent: int = 0):
        """
        Écrit une transformation appliquée à un array.
        """
        if not self._enabled:
            return
        
        timestamp = self._get_timestamp()
        indent_str = "  " * indent
        
        with open(self._log_file_path, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {indent_str}TRANSFORMATION: {operation}\n")
            f.write(f"[{timestamp}] {indent_str}  - Input shape:  {input_shape}\n")
            f.write(f"[{timestamp}] {indent_str}  - Output shape: {output_shape}\n")
    
    def log_coordinate_mapping(self, from_coords: Tuple, to_coords: Tuple, description: str, indent: int = 0):
        """
        Écrit un mapping de coordonnées.
        """
        if not self._enabled:
            return
        
        timestamp = self._get_timestamp()
        indent_str = "  " * indent
        
        with open(self._log_file_path, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {indent_str}COORDINATE MAPPING: {description}\n")
            f.write(f"[{timestamp}] {indent_str}  - From: {from_coords}\n")
            f.write(f"[{timestamp}] {indent_str}  - To:   {to_coords}\n")
    
    def log_ascan_extraction(self, x: int, y: int, volume_indices: Tuple, profile_length: int, 
                            value: float, indent: int = 0):
        """
        Écrit les détails d'une extraction A-scan.
        """
        if not self._enabled:
            return
        
        timestamp = self._get_timestamp()
        indent_str = "  " * indent
        
        with open(self._log_file_path, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {indent_str}A-SCAN EXTRACTION:\n")
            f.write(f"[{timestamp}] {indent_str}  - Image coords: (x={x}, y={y})\n")
            f.write(f"[{timestamp}] {indent_str}  - Volume indices: {volume_indices}\n")
            f.write(f"[{timestamp}] {indent_str}  - Profile length: {profile_length}\n")
            f.write(f"[{timestamp}] {indent_str}  - Extracted value: {value}\n")
    
    def _get_timestamp(self) -> str:
        """Retourne un timestamp relatif au début de la session."""
        if self._session_start is None:
            return "00:00.000"
        
        elapsed = datetime.datetime.now() - self._session_start
        total_seconds = elapsed.total_seconds()
        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60
        return f"{minutes:02d}:{seconds:06.3f}"
    
    def end_session(self):
        """Termine la session de logging."""
        if not self._enabled:
            return
        
        timestamp = self._get_timestamp()
        with open(self._log_file_path, 'a', encoding='utf-8') as f:
            f.write("\n" + "="*80 + "\n")
            f.write(f"[{timestamp}] FIN DE LA SESSION DE LOGGING\n")
            f.write("="*80 + "\n")


# Instance globale
nde_debug_logger = NDEDebugLogger()

