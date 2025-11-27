"""
Modèles pour l'architecture MVC de l'application GUI.

Architecture recommandée avec 3 modèles :
- NDEModel : Données brutes NDT (volume, C-scan, A-scan)
- AnnotationModel : Masques et annotations (polygones, seuils)
- ViewStateModel : État de l'interface utilisateur (slice, zoom, crosshair)
"""

from .nde_model import NDEModel
from .annotation_model import AnnotationModel
from .view_state_model import ViewStateModel

__all__ = [
    'NDEModel',
    'AnnotationModel',
    'ViewStateModel',
]
