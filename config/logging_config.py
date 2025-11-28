"""
Configuration du logging de l'application.
"""
import logging

def configure_logging(log_level: str = 'INFO', log_file: str = None) -> None:
    """
    Configure le logger racine.

    Args:
        log_level: Niveau de log (DEBUG, INFO, WARNING, ERROR)
        log_file: Chemin vers le fichier de log (optionnel)
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Créer un handler pour la console avec encodage UTF-8
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Créer un formateur
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    console_handler.setFormatter(formatter)

    handlers = [console_handler]

    # Ajouter un handler pour fichier si spécifié
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # Configurer le logger racine
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=handlers,
        force=True  # Force la reconfiguration si déjà configuré
    )

