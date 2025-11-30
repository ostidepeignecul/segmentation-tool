from PyQt6.QtWidgets import QApplication
import os
import sys
import logging

# Prevent duplicate OpenMP runtime crashes (libiomp5md.dll) when nnunet/torch/numpy are imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from controllers.master_controller import MasterController

if __name__ == "__main__":
    # Configuration du logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    
    app = QApplication(sys.argv)
    
    # Créer le contrôleur principal
    master_controller = MasterController()
    
    # Démarrer l'application
    master_controller.run()
    
    sys.exit(app.exec())
