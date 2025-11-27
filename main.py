from PyQt6.QtWidgets import QApplication
import sys
import logging

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
