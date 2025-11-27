# 2025-11-26 — Align model/controller skeleton imports

**Tags :** `#models/nde_model.py`, `#models/__init__.py`, `#controllers/master_controller.py`, `#mvc`

**Actions effectuées :**
- Renamed model class to `NDEModel` and updated controller import to match skeleton naming.
- Simplified `models/__init__.py` exports to only existing classes (NDEModel, AnnotationModel, ViewStateModel) removing stale Polygon/ToolType imports.
- Allowed `MasterController` to create its own `QMainWindow` by default, added `run()` to show the window, keeping connections to UI actions/signals.

**Contexte :**
ImportError occurred because `models/__init__.py` expected `NDEModel` and other symbols missing after skeleton creation. main.py constructs `MasterController()` with no args and calls `run()`, so controller now handles that flow without business logic.

**Décisions techniques :**
1. Kept class name as `NDEModel` (acronym uppercase) to align with existing module exports and docs.
2. Removed unused exports to avoid import failures when loading models package; maintained skeleton-only behavior.

---

### **2025-11-27** — ToolsPanel signals and controller wiring

**Tags :** `#views/tools_panel.py`, `#controllers/master_controller.py`, `#views/endview_view.py`, `#views/cscan_view.py`, `#views/ascan_view.py`, `#views/volume_view.py`, `#models/view_state_model.py`, `#mvc`, `#signals-and-slots`

**Actions effectuées :**
- Exposé tous les signaux du ToolsPanel (slice, goto, modes, threshold, toggles, ROI actions) et ajouté `attach_designer_widgets` pour câbler les contrôles Qt Designer sans modifier `ui_mainwindow.py`.
- Ajouté des signaux d’interaction aux vues Endview/CScan/AScan/Volume (polygone, crosshair, volume update, caméra) en maintenant une logique de vue minimale.
- Mis à jour `MasterController` pour connecter chaque signal aux slots dédiés, synchroniser l’état UI (slice, tool mode, threshold, toggles) et laisser les handlers métiers en placeholders.
- Étendu `ViewStateModel` avec des setters concrets (slice, alpha, tool_mode, threshold, toggles overlay/volume/ROI) pour conserver l’état UI.

**Contexte :**
Respect strict du MVC : le ToolsPanel reste une vue et ne transporte aucune logique métier. Le contrôleur injecte les widgets générés par Designer dans la vue via une méthode dédiée, puis orchestre les signaux/slots vers les modèles. Les vues exposent désormais tous les événements nécessaires (dessin, navigation) sans implémenter la logique de traitement.

**Décisions techniques :**
1. Injection des widgets Designer dans `ToolsPanel` via `attach_designer_widgets` pour éviter de toucher au fichier généré et garder le câblage côté vue.
2. Extension des signaux des vues pour couvrir les gestes (polygone/rectangle/point, crosshair, volume/caméra) afin que le contrôleur reste l’unique orchestrateur.
3. Enrichissement de `ViewStateModel` pour suivre l’état UI (slice, mode, seuils, toggles) avant d’appeler la logique métier dans les modèles dédiés.

---

# 2025-11-26 — Respect UI Designer layout for controller/views

**Tags:** `#controllers/master_controller.py`, `#views/tools_panel.py`, `#ui_mainwindow.py`, `#mvc`

**Actions effectuées:**
- MasterController réutilise `Ui_MainWindow` généré par Qt Designer (central widget, splitters, dock) et référence directement les vues créées (`frame_3` Endview, `frame_4` Volume, `frame_5` C-Scan, `frame_7` A-Scan, `dockWidgetContents_2` ToolsPanel) au lieu de reconstruire un layout qui écrasait la fenêtre.
- Connexions de signaux fictifs rétablies sur ces vues et sur le ToolsPanel (`tool_selected`, `mask_class_changed`, `alpha_changed`) tout en conservant uniquement des handlers `pass`.
- ToolsPanel reste contenu du dock (plus ajouté au layout principal), évitant qu’il occupe toute la fenêtre tout en conservant ses boutons fictifs et signaux de démonstration.

**Contexte:**
Un layout vertical recréé dans le contrôleur remplaçait la mise en page Qt Designer, faisant occuper toute la fenêtre par le ToolsPanel et les vues instanciées hors Designer. Le contrôleur s’aligne maintenant sur la structure définie dans `ui_mainwindow.py` pour préserver l’UI.

**Décisions techniques:**
1. Conserver Qt Designer comme source de vérité de la mise en page : le contrôleur se contente de connecter les signaux aux vues déjà posées par `Ui_MainWindow`.
2. Garder le ToolsPanel dans son dock dédié et n’exposer que ses signaux fictifs, sans imposer de layout global depuis le contrôleur pour éviter tout écrasement de l’UI.

---
