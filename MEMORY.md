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

### **2025-11-27** — Menu actions wiring

**Tags :** `#controllers/master_controller.py`, `#ui_mainwindow.py`, `#mvc`, `#signals-and-slots`

**Actions effectuées :**
- Connecté les actions de menu `Sauvegarder`, `Paramètres`, `Quitter` dans `MasterController._connect_actions` avec des slots dédiés pour suivre l’UI Designer sans modifier `ui_mainwindow.py`.
- Ajouté des handlers `_on_save`, `_on_open_settings`, `_on_quit` (fermeture immédiate de la fenêtre pour Quitter, autres en placeholder métier).

**Contexte :**
Alignement du contrôleur sur les actions du menu Fichier créées par Qt Designer afin que le contrôleur reste l’unique orchestrateur des commandes de l’UI.

**Décisions techniques :**
1. Ne pas toucher au fichier généré `ui_mainwindow.py` et injecter les connexions côté contrôleur.
2. Fournir un comportement minimal sécurisé pour Quitter via `main_window.close()`, en laissant Sauvegarder/Paramètres pour la logique métier future.

---

### **2025-11-27** — NDE loading via service in controller/model

**Tags :** `#controllers/master_controller.py`, `#models/nde_model.py`, `#services/nde_loader.py`, `#mvc`, `#signals-and-slots`

**Actions effectuées :**
- Intégré `NdeLoaderService` dans `MasterController` pour ouvrir un fichier .nde via QFileDialog, détecter l’orientation optimale, réordonner/transpose le volume et pousser données + métadonnées vers `NDEModel`.
- Implémenté `_orient_volume` pour déplacer l’axe slice en premier et appliquer un transpose global si requis par le service (axes inversés corrigés avant mise à jour modèle).
- Synchronisé l’état UI après chargement : mise à jour slice 0 dans ViewStateModel, spinbox (min/max), ToolsPanel et EndviewView (set_slice + update_image) avec message de statut.
- Complété `NDEModel` (set_volume, set_a_scan, set_current_slice, clear) pour stocker volume, a-scan, métadonnées et slice courante.

**Contexte :**
Le chargement NDE nécessite des ajustements d’axes selon la structure (public/domain). Le contrôleur s’appuie sur NdeLoaderService pour appliquer ces corrections avant de mettre à jour le modèle, en respectant l’UI générée par Designer.

**Décisions techniques :**
1. Recentrer le flux de chargement dans le contrôleur avec `NdeLoaderService` pour garantir la correction d’orientation (moveaxis + transpose global) avant stockage dans `NDEModel`.
2. Conserver `ui_mainwindow.py` intact en injectant uniquement via QFileDialog et updates vues ; retour utilisateur via statusbar.
3. Simplifier `NDEModel` en conteneur de données/métadonnées sans logique UI, laissant l’orchestration au contrôleur.

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

### **2025-11-27** — NDE loader renvoie un modèle orienté/normalisé prêt

**Tags :** `#services/nde_loader.py`, `#models/nde_model.py`, `#controllers/master_controller.py`, `#mvc`, `#orientation`, `#normalization`

**Actions effectuées :**
- Ajouté `NdeLoaderService.load_nde_model` qui charge les données, détecte l’orientation, réordonne/transpose le volume, le normalise en float32 et construit un `NDEModel` avec métadonnées (chemin, structure, orientation, min/max, normalisation) et slice initiale à 0.
- Créé les helpers `orient_volume` et `normalize_volume` et mis à jour `_current_nde_data/_path` pour exposer le dernier chargement.
- Étendu `NDEModel` avec `normalized_volume` optionnel, un setter dédié et un reset qui efface aussi la version normalisée.
- Simplifié `MasterController._on_open_nde` pour consommer directement le `NDEModel` retourné (plus de `_orient_volume` ni de transformations côté contrôleur), seulement mise à jour des bornes UI/slice.

**Contexte :**
La logique métier (orientation + normalisation) est maintenant confinée au service pour garantir une orientation cohérente avant affichage. Le contrôleur reste un orchestrateur qui met à jour l’état/vues à partir d’un modèle déjà prêt, tout en conservant le volume brut orienté et une copie normalisée pour l’UI ou les traitements ultérieurs.

**Décisions techniques :**
1. Centraliser le pipeline d’orientation dans le service (détection → moveaxis → transpose éventuel) pour assurer une base commune aux endviews ; aucune transformation dans le contrôleur.
2. Garder le volume brut orienté dans le modèle et stocker en parallèle une version normalisée float32 avec métadonnées min/max pour réutilisation ou rendu.
3. Le contrôleur remplace son `NDEModel` par celui du service et réinitialise l’UI sur la slice 0, en dérivant les bornes de spinbox depuis l’axe 0 du volume orienté.

---
