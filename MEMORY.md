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

### **2025-11-27** — Core viewer scaffolding for MVC UI

**Tags :** `#views/endview_view.py`, `#views/cscan_view.py`, `#views/ascan_view.py`, `#views/volume_view.py`, `#controllers/master_controller.py`, `#services/cscan_service.py`, `#mvc`, `#pyqt6`, `#vispy`, `#pyqtgraph`

**Actions effectuées :**
- Implémenté une `EndviewView` basée sur `QGraphicsView` avec zoom molette, crosshair, overlay semi-transparent, et conversion volume → pixmap ; gère `point_selected`, `slice_changed`, `drag_update`.
- Créé `CScanService.compute_top_projection` pour produire une heatmap (Z×X) avec agrégation max/mean sur l’axe Y ; la vue C-scan se contente d’afficher une matrice fournie.
- Ajouté un `CScanView` interactif : `QGraphicsView` + header LUT, curseur croisé et clic pour `slice_requested`, colorisation simple RGB ; conversions `QImage` → `QPixmap` corrigées.
- Remplacé `AScanView` par un widget PyQtGraph avec courbe normalisée, ligne horizontale mobile (`position_changed`).
- Réécrit `VolumeView` autour de VisPy (SceneCanvas + TurntableCamera) avec slider, fallback sur `get_colormap` et nettoyage des `scene.children` avant rerender.
- Adapté `MasterController` pour instancier `CScanService`, rafraîchir toutes les vues après chargement NDE, relier `EndviewView.point_selected` à l’A-scan, synchroniser `CScanView.highlight_slice` et utiliser `NDEModel.normalized_volume`.

**Contexte :**
Restructuration complète de l’UI MVC pour afficher les données NDE sans logique métier dans les vues. Les services calculent les projections (C-scan) et fournissent les volumes normalisés ; les vues ne conservent que l’affichage et les signaux utilisateur. Interaction contrôleur ↔ vues mise à jour pour les nouvelles API.

**Décisions techniques :**
1. Les conversions image utilisent `QPixmap.fromImage` afin d'éviter les erreurs PyQt (type QImage inattendu).
2. Nettoyage d'un `scene` VisPy via `child.parent = None` (impossible de setter `children`). Fallback colormap `get_colormap("viridis")` pour prévenir les couleurs inconnues et conserver un rendu stable.
3. Les vues ne calculent rien : toute projection ou normalisation reste dans les services (`CScanService`, `NdeLoaderService`), garantissant la séparation MVC.

---

### **2025-01-27** — Refactorisation MVC complète pour les vues Endview, 3D et A-scan

**Tags :** `#controllers/master_controller.py`, `#services/ascan_service.py`, `#models/nde_model.py`, `#views/endview_view.py`, `#views/ascan_view.py`, `#mvc`, `#refactoring`, `#architecture`

**Actions effectuées :**
- Analyse complète de l'architecture MVC pour les vues Endview, 3D et A-scan afin de garantir qu'elles dépendent uniquement de `ndeloader`
- Correction du signal `slice_changed` dans `EndviewView` : ajout de l'émission lors du scroll (Ctrl+molette) et méthode `set_crosshair()` pour synchronisation depuis le contrôleur
- Correction de la récursion infinie dans `AScanView` : ajout d'un garde `_suspend_marker_signal` pour éviter les boucles lors des mises à jour programmatiques
- Correction de la représentation A-scan : normalisation des amplitudes entre 0-100%, axes corrects (amplitude % vs positions ultrasoniques), support des différentes orientations (lengthwise/crosswise)
- **Refactorisation majeure** : création de `services/ascan_service.py` (159 lignes) pour extraire toute la logique métier NDT du contrôleur
- Nettoyage de `MasterController` : suppression de 117 lignes de logique métier, remplacement par 24 lignes d'orchestration pure
- Extension de `NDEModel` : ajout de méthodes utilitaires (`get_axis_map()`, `get_active_volume()`, `get_trace()`, etc.)
- Correction de bugs : arrays 2D au lieu de 1D (erreur PyQtGraph), erreur `memoryview` dans `QImage` pour fichiers transposés (ajout de `np.ascontiguousarray()`)

**Contexte :**
Avant de poursuivre le développement de l'application, il était critique de garantir que les vues Endview, 3D et A-scan s'affichent parfaitement et respectent l'architecture MVC à 100%. Toutes ces vues dépendent uniquement de `ndeloader` : un fichier .nde contient toutes les données nécessaires à leur création.

L'analyse a révélé que `MasterController` contenait de la logique métier NDT (mapping d'axes, extraction de profil, normalisation, conversion indices→positions), ce qui violait le principe MVC. Le contrôleur doit uniquement orchestrer, pas calculer.

**Décisions techniques :**
1. **Séparation stricte des responsabilités** : Toute la logique métier NDT (mapping d'axes, extraction de profil 1D, normalisation 0-100%, conversion indices→positions physiques, clamping) a été déplacée dans `AScanService`. Le contrôleur appelle le service et pousse les résultats vers les vues.

2. **Architecture MVC pure** :
   - **Services** : Logique métier NDT (`AscanService`, `CScanService`, `NdeLoaderService`)
   - **Controllers** : Orchestration uniquement (`MasterController` ne fait que coordonner)
   - **Views** : Rendu uniquement (`EndviewView`, `AScanView`, `VolumeView` ne calculent rien)
   - **Models** : Conteneurs de données (`NDEModel` expose les données, pas la logique)

3. **Gestion des signaux** : Les vues émettent des signaux pour les interactions utilisateur, le contrôleur les écoute et met à jour les modèles, puis rafraîchit toutes les vues dépendantes. Garde `_suspend_marker_signal` dans `AScanView` pour éviter les boucles infinies lors des mises à jour programmatiques.

4. **Support des orientations multiples** : `AScanService` gère automatiquement les différentes conventions d'orientation (lengthwise/crosswise) via le mapping d'axes du modèle, garantissant que l'A-scan est toujours extrait correctement quelle que soit l'orientation du fichier .nde.

5. **Normalisation cohérente** : Les amplitudes A-scan sont toujours normalisées entre 0-100% en utilisant soit le volume normalisé du modèle, soit les métadonnées min/max, garantissant une représentation cohérente dans la vue.

**Implémentation :**
```python
# AScanService encapsule toute la logique métier
class AScanService:
    def build_profile(self, model: NDEModel, point_hint: Optional[Tuple[int, int]] = None) -> Optional[AScanProfile]:
        # Mapping d'axes, extraction profil 1D, normalisation, positions physiques
        ...

# MasterController orchestre uniquement
def _update_ascan_trace(self, point: Optional[Tuple[int, int]] = None):
    profile = self.ascan_service.build_profile(self.nde_model, point_hint=point)
    if profile:
        self.ascan_view.set_signal(profile.signal_percent, positions=profile.positions)
        self.ascan_view.set_marker(profile.marker_index)
        self.endview_view.set_crosshair(*profile.crosshair)
```

---

### **2025-01-27** — Correction du bug de la croix C-Scan : ligne verticale qui ne bouge pas

**Tags :** `#views/cscan_view.py`, `#mvc`, `#bugfix`, `#ui`

**Actions effectuées :**
- Correction du bug où la ligne verticale de la croix dans la vue C-Scan ne suivait pas les mouvements de la souris
- Ajout de la mémorisation de la position actuelle de la croix (z, x) dans `CScanView` via l'attribut `_current_crosshair`
- Initialisation de la position de la croix au centre de la projection lors du chargement (`set_projection()`)
- Modification de `highlight_slice()` pour utiliser la dernière position X connue au lieu de la recentrer systématiquement sur le centre
- Mise à jour de `_update_cursor()` pour enregistrer et borner les positions (z, x) dans `_current_crosshair`

**Contexte :**
Lors du déplacement de la croix dans la vue C-Scan, la ligne verticale restait bloquée au centre. L'analyse a révélé que `highlight_slice()` recentrait systématiquement la croix sur l'axe X (`self._projection.shape[1] // 2`) à chaque appel. Quand le contrôleur gérait `crosshair_changed`, il rappelait `highlight_slice()`, ce qui recentrait la ligne verticale après chaque mouvement de souris, créant l'impression que la ligne était bloquée.

**Décisions techniques :**
1. **Mémorisation de l'état** : Ajout de `_current_crosshair: Optional[Tuple[int, int]]` pour stocker la position actuelle de la croix (z, x). Cette position est initialisée au centre de la projection lors du chargement et mise à jour à chaque mouvement de souris.

2. **Préservation de la position X** : `highlight_slice()` utilise maintenant la dernière position X connue (ou le centre si aucune position n'existe) au lieu de la recentrer systématiquement. Cela permet à la ligne verticale de suivre les mouvements de la souris tout en permettant au contrôleur de mettre à jour la position Z (slice) sans perdre la position X.

3. **Clamping et persistance** : `_update_cursor()` clamp les positions (z, x) dans les limites de la projection et met à jour `_current_crosshair`, garantissant que la position est toujours valide et mémorisée.

**Implémentation :**
```python
# Mémorisation de la position
self._current_crosshair: Optional[Tuple[int, int]] = None

# Initialisation lors du chargement
self._current_crosshair = (
    projection.shape[0] // 2,
    projection.shape[1] // 2,
)

# highlight_slice() utilise la dernière position X
def highlight_slice(self, slice_idx: int) -> None:
    _, last_x = self._current_crosshair or (
        slice_idx,
        self._projection.shape[1] // 2,
    )
    self._update_cursor(slice_idx, last_x)

# _update_cursor() enregistre la position
def _update_cursor(self, z: int, x: int) -> None:
    z_clamped = max(0, min(max_z, z))
    x_clamped = max(0, min(max_x, x))
    # ... mise à jour des lignes ...
    self._current_crosshair = (z_clamped, x_clamped)
```

---
