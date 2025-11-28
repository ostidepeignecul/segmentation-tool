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
# 2025-11-27 — VolumeView coupe 3D style volume_plane

**Tags :** `#views/volume_view.py`, `#3d-visualization`, `#vispy`, `#slider`, `#mvc`

**Actions effectuées :**
- Rebuild VolumeView scene : normalisation du volume, création d’un Volume VisPy + plane et Image de slice alignés sur l’axe Z (orientation endview) avec transforms calées sur la slice courante.
- Le slider pilote `set_slice_index` qui met à jour plane/image, clamp l’index et émet `slice_changed` + `camera_changed` pour la synchro contrôleur.
- Plan jaune opaque (depth off, ordre élevé) et image sous-jacente pour rendre la slice visible ; fallback colormap par nom de chaîne et nettoyage de la scène avant rebuild.

**Contexte :**
Le slider 3D devait illuminer la slice correspondante dans le volume, à la manière de l’exemple VisPy volume_plane. La vue 3D affiche désormais la slice courante comme texture plus un plan repère, synchronisés avec le slider et le contrôleur.

**Décisions techniques :**
1. Normaliser avec min/max du volume et utiliser l’axe 0 comme pile de slices ; Image placée à z = slice, plan au même z avec `order=10` et `depth_test=False`.
2. Conserver les signaux de navigation (`slice_changed`/`camera_changed`) pour que MasterController puisse réutiliser `_on_slice_changed` sans autre câblage.

---

### **2025-11-27** — Forcer l’orientation Domain sur l’axe lengthwise

**Tags :** `#services/nde_loader.py`, `#orientation`, `#nde_loader`, `#domain-structure`, `#mvc`

**Actions effectuées :**
- Ajout d’un cas dédié dans `detect_optimal_orientation` pour les fichiers Domain : on fixe `slice_orientation` à `lengthwise` et on calcule l’aspect du premier slice pour un éventuel transpose.
- Log d’orientation mis à jour avec le motif `reason: domain: preserve lengthwise as slice axis` pour tracer la décision.

**Contexte :**
Un fichier Domain (shape brut 268×301×568) devait produire 268 slices (endview 301×568), mais l’heuristique sélectionnait `ultrasound` comme axe slice (568 images). En forçant l’axe lengthwise pour Domain, le volume orienté reste (268, 301, 568) avec slices 301×568, conforme aux attentes de l’Endview et du C-scan.

**Décisions techniques :**
1. Préserver l’axe lengthwise comme axe des slices pour les structures Domain, car il représente l’empilement attendu et évite une inversion non souhaitée vers l’axe ultrasound.
2. Conserver la détection Public inchangée et ne toucher au transpose que si l’aspect du slice est < 1.0.

**Implémentation (extrait) :**
```python
if structure == "domain":
    sample = data_array[0, :, :]
    aspect = sample.shape[1] / sample.shape[0] if sample.shape[0] else 1.0
    cfg = {
        "slice_orientation": "lengthwise",
        "transpose": aspect < 1.0,
        "num_images": lengthwise_qty,
        "shape": sample.shape,
        "aspect": aspect,
        "reason": "domain: preserve lengthwise as slice axis",
    }
    nde_debug_logger.log_variable("orientation_config", cfg, indent=1)
    return cfg
```

---
### **2025-11-27** — Crosshair sync gated to Shift-click

**Tags :** `#views/cscan_view.py`, `#views/endview_view.py`, `#controllers/master_controller.py`, `#crosshair`, `#mvc`, `#signals-and-slots`

**Actions effectuées :**
- Bloqué les mouvements de croix au survol dans CScanView et exigé Shift+clic gauche pour mettre à jour/emitter `crosshair_changed` et `slice_requested`.
- Ajouté `set_crosshair(slice_idx, x)` dans CScanView pour synchroniser la croix depuis le contrôleur.
- Forcé EndviewView à ne déplacer la croix qu’au Shift+clic gauche (plus de mise à jour au hover) et à émettre les signaux à ce moment.
- Synchronisé le contrôleur pour pousser la croix C-scan lors de la mise à jour du profil A-scan, alignant la ligne verticale entre vues.

**Contexte :**
La croix bougeait au survol dans Endview/CScan, et seule la C-scan reflétait les clics. L’interaction doit rester fixe sauf action explicite. Les vues sont des renderers PyQt6 et doivent rester passives côté logique, le contrôleur orchestrant les positions partagées.

**Décisions techniques :**
1. Garde Shift+clic pour les interactions afin d’éviter les déplacements accidentels tout en préservant les signaux existants.
2. Ajout d’une API `set_crosshair` côté CScanView pour permettre au contrôleur de maintenir la cohérence verticale quand un point est choisi dans Endview/AScan.
3. Pas de mise à jour au hover dans les deux vues pour garder la croix stable ; seuls les appels programmatiques (highlight/set_crosshair) déplacent la croix.

**Implémentation (extrait) :**
```python
# CScanView event
if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.LeftButton:
    if event.modifiers() & Qt.ShiftModifier:
        z, x = coords
        self._update_cursor(z, x)
        self.crosshair_changed.emit(z, x)
        self.slice_requested.emit(z)

# CScanView API
def set_crosshair(self, slice_idx: int, x: int) -> None:
    self._update_cursor(clamped_z, clamped_x)

# Controller
self.cscan_view.set_crosshair(slice_idx, profile.crosshair[0])
```

---

### **2025-11-27** — Heuristique Domain assouplie pour l’orientation

**Tags :** `#services/nde_loader.py`, `#orientation`, `#domain-structure`, `#nde_loader`

**Actions effectuées :**
- Remplacé le forcing systématique de l’axe slice pour Domain par un biais : on préfère l’axe lengthwise si son nombre de slices est raisonnable (50–500) ou si l’aspect du slice est dans [0.2, 5], mais on laisse l’heuristique générique choisir sinon.
- Introduit un score avec biais (+8) pour l’orientation préférée afin de rester flexible tout en conservant la priorisation lengthwise dans les cas courants.
- Conserve le logging `reason` uniquement si l’orientation retenue correspond à la préférence Domain.

**Contexte :**
Certaines structures Domain doivent rester sur l’axe lengthwise (ex: 268×301×568 → 268 slices), mais d’autres peuvent nécessiter un autre axe. Il fallait éviter un forcing systématique tout en gardant un biais en faveur de lengthwise dans les cas attendus.

**Décisions techniques :**
1. Biais Domain : préférer lengthwise quand le compte de slices est dans une plage « normale » ou que l’aspect n’est pas trop extrême, sans empêcher un fallback.
2. Scoring unifié avec bonus pour l’orientation préférée pour ne pas régresser les heuristiques Public.

**Implémentation (extrait) :**
```python
if structure == "domain":
    aspect = sample.shape[1] / sample.shape[0] if sample.shape[0] else 1.0
    prefer_lengthwise = (50 <= lengthwise_qty <= 500) or (0.2 <= aspect <= 5.0)
    if prefer_lengthwise:
        preferred_orientation = {..., "reason": "domain: prefer lengthwise (qty/aspect heuristic)"}
...
def _score_orientation(o: Dict) -> int:
    base = ...
    aspect_score = ...
    bias = 8 if preferred_orientation and o["name"] == preferred_orientation["slice_orientation"] else 0
    return base + aspect_score + bias
```

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

### **2025-11-27** — Position label + cross toggle via ViewStateModel

**Tags :** `#views/tools_panel.py`, `#controllers/master_controller.py`, `#views/endview_view.py`, `#views/cscan_view.py`, `#views/ascan_view.py`, `#models/view_state_model.py`, `#ui`, `#crosshair`, `#mvc`

**Actions effectuées :**
- Ajouté signaux/toggles pour overlay et cross dans `ToolsPanel`, support du label de position, méthodes de mise à jour sans réémission, et wiring des widgets Designer.
- Étendu `ViewStateModel` avec `show_cross` et `cursor_position` plus setters, pour conserver l’état UI (cross/overlay/position) hors contrôleur/vue.
- Rétabli l’émission de la position souris Endview via `drag_update` sans bouger la croix, et ajouté les APIs `set_cross_visible` (Endview/CScan) et `set_marker_visible` (AScan).
- Contrôleur orchestre : synchronise les checkboxes initiales, stocke position dans le modèle, met à jour le label via ToolsPanel, et applique le toggle cross à toutes les vues.

**Contexte :**
Nouvelles cases « Toggle overlay » et « Toggle cross » + label de position ajoutées dans le UI Designer. La croix ne doit pas bouger au survol, mais la position souris doit se refléter dans le label, et le cross doit être masquable dans les trois vues.

**Décisions techniques :**
1. Stocker l’état (cross visible, overlay, dernière position) dans `ViewStateModel` pour éviter logique UI dans contrôleur/vues.
2. Exposer des méthodes dédiées dans les vues pour basculer la visibilité des repères, sans logique métier, et réémettre uniquement la position sur mouvement.
3. Synchroniser les widgets Designer via `ToolsPanel` avec blocage de signaux pour les mises à jour programmatiques afin d’éviter les boucles.

**Implémentation (extraits) :**
```python
# ViewStateModel
self.show_cross: bool = True
self.cursor_position: Optional[tuple[int, int]] = None

# ToolsPanel signals
overlay_toggled = pyqtSignal(bool)
cross_toggled = pyqtSignal(bool)

# Endview mouse move (pas de déplacement de croix)
coords = self._scene_coords_from_event(event)
if coords:
    self.drag_update.emit(coords)

# Controller toggle cross
self.view_state_model.set_show_cross(enabled)
self.endview_view.set_cross_visible(enabled)
self.cscan_view.set_cross_visible(enabled)
self.ascan_view.set_marker_visible(enabled)
```

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

### **2025-11-27** — VolumeView slice highlight via plane-mode Volume

**Tags :** `#views/volume_view.py`, `#vispy`, `#3d-visualization`, `#raycasting`

**Actions effectuées :**
- Conservé le Volume principal en mode `mip` et ajouté un second `scene.visuals.Volume` en `raycasting_mode="plane"` pour surligner la slice courante (axe 0), avec `plane_normal=(1, 0, 0)` et `plane_position=(slice, height/2, width/2)` configurés dans `_build_scene`.
- Mis à jour `_update_slice_plane` pour ne déplacer que `plane_position` du volume de surlignage selon `_current_slice`, en clampant la profondeur et en gardant l’image 2D de slice comme overlay de référence.

**Contexte :**
L’ancien plan séparé était mal aligné; on suit l’exemple VisPy volume_plane en laissant le volume gérer la coupe interne. Le volume complet reste visible tandis que la slice courante est mise en évidence via le volume en mode plane, alignée sur l’axe de pile (z=axis 0).

**Décisions techniques :**
1. Utiliser un Volume dédié en mode plane plutôt que modifier le Volume principal pour conserver le rendu volumique global et un surlignage clair de la slice active.
2. Aligner `plane_normal` sur l’axe 0 (stack) et centrer `plane_position` en x/y pour éviter les décalages; clamp de l’index pour rester dans les bornes du volume.

---
