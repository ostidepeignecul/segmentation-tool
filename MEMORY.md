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

### **2025-11-29** — Refactor corrosion workflow (C-scan/AScan controllers)

**Tags :** `#controllers/master_controller.py`, `#controllers/cscan_controller.py`, `#controllers/ascan_controller.py`, `#services/cscan_corrosion_service.py`, `#services/ascan_service.py`, `#services/distance_measurement.py`, `#views/cscan_view_corrosion.py`, `#mvc`, `#corrosion`, `#ascan`

**Actions effectuées :**
- Fusionné l’extraction A-Scan dans `services/ascan_service.py` (classe `AScanExtractor` + `export_ascan_values_to_json`) et supprimé `ascan_extractor.py`.
- Fusionné l’orchestrateur corrosion dans `services/cscan_corrosion_service.py` (ex-`CorrosionAnalysisService`) avec calcul distances, carte (Z×X) et overlay lignes; supprimé `corrosion_analysis_service.py`.
- Ajouté `controllers/cscan_controller.py` pour gérer la pile C-scan standard/corrosion, le workflow corrosion et la sélection stricte de 2 labels visibles; `controllers/ascan_controller.py` pour l’affichage A-Scan et synchro crosshair.
- Nettoyé `controllers/master_controller.py` pour déléguer C-scan/AScan, connecter l’action corrosion au contrôleur dédié et réinitialiser corrosion au chargement.
- Mis à jour `views/cscan_view_corrosion.py` (LUT cachée) et `services/distance_measurement.py` pour pointer vers la nouvelle extraction.

**Contexte :**
Le workflow corrosion devait fonctionner avec labels dynamiques (exactement 2 visibles), afficher une heatmap de distances (Z×X) avec colormap rouge→orange→jaune→bleu, et utiliser une vue/service dédiés sans impacter la C-scan standard. L’architecture a été recentrée en MVC : contrôleurs spécialisés pour C-scan et A-Scan, services fusionnés pour réduire les dépendances et clarifier les imports. La sélection des labels provient de la visibilité overlay ; en cas de nombre ≠2, le contrôleur journalise une erreur et n’exécute pas l’analyse.

**Décisions techniques :**
1. `CScanCorrosionService` orchestre extraction A-Scan filtrée, distances (pixels par défaut), carte de distances et overlay NPZ; projection corrosion nettoie NaN/inf.
2. `CScanController` possède la pile de vues (standard + `CscanViewCorrosion`), bascule la vue, met à jour les projections et lance l’analyse en validant les labels visibles; reset corrosion sur chargement.
3. `AScanController` gère l’affichage du profil, met à jour l’endview et synchronise le crosshair C-scan via un callback, allégeant `MasterController`.
4. `AScanExtractor` demeure vectorisé et acceptant un filtre de labels; les pixels sont indexés par id string, avec nom conservé pour métadonnées; carte de distances produite via `DistanceMeasurementService.build_distance_map` (pixels/mm optionnel).

---
### **2025-11-29** — Gestion overlay centralisée et labels dynamiques

**Tags :** `#controllers/master_controller.py`, `#models/annotation_model.py`, `#views/overlay_settings_view.py`, `#services/overlay_loader.py`, `#overlay`, `#mvc`

**Actions effectuées :**
- Déplacé la palette/visibilité des labels dans `AnnotationModel` : masque 3D, palette BGRA et visibilité vides par défaut, rebuild à partir d’un volume ou d’un NPZ (`set_mask_volume` réinitialise palette/visibilités et enregistre les classes présentes).
- Simplifié `OverlayLoader` : ne fait plus que charger un NPZ/NPY et retourner un volume uint8 (Z,H,W) aligné (transpose toléré 0,2,1), plus de palette/visibilité.
- `MasterController` : sur chargement NDE ou NPZ, clear `AnnotationModel` et `OverlaySettingsView`, sync des labels via `_sync_overlay_settings_with_model`, overlay poussé depuis `annotation_model.build_overlay_rgba()` après avoir vidé les overlays des vues pour éviter un stale render.
- `OverlaySettingsView` : plus de label par défaut, ajoute `clear_labels()` et nettoie avant `set_labels`; ajout manuel de labels conserve l’ID suivant via roue de teinte.

**Contexte :**
Le label “1” forcé créait des IDs fantômes et un overlay incohérent lors du chargement de NPZ (classes 1/2 devenaient 5/6). La palette/visibilité devait vivre dans le modèle d’annotation plutôt que dans la vue ou le loader pour garantir la cohérence entre overlay calculé et UI.

**Décisions techniques :**
1. Toujours repartir d’une palette/visibilité vide lors d’un nouveau volume ou NPZ pour refléter exactement les classes présentes.
2. Forcer un clear de l’overlay dans les vues avant de pousser le nouvel overlay afin d’éviter que VisPy/QPixmap conservent l’ancien rendu quand des labels sont masqués/affichés.
3. Laisser `OverlaySettingsView` purement déclarative : elle reflète l’état du modèle et ne crée rien tant qu’on ne lui passe pas de labels ou qu’on n’appuie pas sur “Ajouter un label”.

---

### **2025-11-28** — Overlay NPZ/NPY géré via NPZOverlayService et annotation_model

**Tags :** `#services/npz_overlay.py`, `#controllers/master_controller.py`, `#views/endview_view.py`, `#overlay`, `#mvc`

**Actions effectuées :**
- Simplifié NPZOverlayService : initialisation d’un volume masque vide aligné sur le NDE, chargement NPZ/NPY (shape validée), construction d’un volume RGBA via palette `MASK_COLORS_BGRA`, stockage `mask_volume`/`overlay_rgba`, clear.
- MasterController : délégation overlay au service (initialisation après chargement NDE, chargement overlay via service, push conditionnel vers la vue selon toggle overlay) sans logique de coloration ; annotation_model reçoit `mask_volume` du service.
- EndviewView : support overlay volumique RGBA (Z,H,W,4) ou slice 2D via extraction auto de la slice courante et conversion directe en QPixmap sans remapping.

**Contexte :**
Éviter la logique d’overlay dans le contrôleur : le service gère chargement et palette, le modèle stocke les masques, la vue affiche un volume RGBA prêt. L’overlay est créé vide au chargement NDE pour servir de toile lors des futurs outils de dessin.

**Décisions techniques :**
1. Validation stricte de la shape overlay contre le volume NDE et conversion en uint8 côté service ; couleurs provenant de `MASK_COLORS_BGRA` (fallback magenta semi-transparente pour classes inconnues).
2. La vue gère les volumes overlay RGBA en 4D en extrayant la slice courante ; le contrôleur ne fait qu’orchestrer les appels service/modèle et pousse l’overlay selon le toggle.

---

### **2025-11-28** — Overlay NPZ: auto-transpose H/W

**Tags :** `#services/npz_overlay.py`, `#overlay`, `#mvc`

**Actions effectuées :**
- `NPZOverlayService.load` tolère un overlay dont les dimensions H/W sont inversées par rapport au volume (ex: (Z, W, H) vs (Z, H, W)) : détection (même profondeur, axes 1 et 2 permutés), transpose `(0,2,1)` et log avant de bâtir l’RGBA.
- Les autres shapes restent rejetées par une erreur.

**Contexte :**
Des NPZ/NPY fournis avec H/W échangés doivent être acceptés sans demander de retoucher les vues/contrôleur ; le service corrige l’ordre pour livrer un masque aligné au volume NDE.

**Décision technique :**
1. Vérifier la profondeur puis appliquer un transpose (0,2,1) sur mismatch H/W au lieu d’échouer, afin de conserver une orientation cohérente avec le volume attendu.

---

### **2025-11-28** — Overlay 3D poussé via contrôleur

**Tags :** `#controllers/master_controller.py`, `#views/volume_view.py`, `#overlay`, `#mvc`

**Actions effectuées :**
- `_push_overlay` envoie désormais l’overlay RGBA du service à la fois à l’Endview et à la VolumeView, et les nettoie toutes deux si le toggle overlay est désactivé.

**Contexte :**
Le volume overlay (annotations) n’apparaissait que dans l’Endview ; le contrôleur pousse maintenant le même overlay vers la vue 3D pour afficher l’annotation par-dessus le volume NDE.

**Décision technique :**
1. Mutualiser le push overlay pour les deux vues afin de garder une seule source (service overlay_rgba) et respecter le toggle overlay.

---

### **2025-11-28** — Fix compat numpy (overlay asarray copy)

**Tags :** `#services/npz_overlay.py`, `#overlay`, `#numpy`

**Actions effectuées :**
- Remplacé `np.asarray(..., copy=False)` par `np.array(..., copy=False)` lors du chargement overlay pour éviter l’erreur `asarray() got an unexpected keyword argument 'copy'` sur les versions numpy antérieures.

**Contexte :**
La compatibilité numpy nécessitait d’éviter l’argument `copy` avec `asarray`; `np.array` supporte `copy` et conserve le cast en uint8 sans duplication inutile.

---

### **2025-11-28** — Rotation 90° horaire appliquée dans SimpleNdeLoader

**Tags :** `#services/simple_nde_loader.py`, `#orientation`, `#rotation`, `#mvc`

**Actions effectuées :**
- Ajout d’un pipeline de rotation 90° horaire dans le loader : `np.rot90(k=-1, axes=(1,2))` appliqué aux volumes après orientation fallback pour Public/Domain.
- Mise à jour de l’`axis_order` en inversant les axes Y/X (`[a0, a2, a1, *rest]`) et inversion du tableau de positions de l’ancien axe X (devenu Y) pour refléter le flip induit par la rotation ; l’ancien axe Y devient l’axe X sans inversion.

**Contexte :**
La rotation devait être effectuée côté loader pour respecter le MVC, les vues étant redevenues passives. Le loader fournit désormais un volume déjà roté (shape Z, W, H) avec métadonnées cohérentes.

**Décisions techniques :**
1. Rotation slice-wise dans le plan (Y,X) : conserve l’axe des slices (0) et permute largeur/hauteur avec inversion de l’ancien axe X.
2. Synchroniser métadonnées : swap axis_order et inverser uniquement les positions du nouvel axe Y (ancien X) pour suivre la rotation clockwise.

---
### **2025-11-28** — Orientation fix via display transforms (pas de rotation des données)

**Tags :** `#views/endview_view.py`, `#views/volume_view.py`, `#3d-visualization`, `#ui`, `#mvc`

**Actions effectuées :**
- Supprimé les `rot90` dans EndviewView : affichage direct des slices/overlays, coords écran ↔ données inchangées, croix tracée sans remapping.
- Supprimé les `rot90` dans VolumeView : le volume est utilisé tel que fourni par le loader ; ajout d’un flip visuel XY via `STTransform(scale=(-1, -1, 1), translate=(width, height, 0))` sur le volume et l’image de slice, avec translation z pour la slice.
- Caméra recentrée inchangée (centre sur la slice courante) ; la cohérence 2D/3D vient désormais du flip d’affichage et non d’une rotation des données.

**Contexte :**
Les rotations appliquées dans les vues violaient le MVC (données modifiées côté UI). Le besoin d’alignement 2D/3D est couvert par un flip visuel de la 3D, tandis que les données restent orientées par le loader. Endview redevient un simple renderer sans remapping de coordonnées.

**Décisions techniques :**
1. Pas de transformation des données dans les vues : respecter le rôle du loader/modèle pour l’orientation ; la vue ne fait qu’un flip d’affichage côté 3D.
2. Alignement 3D ↔ 2D via `STTransform` sur les visuels (volume + image slice) au lieu de rot90 des arrays ; conserver un centre caméra sur la slice active pour la navigation.

---
### **2025-11-28** — Rotations différenciées Endview/Volume

**Tags :** `#views/endview_view.py`, `#views/volume_view.py`, `#3d-visualization`, `#ui`, `#mvc`

**Actions effectuées :**
- Endview : affichage roté de 90° horaire (rot90 k=-1) pour chaque slice et overlay ; mapping des clics/croix converti pour rester dans le repère données (x_data = y_disp, y_data = H-1-x_disp ; croix inverse : x_disp = H-1-y, y_disp = x).
- VolumeView : affichage roté de 90° antihoraire sur chaque slice pour le rendu 3D (np.rot90 k=1 axes (1,2)), en conservant navigation/slider/caméra alignés sur les dimensions rotées.

**Contexte :**
Besoin d’appliquer une rotation horaire sur l’Endview tout en appliquant la rotation inverse sur la vue 3D, sans casser les coordonnées de données utilisées par le contrôleur ni la navigation.

**Décisions techniques :**
1. Rotation Endview uniquement visuelle : on pivote les arrays avant conversion en pixmap et on traduit les coordonnées écran → données et inversement pour la croix afin de garder le protocole (x,y) intact côté contrôleur.
2. Rotation VolumeView appliquée aux volumes normalisé/brut pour le rendu VisPy ; la profondeur reste en axe 0, largeur/hauteur sont permutés après rotation et la caméra/ranges utilisent ces dimensions rotées automatiquement.

---
### **2025-11-28** — Caméra centrée sur la slice courante en 3D

**Tags :** `#views/volume_view.py`, `#3d-visualization`, `#camera`, `#mvc`

**Actions effectuées :**
- Recentrage de la TurntableCamera sur le centre de la slice active : `camera.center = (width/2, height/2, slice_idx)` dans `_focus_camera_on_slice`.
- Initialisation cohérente du centre dans `_configure_camera` en plaçant directement le focus sur la slice courante plutôt qu’au milieu du volume.

**Contexte :**
Le pivot de caméra devait suivre la slice sélectionnée pour que les rotations se fassent autour de cette coupe (alignée avec l’endview) plutôt qu’autour du centre global du volume.

**Décisions techniques :**
1. Garder x/y centrés sur la largeur/hauteur pour un pivot stable et ne déplacer que la coordonnée z selon l’index de slice clampé.
2. Appliquer le même centre dès la configuration initiale pour éviter un premier focus au milieu du volume avant le repositionnement.

---

### **2025-11-28** — Désactivation du plan de surlignage VolumeView

**Tags :** `#views/volume_view.py`, `#3d-visualization`, `#mvc`

**Actions effectuées :**
- Commenté la création du visual VisPy en mode `raycasting_mode="plane"` dans `_build_scene`, désactivant le plan de surlignage de slice dans la vue 3D.
- La mise à jour de `plane_position` reste inoffensive car `_slice_highlight_visual` reste à `None`, évitant tout mouvement de plan.

**Contexte :**
Le plan mobile de surlignage devait être désactivé tout en conservant le rendu volumique et la navigation par slider. La vue consomme déjà un volume orienté comme l’Endview ; seul le repère de slice était à neutraliser.

**Décisions techniques :**
1. Conserver le code commenté pour une réactivation rapide, mais laisser `_slice_highlight_visual` à `None` pour que `_update_slice_plane` s’arrête immédiatement.
2. Ne pas toucher aux autres visuels (volume MIP, image 2D) ni à la caméra afin de limiter le changement au surlignage uniquement.

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
### **2025-11-28** — Rotation slice-wise après orientation NDE

**Tags:** `#services/nde_loader.py`, `#controllers/master_controller.py`, `#rotation`, `#orientation`, `#mvc`

**Actions effectuées:**
- Ajouté `force_rotation_k` et `set_rotation(quarter_turns)` dans `NdeLoaderService`, appliquant `np.rot90` sur les axes `(2, 1)` après `orient_volume` lorsque `force_rotation_k` est non nul, pour faire pivoter chaque slice tout en conservant le shape `[Z, H, W]`.
- Relié le contrôleur : connexion conditionnelle des actions Designer `actionRotate90/180/270/Reset` à `nde_loader.set_rotation`, et fallback `set_rotation(1)` (90°) juste avant `load_nde_model` si aucune action n’est disponible, tout en préservant le pipeline `detect → orient → rotate → normalize`.

**Contexte:**
Exigence d’ajouter un mécanisme de rotation 90°/180°/270° sur l’axe des slices sans toucher à la détection ou à la normalisation. La rotation est appliquée après l’orientation automatique dans le loader, et le contrôleur orchestre soit via des actions UI (si présentes), soit via un défaut 90° pour garantir l’application sans casser Endview/C-Scan/Volume.

**Décisions techniques:**
1. Utiliser `np.rot90(..., axes=(2, 1))` pour pivoter dans le plan (Y, X) et préserver l’axe stack (Z), évitant de perturber les indices de slice et la crosshair.
2. Connecter les actions UI si elles existent, sinon définir 90° avant chargement afin que la rotation soit disponible sans modifier les vues ni la logique d’orientation.

**Implémentation (extrait):**
```python
if self.force_rotation_k != 0:
    oriented_volume = np.rot90(
        oriented_volume,
        k=self.force_rotation_k,
        axes=(2, 1),
    )
```

---
### **2025-11-28** — Rotation appliquée aux endviews en mémoire

**Tags:** `#services/nde_loader.py`, `#rotation`, `#orientation`, `#endview`, `#mvc`

**Actions effectuées:**
- Ajouté l’appel à `np.rot90` conditionné par `force_rotation_k` dans `load_nde_as_memory_images` après le transpose des slices, pour appliquer les rotations 90°/180°/270° dans le plan (Y, X) sur chaque slice générée en mémoire.
- Rotation appliquée avant normalisation et conversions uint8/BGR, afin que les caches `_cached_raw_slices` et `_cached_images` reflètent la même orientation que le volume orienté/roté utilisé par les vues.

**Contexte:**
Après avoir introduit `set_rotation` dans le loader, les endviews générées en mémoire devaient suivre la même rotation slice-wise que le volume principal afin de préparer l’annotation future sur ces images sans divergence d’orientation.

**Décisions techniques:**
1. Insérer `np.rot90(img_data, k=self.force_rotation_k)` juste après le transpose pour opérer dans le plan (Y, X) de la slice déjà orientée, en conservant la pile `[Z, H, W]`.
2. Laisser la rotation d’affichage existante (rot90 k=-1 quand `transpose` est False) pour ne pas changer le rendu attendu, tout en garantissant que la rotation forcée soit déjà intégrée dans les données normalisées et BGR retournées.

**Implémentation (extrait):**
```python
img_data = self.extract_slice(data_array, idx, orientation)
if transpose:
    img_data = img_data.T

if self.force_rotation_k != 0:
    img_data = np.rot90(img_data, k=self.force_rotation_k)
```

---
### **2025-11-28** — Orientation caméra VolumeView alignée sur Endview

**Tags:** `#views/volume_view.py`, `#camera`, `#orientation`, `#vispy`, `#mvc`

**Actions effectuées:**
- Initialisé la TurntableCamera avec `up="+y"`, `azimuth=90`, `elevation=0`, et centre (0,0,0) pour viser frontalement les slices comme l’Endview.
- Introduit `_configure_camera(depth, height, width)` appelé après build de scène pour recentrer la caméra sur le volume, fixer up/azimuth/elevation, et définir le range (0..dim) afin que la rotation souris pivote autour du centre du volume.

**Contexte:**
La vue 3D devait présenter le volume dans le même sens que l’Endview rotatée, avec un point d’ancrage centré pour les rotations souris. L’ancienne orientation (elevation/azimuth 30/45 par défaut) affichait le volume sous un angle générique, décalé du référentiel Endview.

**Décisions techniques:**
1. Utiliser `up="+y"` et `azimuth=90/elevation=0` pour aligner l’axe vertical sur la hauteur des slices et regarder perpendiculairement à l’axe de pile, reflétant l’Endview rotatée.
2. Recaler la caméra sur `(depth/2, height/2, width/2)` et appeler `set_range` avec les bornes des trois axes pour que la rotation se fasse autour du centre du volume sans glisser hors cadre.

**Implémentation (extrait):**
```python
self._view.camera = scene.TurntableCamera(
    fov=45,
    elevation=0,
    azimuth=90,
    up="+y",
    center=(0.0, 0.0, 0.0),
)
...
self._view.camera.up = "+y"
self._view.camera.center = center
self._view.camera.azimuth = 90.0
self._view.camera.elevation = 0.0
self._view.camera.set_range(
    x=(0.0, float(depth)),
    y=(0.0, float(height)),
    z=(0.0, float(width)),
)
```

---
### **2025-11-28** — Caméra VolumeView centrée sur la slice courante

**Tags:** `#views/volume_view.py`, `#camera`, `#orientation`, `#vispy`, `#mvc`

**Actions effectuées:**
- Ajouté `_focus_camera_on_slice()` et appelé depuis `set_slice_index` et `_configure_camera` pour recaler la TurntableCamera sur `(current_slice, height/2, width/2)` à chaque changement de slice ou rebuild de scène.
- Conserve l’orientation `up="+y"`, `azimuth=90`, `elevation=0` et range calé sur les dimensions, en appliquant la mise au point slice après `set_range` pour garder le pivot sur la coupe active.

**Contexte:**
La vue 3D devait rester focalisée sur la slice courante (comme l’Endview rotatée) lors des déplacements du slider ou des reconstructions de scène, pour que les rotations souris pivotent autour de la coupe active.

**Décisions techniques:**
1. Centrer la caméra sur la slice active plutôt qu’au milieu du volume pour aligner le référentiel de navigation avec la coupe visualisée.
2. Appliquer le recentrage après `set_range` pour éviter que VisPy ne réécrase le center et garantir une ancre stable lors des rotations.

**Implémentation (extrait):**
```python
def _focus_camera_on_slice(self) -> None:
    if self._view.camera is None or self._volume is None:
        return
    depth, height, width = self._volume.shape
    clamped = max(0, min(depth - 1, int(self._current_slice)))
    self._view.camera.center = (
        float(clamped),
        height / 2.0,
        width / 2.0,
    )
```

---
### **2025-11-28** — Focalisation caméra VolumeView sur l’axe Z

**Tags:** `#views/volume_view.py`, `#camera`, `#orientation`, `#vispy`, `#mvc`

**Actions effectuées:**
- Ajusté `_focus_camera_on_slice` pour déplacer le centre de la caméra le long de l’axe Z (largeur) en fonction de la slice courante, tout en laissant X/Y centrés sur le milieu du volume.
- Clampe l’index de slice sur la dimension Z avant de le convertir en focus, afin d’éviter de sortir des bornes si la profondeur diffère de la largeur.

**Contexte:**
La caméra se recadrait perpendiculairement à la pile (axe slices). La demande est de faire suivre le mouvement du slider en focalisant sur l’axe Z du volume au lieu de l’axe des slices, pour un point d’ancrage aligné sur la largeur (Z) plutôt que sur la profondeur (stack).

**Décisions techniques:**
1. Map le slider (slice index) sur l’axe Z via clamp `[0, width-1]`, en gardant le centre X/Y sur la moitié du volume pour conserver la stabilité des rotations.
2. Ne pas toucher à l’orientation de la caméra (up/azimuth/elevation) déjà alignée sur l’Endview ; seul le point focal se déplace sur l’axe Z.

**Implémentation (extrait):**
```python
z_focus = max(0, min(width - 1, int(self._current_slice)))
self._view.camera.center = (
    depth / 2.0,
    height / 2.0,
    float(z_focus),
)
```

---
### **2025-11-28** — Référence Volume/CScan mises à jour selon ui_mainwindow

**Tags:** `#controllers/master_controller.py`, `#ui_mainwindow.py`, `#mvc`, `#views/volume_view.py`, `#views/cscan_view.py`

**Actions effectuées:**
- Inversé les références des vues dans `MasterController` pour suivre l’échange de frames dans `ui_mainwindow.py`: `cscan_view` pointe maintenant sur `frame_4` et `volume_view` sur `frame_5`.

**Contexte:**
Le fichier Designer a été modifié pour intervertir les frames (CScanView sur `frame_4`, VolumeView sur `frame_5`). Le contrôleur devait refléter ce mapping pour que les signaux et mises à jour utilisent les bons widgets.

**Décisions techniques:**
1. Mettre à jour uniquement les affectations de références sans toucher au wiring existant, afin de conserver la logique MVC tout en alignant le contrôleur avec la nouvelle structure UI.

---
### **2025-11-28** — Caméra VolumeView flippée de 180°

**Tags:** `#views/volume_view.py`, `#camera`, `#orientation`, `#vispy`, `#mvc`

**Actions effectuées:**
- Modifié l’azimuth de la TurntableCamera à 270° (au lieu de 90°) dans l’initialisation et dans `_configure_camera`, appliquant un flip de 180° de la vue 3D tout en conservant `up="+y"` et `elevation=180°`.

**Contexte:**
La vue 3D devait être inversée de 180° pour correspondre à l’orientation souhaitée. Seule la direction de vue change, la focalisation sur la slice courante et le point d’ancrage existant restent inchangés.

**Décisions techniques:**
1. Ajuster l’azimuth à 270° et l’elevation à 180° pour obtenir le flip sans modifier up ou la logique de focus sur la slice.
2. Garder `_configure_camera` en cohérence pour que tout rebuild conserve l’orientation inversée.

**Implémentation (extrait):**
```python
self._view.camera = scene.TurntableCamera(
    fov=45,
    elevation=180,
    azimuth=270,
    up="+y",
    center=(0.0, 0.0, 0.0),
)
...
self._view.camera.azimuth = 270.0
self._view.camera.elevation = 180.0
```

---
### **2025-11-28** — Flip vertical (Y) de la scène VolumeView

**Tags:** `#views/volume_view.py`, `#camera`, `#orientation`, `#vispy`, `#mvc`

**Actions effectuées:**
- Ajouté un transform de scène `STTransform(scale=(1, -1, 1), translate=(0, height, 0))` dans `_configure_camera` pour inverser l’axe Y du rendu 3D (sol ↔ plafond) sans modifier les données du volume.

**Contexte:**
Le volume était correct mais la perception devait être entièrement « retournée » comme si on plaçait le bloc à l’envers. Plutôt que de modifier les données, on applique un flip vertical au niveau de la scène VisPy.

**Décisions techniques:**
1. Utiliser un `STTransform` de la scène pour inverser Y avec une translation égale à la hauteur afin de garder le volume dans le champ visuel après inversion.
2. Laisser la configuration caméra (azimuth 270°, elevation 180°, up "+y") intacte pour conserver le reste de l’orientation et le focus slice existant.

**Implémentation (extrait):**
```python
self._view.scene.transform = STTransform(
    scale=(1.0, -1.0, 1.0),
    translate=(0.0, float(height), 0.0),
)
```

---
### **2025-11-28** — Flip Y appliqué aux visuels VolumeView

**Tags:** `#views/volume_view.py`, `#camera`, `#orientation`, `#vispy`, `#mvc`

**Actions effectuées:**
- Appliqué le flip vertical directement sur chaque visuel (`Volume`, `slice_highlight`, `Image`) via `STTransform(scale=(1,-1,1), translate=(0, height, 0))`, au lieu de la scène globale.
- Refixé l’elevation caméra à 0 tout en conservant azimuth 270 et up "+y"; la mise au point slice reste en place.

**Contexte:**
Le flip de scène ne produisait pas l’effet visuel attendu. En flippant chaque visuel, le rendu s’inverse réellement sans modifier les données, avec la caméra conservant son orientation et son focus sur la slice.

**Décisions techniques:**
1. Placer le transform sur les visuels pour garantir l’inversion de rendu même avec la caméra actuelle, et translater de `height` pour garder le volume dans le cadre après flip.
2. Laisser la configuration caméra (azimuth 270, elevation 0, up "+y") et `_focus_camera_on_slice` intacts pour isoler le flip au niveau des visuels.

**Implémentation (extrait):**
```python
flip_y = STTransform(scale=(1.0, -1.0, 1.0), translate=(0.0, float(height), 0.0))
self._volume_visual.transform = flip_y
self._slice_highlight_visual.transform = flip_y
...
self._slice_image.transform = STTransform(
    scale=(1.0, -1.0, 1.0),
    translate=(0.0, float(height), 0.0),
)
```

---
### **2025-11-28** — Flip appliqué aux mises à jour d’image de slice

**Tags:** `#views/volume_view.py`, `#vispy`, `#orientation`, `#mvc`

**Actions effectuées:**
- `_update_slice_image` réapplique désormais le flip vertical lors de chaque mise à jour, en utilisant `STTransform(scale=(1,-1,1), translate=(0, height, slice_idx))` pour conserver l’inversion visuelle du slider/plane lorsqu’on change de slice.

**Contexte:**
Le flip de l’Image était écrasé par `_update_slice_image` qui remplaçait la transform par une simple translation; la slice mise en évidence ne reflétait plus l’inversion. La transform est maintenant cohérente à chaque rafraîchissement.

**Décisions techniques:**
1. Inclure le flip (scale Y = -1, translate = height) dans `_update_slice_image` pour ne pas perdre l’inversion après un changement de slice.
2. Conserver la translation Z sur `current_slice` pour aligner l’overlay avec la position du slider.

**Implémentation (extrait):**
```python
if self._volume is not None:
    height = self._volume.shape[1]
    self._slice_image.transform = STTransform(
        scale=(1.0, -1.0, 1.0),
        translate=(0.0, float(height), float(self._current_slice)),
    )
```

---
### **2025-11-28** — Fallback d’orientation pour NDE sans métadonnées

**Tags:** `#services/simple_nde_loader.py`, `#controllers/master_controller.py`, `#services/ascan_service.py`, `#orientation`, `#domain-structure`, `#logging`, `#simple_nde_model`, `#mvc`

**Actions effectuées:**
- Ajouté un repli d’orientation dans `SimpleNdeLoader`: si les `dimensions` JSON sont absents (axes auto `axis_*`), on déplace l’axe le plus long en axe de slice (index 0) via `np.moveaxis`, en conservant l’axe ultrasound en dernière position ; log `[structure] fallback orientation: moved axis …`.
- Log détaillé des axes dans le loader (nom, quantity, shape) et dans `MasterController` au chargement (structure, shape, axis_order, taille des positions, path) pour diagnostiquer les fichiers Domain sans métadonnées.
- Corrigé `AScanService.build_profile` pour passer l’argument `ultrasound_axis` à `_axis_positions_for_profile`, évitant un `TypeError` lors du recalcul des positions.

**Contexte:**
Un fichier Domain (shape brut 113×3408×276) était interprété avec 113 slices faute de métadonnées `dimensions`. Le fallback réoriente désormais automatiquement le volume pour mettre l’axe le plus long (3408) en axe slice, améliorant l’affichage Endview/C-Scan/A-Scan tout en loggant l’opération. Les logs permettent de comparer avec les fichiers Public qui fournissent U/V/Ultrasound explicites.

**Décisions techniques:**
1. N’appliquer le repli que si `axis_order` est généré (`axis_*`), afin de ne pas modifier les fichiers disposant de métadonnées explicites.
2. Utiliser `np.argmax(shape)` pour cibler l’axe slice et `np.moveaxis` pour le placer en index 0 ; conserver l’ordre des autres axes (ultrasound restant en dernière position par défaut) et reconstruire `positions` selon le nouvel ordre.
3. Consolider la traçabilité: logs loader+controller au niveau INFO pour inspecter shape/axes/paths lors de l’ouverture d’un NDE.

**Implémentation (extrait):**
```python
# services/simple_nde_loader.py
slice_axis = int(np.argmax(shape))
if slice_axis != 0:
    reordered_data = np.moveaxis(data, slice_axis, 0)
    reordered_axis_order = [axis_order[slice_axis], *axis_order[:slice_axis], *axis_order[slice_axis + 1:]]
    logger.info("[%s] fallback orientation: moved axis %s to slice axis | old shape=%s | new shape=%s",
                structure, slice_axis, shape, reordered_data.shape)
    return reordered_data, reordered_axis_order, reordered_positions
```
```python
# controllers/master_controller.py
self.logger.info(
    "NDE loaded | structure=%s | shape=%s | axes=%s | path=%s",
    loaded_model.metadata.get("structure"),
    volume.shape,
    "; ".join(axes_info) if axes_info else "n/a",
    loaded_model.metadata.get("path"),
)
```
```python
# services/ascan_service.py
positions = self._axis_positions_for_profile(
    model,
    normalized.size,
    ultrasound_axis,
)
```

---

### **2025-11-28** — Overlay 3D translucide dans VolumeView

**Tags :** `#views/volume_view.py`, `#overlay`, `#vispy`, `#3d-visualization`, `#mvc`

**Actions effectuées :**
- Ajout d’un VolumeVisual overlay translucide en plus de l’image de slice : conversion de l’overlay RGBA (Z,H,W,4) en volume alpha float32 [0,1], création `scene.visuals.Volume(..., method="translucent", cmap=_TranslucentMask(), threshold=0.1)` avec blend actif et depth_test désactivé.
- Maintien de la synchronisation slice/flip XY : le Volume overlay partage `STTransform(scale=(-1,-1,1), translate=(width,height,0))` appliqué avec le volume principal ; `set_overlay(None)` nettoie VolumeVisual et image 2D, reconstruction après `set_volume`.

**Contexte :**
L’overlay n’apparaissait qu’en 2D (Image de slice). Le canal alpha du RGBA pilote maintenant un volume translucent en 3D, tout en conservant l’overlay de slice existant et le toggle overlay côté contrôleur.

**Décisions techniques :**
1. Utiliser l’alpha du RGBA comme scalaire (pas de remappage couleurs) et un colormap GLSL translucide type TransFire pour mélanger proprement avec le volume principal.
2. Désactiver le depth test et activer le blend sur le Volume overlay pour le dessiner au-dessus du MIP, tout en réappliquant le flip XY via `_apply_visual_transform` afin d’aligner Endview et 3D.

---

### **2025-11-28** — Overlay non repush sur navigation de slices

**Tags :** `#controllers/master_controller.py`, `#overlay`, `#performance`, `#mvc`

**Actions effectuées :**
- Supprimé l’appel à `_push_overlay()` dans `_on_slice_changed` pour éviter de réinjecter le volume RGBA à chaque déplacement de slice.

**Contexte :**
Chaque mouvement de slider déclenchait un push complet de l’overlay RGBA vers les vues, générant des logs et un coût inutile. Les vues gèrent déjà la mise à jour de slice via `set_slice` / `set_slice_index` et l’overlay 2D/3D se met à jour localement sans recharger les données.

**Décisions techniques :**
1. Laisser `_push_overlay` uniquement sur chargement NDE, chargement overlay ou toggle overlay, car ces événements changent effectivement les données.
2. S’appuyer sur les vues (Endview/VolumeView) pour rafraîchir l’overlay slice côté rendu lors des changements d’index, évitant un repush volumique coûteux.

---

### **2025-11-28** — Log diagnostique A-Scan au chargement

**Tags :** `#controllers/master_controller.py`, `#services/ascan_service.py`, `#ascan`, `#logging`, `#mvc`

**Actions effectuées :**
- Ajout de `_log_ascan_preview` appelé juste après le chargement NDE : extrait un profil A-Scan au centre (slice, x, y) et loggue longueur, min/max/mean, head(5), marker et crosshair.

**Contexte :**
Les fichiers Domain sans métadonnées ont un A-Scan « spécial » après fallback d’orientation. Le log console permet d’inspecter rapidement le contenu du profil sans interagir dans l’UI.

**Décisions techniques :**
1. Utiliser `AScanService.build_profile` pour respecter l’axe ultrasound détecté et les positions ; log structuré seulement si profil non vide.
2. Éviter le bruit : early return si volume invalide/vide, head limité à 5 valeurs, stats simples (len/min/max/mean) pour lecture console.

---

### **2025-11-28** — Compat numpy pour log A-Scan

**Tags :** `#controllers/master_controller.py`, `#ascan`, `#logging`, `#numpy`

**Actions effectuées :**
- Supprimé l’usage de `initial=` dans `min/max/mean` pour le log A-Scan (incompatible avec certaines versions numpy). Utilisation directe de `sig.min()/max()/mean()` après vérification `size>0`.

**Contexte :**
Une exception `_mean() got an unexpected keyword argument 'initial'` apparaissait sur des environnements numpy plus anciens lors du chargement NDE. Le log diagnostic A-Scan fonctionne maintenant sans ce paramètre.

**Décisions techniques :**
1. Conserver le guard `size>0` avant calcul, ce qui rend inutile l’argument `initial`.

---

### **2025-11-28** — Logs A-Scan enrichis (axe ultrasound, brut)

**Tags :** `#controllers/master_controller.py`, `#ascan`, `#logging`, `#diagnostic`

**Actions effectuées :**
- `_log_ascan_preview` appelle désormais `_log_raw_ascan` qui extrait le profil brut selon l’axe ultrasound détecté (mêmes règles que l’AScanService) et loggue len/min/max/mean/head(5) + axe utilisé.
- Ajout de `_detect_ultrasound_axis` côté contrôleur pour refléter la logique de l’AScanService quand l’axe « ultrasound » n’est pas nommé, fallback sur le dernier axe.

**Contexte :**
Pour les NDE Domain sans métadonnées (axis_*), il faut comparer les profils normalisés et bruts afin d’identifier les axes et valeurs atypiques. Les logs bruts donnent un aperçu immédiat dans la console après chargement.

**Décisions techniques :**
1. Calculer le profil brut directement depuis `nde_model.volume` en suivant l’axe ultrasound détecté pour rester cohérent avec la construction du profil normalisé.
2. Garder la sortie concise (head 5) et ignorer si volume/profil vide pour ne pas polluer la console.

---

### **2025-11-28** — Heuristique ultrasound sur axe le plus long

**Tags :** `#services/ascan_service.py`, `#ascan`, `#domain-structure`, `#orientation`, `#mvc`

**Actions effectuées :**
- Ajusté `_ultrasound_axis_index` : si aucun axe n’est nommé « ultrasound », on choisit désormais l’axe le plus long parmi les axes > 0 (non-slice) au lieu du dernier axe par défaut.

**Contexte :**
Les fichiers Domain sans métadonnées `dimensions` voyaient l’axe ultrasound sélectionné par défaut sur le dernier axe (longueur 113), alors que l’axe attendu est de longueur 276. Cette heuristique sélectionne désormais l’axe le plus long (ici 276) pour obtenir des profils A-Scan corrects.

**Décisions techniques :**
1. Considérer les axes après l’axe slice (index 0) et prendre le plus long, tri décroissant, pour refléter l’axe ultrasound probable quand il n’est pas nommé.
2. Garder le fallback `max(0, len(shape)-1)` si la shape est trop courte (<3 axes).

---

### **2025-11-28** — Logs A-Scan déplacés dans le service

**Tags :** `#services/ascan_service.py`, `#controllers/master_controller.py`, `#ascan`, `#logging`, `#mvc`

**Actions effectuées :**
- Ajout de `AScanService.log_preview(logger, model, volume, slice_idx=None, point=None)` qui loggue profil normalisé et brut (longueur, min/max/mean, head5, marker/crosshair, axe ultrasound détecté) en réutilisant `_ultrasound_axis_index`.
- `MasterController` délègue désormais le diagnostic A-Scan à `AScanService.log_preview` après chargement NDE ; suppression des méthodes de log internes et de l’import numpy inutile.

**Contexte :**
Centraliser les logs A-Scan dans la couche service évite la duplication de logique de détection d’axe ultrasound dans le contrôleur et facilite le diagnostic des NDE Domain/Public.

**Décisions techniques :**
1. Utiliser `build_profile` pour le profil normalisé (respect de l’axe ultrasound détecté), puis extraire un profil brut cohérent depuis `model.volume` pour comparaison.
2. Garder des logs concis (head5, stats de base) et ignorer les cas volume/profil vides pour éviter le bruit.

---

### **2025-11-28** — Debug log A-Scan externalisé

**Tags :** `#services/ascan_service.py`, `#controllers/master_controller.py`, `#services/ascan_debug_logger.py`, `#logging`, `#ascan`, `#mvc`

**Actions effectuées :**
- Créé `services/ascan_debug_logger.py` : singleton écrivant `ascan_debug_log.txt` (start_session/ensure_session/end_session, log_preview pour stats normalisées/brutes).
- Créé `services/cscan_debug_logger.py` (squelette) pour futurs besoins C-Scan.
- `AScanService.log_preview` loggue toujours via le logger standard et en parallèle via `ascan_debug_logger` (profil normalisé + brut). Si profil absent, log un message et envoie `normalized=None`.
- `MasterController` démarre la session A-Scan debug lors du chargement NDE (`ascan_service_start_session(file_path)`) et délègue le diagnostic A-Scan au service (plus de code de log dans le contrôleur, import numpy supprimé).
- Heuristique ultrasound inchangée (axe non nommé → plus long des axes > 0) utilisée aussi pour le brut.

**Contexte :**
Centralisation du logging A-Scan hors contrôleur pour faciliter le diagnostic Domain/Public, avec fichiers dédiés aux traces A-Scan et un squelette pour les futurs logs C-Scan.

**Décisions techniques :**
1. Un fichier de log par session NDE (`start_session(source)` écrase l’ancien fichier et ajoute l’en-tête).
2. Logs concis (head5, stats de base) et silencieux si volume/profil vide.

---

### **2025-11-29** — Renommage NDE/overlay loaders et modèle

**Tags :** `#services/nde_loader.py`, `#services/overlay_loader.py`, `#models/nde_model.py`, `#controllers/master_controller.py`, `#services/ascan_service.py`, `#models/__init__.py`, `#mvc`, `#rename`

**Actions effectuées :**
- Renommé `services/simple_nde_loader.py` en `services/nde_loader.py` et la classe `SimpleNdeLoader` en `NdeLoader`, mis à jour docstrings, imports et instanciations.
- Renommé `services/npz_overlay.py` en `services/overlay_loader.py` et la classe `NPZOverlayService` en `OverlayLoader`, ajusté l’import du contrôleur.
- Renommé `models/simple_nde_model.py` en `models/nde_model.py` et la classe `SimpleNDEModel` en `NdeModel`, exportée via `models/__init__.py` et retipée dans `AScanService` et `MasterController`.
- Vérifié via `rg` qu’aucune référence code aux anciens noms/fichiers ne reste (hors historiques dans MEMORY.md).

**Contexte :**
Harmonisation du naming NDE/overlay pour cohérence avec les autres services et modèles. Le comportement reste identique : le loader NDE prépare toujours un `NdeModel`, le service overlay construit l’overlay RGBA, et le contrôleur consomme les nouveaux noms.

**Décisions techniques :**
1. Aucun alias conservé pour les anciens noms afin d’éviter la dérive; tous les imports/usages pointent sur les nouveaux fichiers/classes.
2. Pas de changement fonctionnel ni de dépendance : refactor strict de nommage avec vérification par recherche textuelle (`rg`).

---

### **2025-11-29** — ViewStateModel centralisation & controller cleanup

**Tags :** `#models/view_state_model.py`, `#controllers/master_controller.py`, `#controllers/cscan_controller.py`, `#controllers/ascan_controller.py`, `#mvc`, `#corrosion`, `#state-management`

**Actions effectuées :**
- Remplacé `ViewStateModel` par une version riche: bornes de slices (set_slice_bounds/clamp_slice/set_slice), gestion crosshair/cursor (update_crosshair, set_current_point), toggles overlay/volume/crosshair, axes/camera state, et état corrosion (activate/deactivate + projection).
- Ajusté `MasterController` pour déléguer le clamping/état au modèle (set_slice/limits, update_crosshair), stocker axis_order, synchroniser camera_state, et laisser `CScanController`/`AScanController` piloter les vues.
- Mis à jour `CScanController` pour utiliser l’état corrosion du modèle (activate/deactivate_corrosion, projection stockée dans le modèle) et réinitialiser en cas d’erreur ou d’input invalide.
- Mis à jour `AScanController` pour mettre à jour le crosshair via le modèle et nettoyer current_point/cursor_position lors des clear.

**Contexte :**
Objectif: sortir la logique de state (clamping, crosshair, corrosion flags) des contrôleurs vers `ViewStateModel`, réduire la duplication et faciliter la bascule standard/corrosion.

**Décisions techniques :**
1. `ViewStateModel` porte désormais les bornes de tranches, l’état corrosion (flag + projection), l’axis_order et le camera_state pour éviter du code dans les contrôleurs.
2. Les contrôleurs appellent `set_slice`, `update_crosshair`, `activate_corrosion`/`deactivate_corrosion`; tout clamping passe par `ViewStateModel.clamp_slice`.
3. En cas d’échec du workflow corrosion (volume/mask manquants, labels ≠2, exceptions), `CScanController` désactive systématiquement le mode corrosion via le modèle avant de sortir.

---

### **2025-11-29** — Extraction du rendu overlay vers OverlayService

**Tags :** `#services/overlay_service.py`, `#models/annotation_model.py`, `#controllers/master_controller.py`, `#overlay`, `#mvc`, `#numpy`, `#pyqt6`

**Actions effectuées :**
- Création de `OverlayService.build_overlay_rgba(mask_volume, label_palette, visible_labels)` qui génère un volume BGRA (Z,H,W,4) sans dépendance PyQt6 en appliquant la palette (avec fallback) et un filtrage des labels visibles.
- Suppression de la logique de rendu RGBA dans `AnnotationModel`, ajout d’accesseurs `get_mask_volume`, `get_visible_labels`, `get_label_palette` pour exposer volume/palette/visibilité sans NumPy lié au rendu.
- Mise à jour de `MasterController._push_overlay` pour appeler `OverlayService` après avoir récupéré volume/palette/labels visibles depuis `AnnotationModel`, avec instanciation `self.overlay_service` dans `__init__`.

**Contexte :**
Objectif d’aligner l’architecture sur un MVC plus strict : sortir le rendu d’overlay hors du modèle d’annotation. Le modèle conserve uniquement l’état (volume de masques, palette BGRA, visibilité). Le contrôleur orchestre désormais la construction RGBA via un service pur Python/NumPy, tout en conservant le flux de push overlay existant vers Endview et VolumeView. Aucun signal/slot PyQt6 modifié.

**Décisions techniques :**
1. Le service accepte `mask_volume` optionnel et renvoie `None` si absent, tout en validant une forme 3D pour éviter des erreurs silencieuses.
2. Palette par défaut dérivée de `MASK_COLORS_BGRA` avec couleur de secours magenta semi-transparente pour tout label non mappé, afin de conserver le comportement historique.
3. Maintien d’un alias `visible_labels` dans le modèle pour compatibilité, avec `get_visible_labels` utilisé côté contrôleur pour expliciter l’intention et réduire le couplage.

---
