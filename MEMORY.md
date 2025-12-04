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

### **2025-12-01** — Nettoyage palette générique et suppression logging_config

**Tags :** `#config/constants.py`, `#config/logging_config.py`, `#palette`, `#overlay`, `#cleanup`, `#branch:annotation`

**Actions effectuées :**
- Supprimé `CLASS_MAP`, `LABEL_SETTINGS` et `LABEL_COLORS_HEX` devenus inutiles; palettes `MASK_COLORS_*` conservées mais commentées de façon générique (labels numériques dynamiques).
- Neutralisé les commentaires des palettes pour retirer toute référence frontwall/backwall et garder un fallback couleur par défaut.
- Supprimé le fichier inutilisé `config/logging_config.py` (aucun import référencé).

**Contexte :**
Les labels sont désormais créés à la volée et ne portent plus de noms fixes; la seule palette requise est `MASK_COLORS_BGRA` (fallback overlay). Le reste des mappings nommés entretenait une dette inutile et ne servait plus.

**Décisions techniques :**
1. Conserver `MASK_COLORS_BGRA` (et variantes) comme palette par défaut/fallback pour l’overlay afin d’éviter un comportement sans couleur si la palette est vide.
2. Retirer les structures de configuration non utilisées et le module de logging inutilisé pour réduire le bruit et les imports potentiels morts.

---

### **2025-12-01** — Squelettes annotation_view / annotation_service et routage ROI vers AnnotationController

**Tags :** `#views/annotation_view.py`, `#services/annotation_service.py`, `#controllers/annotation_controller.py`, `#controllers/master_controller.py`, `#ui_mainwindow.py`, `#untitled.ui`, `#mvc`, `#stubs`, `#branch:annotation`

**Actions effectuées :**
- Créé `AnnotationView` (sous-classe d’EndviewView) avec placeholders pour polygone/rectangle/temp ROI (`set_temp_polygon`, `set_temp_rectangle`, `clear_temp_shapes`, `set_roi_overlay`, `clear_roi_overlay`).
- Ajouté `AnnotationService` stub (compute_threshold, build_roi_mask, apply_label_on_slice, propagate_volume) pour encapsuler la logique ROI/propagation future.
- Étendu `AnnotationController` pour injecter le service + la nouvelle vue, ajouté des handlers stub pour threshold/ROI/dessin (`on_tool_mode_changed`, `on_threshold_*`, `on_roi_*`, événements souris/polygone/rectangle/point) et redirigé l’overlay vers `annotation_view`.
- Reconfiguré `MasterController` : instancie `AnnotationService`, connecte ToolsPanel et signaux de la vue vers les handlers d’AnnotationController (ROI/threshold/dessin), utilise `AnnotationView` pour la fenêtre principale, passe la nouvelle vue aux contrôleurs concernés.
- Mis à jour `ui_mainwindow.py` et `untitled.ui` pour instancier `AnnotationView` au lieu de `EndviewView` dans l’UI Designer.

**Contexte :**
Préparation du refactoring ROI/dessin : les handlers sont déplacés dans `AnnotationController` mais restent vides pour implémentation ultérieure; la nouvelle vue et le service servent de point d’ancrage pour l’affichage ROI et la logique métier. L’intégration UI est basculée sur la sous-classe afin de permettre l’enrichissement futur sans toucher aux autres vues.

**Décisions techniques :**
1. Garder les handlers vides dans `AnnotationController` et conserver l’état UI minimal (tool_mode/threshold/apply/persistence) pour ne pas casser les structures MVC tout en permettant la suite du développement.
2. Sous-classer EndviewView en AnnotationView et mettre à jour l’UI générée pour garantir que toutes les futures fonctionnalités ROI/dessin s’appuient sur la vue dédiée plutôt que d’étendre la base générique.


---

### **2025-12-01** — Export overlay NPZ via AnnotationController

**Tags :** `#services/overlay_export.py`, `#controllers/annotation_controller.py`, `#controllers/master_controller.py`, `#overlay`, `#npz`, `#mvc`, `#branch:main`

**Actions effectuées :**
- Créé `OverlayExport.save_npz` pour valider un volume de masques (3D, uint8, non vide, shape optionnelle) et sauvegarder en `.npz` compressé avec extension auto.
- Injecté `overlay_export` dans `AnnotationController` et ajouté `save_overlay_via_dialog` qui ouvre un QFileDialog, vérifie le shape du masque vs volume NDE, et appelle le service d’export.
- Raccordé `_on_save` dans `MasterController` au menu Overlay > `actionExporter_npz` et délégué l’export au contrôleur d’annotation avec feedback UI (status/erreurs).

**Contexte :**
Le flux de sauvegarde d’overlay passe désormais par l’AnnotationController plutôt que le MasterController direct. Le service dédié centralise la validation et l’écriture NPZ, respectant l’architecture MVC et l’existant overlay_loader/overlay_service. Le menu Fichier>“Sauvegarder” n’est plus utilisé pour l’export, remplacé par Overlay>“Exporter .npz”.

**Décisions techniques :**
1. Service `OverlayExport` dédié pour isoler la logique d’export NPZ et assurer validation de shape avant écriture.
2. Le contrôleur d’annotation reste l’unique point d’accès overlay: ajout d’une méthode dialog pour limiter le couplage UI côté master et réutiliser la même validation.
3. Connexion UI déplacée vers `actionExporter_npz` (menu Overlay) pour aligner le geste utilisateur avec la fonction d’export mask volume.

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

### **2025-11-29** — Pile C-scan construite dans MasterController

**Tags :** `#controllers/master_controller.py`, `#controllers/cscan_controller.py`, `#views/cscan_view_corrosion.py`, `#mvc`, `#pyqt6`, `#layout`

**Actions effectuées :**
- Création du `QStackedLayout` C-scan dans `MasterController` en réparent `frame_4` comme vue standard, ajout d’une nouvelle `CscanViewCorrosion`, et insertion du container dans le splitter pour piloter standard/corrosion sans toucher au contrôleur.
- `CScanController` reçoit désormais `standard_view`, `corrosion_view` et `stacked_layout` injectés, sans dépendance à `ui_mainwindow` ni manipulation du splitter/layout.
- `show_standard` / `show_corrosion` se contentent de sélectionner le widget courant du stack existant, tout en conservant les helpers de crosshair/projection.

**Contexte :**
Objectif d’isoler la plomberie Qt hors du contrôleur pour respecter MVC : le contrôleur orchestre les vues existantes mais ne construit plus les layouts Designer. MasterController instancie et insère les vues C-scan dans un `QStackedLayout` dédié, puis injecte ces références dans le contrôleur.

**Décisions techniques :**
1. Utilisation d’un container `QWidget` parent du splitter pour contenir le `QStackedLayout`, en remplaçant le slot de `frame_4` afin de préserver la hiérarchie UI existante tout en ajoutant la vue corrosion.
2. Le contrôleur reste tolérant si `stacked_layout` ou `corrosion_view` est absent (check None) afin d’éviter des crashes sur des environnements partiels ou des tests headless.

---

### **2025-11-29** — Service de workflow corrosion injecté

**Tags :** `#services/cscan_corrosion_service.py`, `#controllers/cscan_controller.py`, `#controllers/master_controller.py`, `#corrosion`, `#mvc`, `#service-layer`, `#pyqt6`

**Actions effectuées :**
- Ajout de `CorrosionWorkflowService` et `CorrosionWorkflowResult` dans `cscan_corrosion_service.py` : valide volume/masques, contrôle exactement 2 labels visibles, extrait résolutions depuis `NdeModel.metadata`, déduit l’output_dir, lance `CScanCorrosionService.run_analysis` puis `compute_corrosion_projection`, retourne projection/valeurs ou message d’erreur structuré.
- Injection du workflow dans `CScanController` (`corrosion_workflow_service` optionnel) avec fallback auto ; le contrôleur se contente d’orchestrer et d’activer la corrosion via `ViewStateModel` après succès.
- `run_corrosion_analysis` délègue entièrement au service, gère les erreurs via status_callback et désactive corrosion en cas d’échec ; suppression de l’ancien `_extract_resolutions` et de la logique locale (masques, labels, output dir).
- `MasterController` instancie `CScanCorrosionService` + `CorrosionWorkflowService` et les passe à `CScanController` pour éviter toute dépendance au UI layout côté contrôleur.

**Contexte :**
Objectif de déplacer l’orchestration corrosion hors du contrôleur vers un service dédié, pour respecter MVC et centraliser validations/résolutions/output. Le contrôleur garde uniquement le pilotage des vues et de l’état (activation corrosion) sans toucher à la plomberie PyQt.

**Décisions techniques :**
1. Le workflow renvoie un objet résultat structuré (ok/message/projection/value_range) pour une gestion simple côté contrôleur et logging clair.
2. Résolutions cross/ultra récupérées depuis `NdeModel.metadata.dimensions` avec défauts à 1.0 pour robustesse.
3. Injection du même `CScanCorrosionService` dans le workflow et le contrôleur pour éviter des instanciations divergentes et faciliter les tests/mocking.

---

### **2025-11-29** — Extraction overlay vers AnnotationController

**Tags :** `#controllers/annotation_controller.py`, `#controllers/master_controller.py`, `#overlay`, `#mvc`, `#pyqt6`, `#service-layer`

**Actions effectuées :**
- Création d’un `AnnotationController` dédié aux overlays (labels, visibilité, couleurs) utilisant `OverlayService` pour générer l’overlay BGRA et pousser vers Endview/VolumeView, avec sync des OverlaySettings et conversion BGRA/QColor internalisée.
- MasterController instancie et injecte ce contrôleur (annotation_model, view_state_model, overlay_service, endview/volume/overlay_settings views) puis connecte actions/signaux overlay (menu, checkbox, settings events) vers lui.
- Remplacement de tous les `_push_overlay`/handlers overlay du MasterController par `annotation_controller.refresh_overlay` et helpers dédiés (`clear_labels`, `sync_overlay_settings`), suppression des méthodes overlay locales.

**Contexte :**
Alignement MVC : la gestion overlay quitte MasterController pour un contrôleur spécialisé sans logique UI dans le modèle. OverlaySettingsView reste pilotée par ce contrôleur, et le toggle overlay se fait via ViewStateModel + recalcul de l’overlay via OverlayService.

**Décisions techniques :**
1. `refresh_overlay` nettoie d’abord les vues pour éviter les overlays obsolètes puis pousse l’overlay si `show_overlay` est actif, en loggant la forme/dtype.
2. Les signaux OverlaySettings (ajout/couleur/visibilité) et le menu « Paramètres overlay » transitent maintenant par AnnotationController, isolant la plomberie overlay du MasterController.
3. Chargement NDE/NPZ : MasterController s’appuie sur `annotation_controller.clear_labels()/sync_overlay_settings()/refresh_overlay` pour garder palette/visibilité synchronisées après reset ou import de masques.

---

### **2025-11-29** — État de navigation centralisé dans ViewStateModel

**Tags :** `#controllers/master_controller.py`, `#models/view_state_model.py`, `#mvc`, `#state-management`

**Actions effectuées :**
- Supprimé le champ privé `_current_point` de `MasterController` et remplacé son usage par `view_state_model.current_point` via les méthodes existantes.
- Nettoyé les assignations inutiles (reset et update) afin de conserver ViewStateModel comme source de vérité pour la sélection/crosshair.

**Contexte :**
Objectif d’alléger le contrôleur et de centraliser l’état de navigation/sélection dans `ViewStateModel`, déjà responsable des bornes de slice et du crosshair.

**Décisions techniques :**
1. Pas de nouvelle API nécessaire dans `ViewStateModel`, les setters existants (`set_current_point`, `update_crosshair`) couvrent les usages.
2. `MasterController._update_ascan_trace` ne recopie plus l’état localement, évitant toute divergence avec `ViewStateModel`.

---

### **2025-11-29** — Renommage OverlayDebugLogger

**Tags :** `#services/overlay_debug_logger.py`, `#services/overlay_loader.py`, `#overlay`, `#logging`

**Actions effectuées :**
- Renommé le service de debug NPZ en `overlay_debug_logger.py` avec la classe `OverlayDebugLogger` et l’instance globale `overlay_debug_logger`, les sessions écrivent désormais dans `overlay_debug_log.txt` avec un header overlay explicite.
- La méthode de traçage de chargement devient `log_overlay_loading` (paramètre `overlay_path`), conservant le log des shapes/dtypes et la déduplication des événements.
- `OverlayLoader` importe le nouveau module/logger et utilise `log_overlay_loading` + `log_variable` pour tracer le chargement et les classes uniques.

**Contexte :**
Alignement du logger avec son usage overlay (NPZ/NPY) pour éviter la confusion avec l’ancien nom NPZ et rester cohérent avec l’OverlayLoader qui s’appuie sur ce service pour diagnostiquer les volumes de masques.

**Décisions techniques :**
1. Conserver le pattern singleton et l’API existante en ne changeant que les noms/logfile afin de limiter l’impact au seul OverlayLoader.
2. Renommer le fichier de log en `overlay_debug_log.txt` pour que le nom reflète la portée (overlay) et faciliter le tri des traces aux côtés des autres fichiers de debug (ascan/cscan/etc.).

---

### **2025-11-29** — Overlay alpha compact + refresh sans reset

**Tags :** `#controllers/annotation_controller.py`, `#services/overlay_service.py`, `#views/volume_view.py`, `#views/endview_view.py`, `#models/overlay_data.py`, `#overlay`, `#performance`, `#mvc`

**Actions effectuées :**
- Ajout de `OverlayData` (mask uint8 filtré + alpha float32 + palette BGRA) pour transporter l’overlay sans volume RGBA complet.
- `OverlayService.build_overlay_data` retourne ce payload compact en appliquant la visibilité sur le mask et en mappant l’alpha depuis la palette (sans créer de RGBA (Z,H,W,4)).
- `AnnotationController.refresh_overlay` ne fait plus de reset `set_overlay(None)` avant push ; il ne nettoie que si l’overlay est absent/masqué et pousse `OverlayData` aux vues.
- `VolumeView.set_overlay` consomme `OverlayData` (alpha pour le VolumeVisual, palette pour coloriser la slice) et met à jour les visuels sans recréer, avec helper de clear centralisé.
- `EndviewView.set_overlay` accepte `OverlayData`, génère la slice RGBA à la volée via la palette (plus de stockage d’un volume RGBA complet).

**Contexte :**
Réduire les copies et uploads GPU du volume overlay 3D : le pipeline ne transmet plus un RGBA volumique, évite le double passage `None → set_overlay`, et met à jour les visuels existants. La colorisation par palette reste en vue 2D/3D slice, mais le 3D utilise uniquement l’alpha et un colormap léger.

**Décisions techniques :**
1. Overlay 3D = volume alpha float32 + colormap `_TranslucentMask` (pas de RGBA envoyé au GPU) ; slice overlay colorisée via palette côté vue pour garder les couleurs labels sans volume RGBA.
2. Contrôleur overlay reste orchestration pure : pas de logique UI/métier, pas de reset inutile ; le service gère la visibilité et la génération du payload compact.
3. Nettoyage overlay factorisé `_clear_overlay_visuals` pour éviter les visuels orphelins lors des changements d’overlay/volume ou des entrées invalides.

**Implémentation (extrait) :**
```python
# services/overlay_service.py
alpha = np.zeros_like(filtered_mask, dtype=np.float32)
for cls_value in np.unique(filtered_mask):
    cls_int = int(cls_value)
    if cls_int == 0:
        continue
    a = float(palette.get(cls_int, FallbackColor)[3]) / 255.0
    if a <= 0.0:
        continue
    alpha[filtered_mask == cls_int] = a
return OverlayData(mask=filtered_mask, alpha=alpha, palette=palette)
```

---

### **2025-11-29** — Fix asarray(copy) crash on overlay load

**Tags :** `#views/volume_view.py`, `#overlay`, `#numpy`, `#bugfix`

**Actions effectuées :**
- Remplacé `np.asarray(..., dtype=np.float32, copy=False)` par `np.array(..., dtype=np.float32, copy=False)` dans `VolumeView.set_overlay` pour supporter les versions de numpy où `asarray` n’accepte pas `copy`.

**Contexte :**
Chargement d’NPZ déclenchait `asarray() got an unexpected keyword argument 'copy'` lors de la conversion du volume alpha overlay; utiliser `np.array` supprime l’argument incompatible tout en conservant la conversion float32 sans copie inutile.

**Décisions techniques :**
1. Conversion alpha via `np.array(..., copy=False)` pour rester zéro-copy quand possible et compatible avec numpy <1.26.
2. Aucun impact sur le pipeline overlay (payload compact mask+alpha+palette) ; uniquement la conversion d’entrée côté vue 3D.

---

### **2025-11-30** — OverlayService perf: filtrage sans copie globale

**Tags :** `#services/overlay_service.py`, `#overlay`, `#performance`, `#numpy`, `#mvc`

**Actions effectuées :**
- `build_overlay_data` génère désormais un volume alpha par label pour tous les labels présents (pas de filtrage de visibilité dans le service).
- Suppression du paramètre `visible_labels` pour clarifier que le filtrage se fait côté vues.

**Contexte :**
Le service fournit un overlay complet (tous labels) et laisse les vues gérer la visibilité pour éviter toute ambiguïté de responsabilité.

**Décisions techniques :**
1. Responsabilité unique : service = données complètes; vues = filtrage des labels visibles.
2. Moins de confusion API en enlevant `visible_labels`.

---

### **2025-11-30** — Déferlement overlay 3D pour toggles rapides

**Tags :** `#views/volume_view.py`, `#controllers/annotation_controller.py`, `#overlay`, `#performance`, `#vispy`, `#qt`

**Actions effectuées :**
- VolumeView : ajout d’un `defer_3d` dans `set_overlay`, avec QTimer (120ms) pour coalescer les uploads du VolumeVisual overlay ; mise à jour immédiate de l’overlay 2D, upload 3D différé via `_apply_pending_overlay_volume`, clear/stop du timer sur reset, application forcée si on change de slice pendant un upload en attente.
- AnnotationController : les interactions OverlaySettings (visibility/color/add) appellent `refresh_overlay(defer_volume=True)`, qui pousse Endview immédiatement et VolumeView avec defer pour éviter le spam GPU ; les autres chemins (toggle global) restent immédiats.

**Contexte :**
Les cases de visibilité étaient lentes car chaque clic re-uploadait le volume overlay 3D. Le différé limite les uploads pendant les toggles rapides tout en gardant la slice 2D à jour.

**Décisions techniques :**
1. Timer single-shot 120ms pour regrouper les modifications, flag `_pending_overlay_apply` pour savoir si un upload est en attente.
2. Application forcée avant un changement de slice pour garder 3D/slice cohérents.
3. Chemins overlay_settings seuls passent en defer pour ne pas changer le comportement des autres toggles (on/off global).

**Implémentation (extrait) :**
```python
# views/volume_view.py
self._overlay_timer = QTimer(self)
self._overlay_timer.setSingleShot(True)
self._overlay_timer.timeout.connect(self._apply_pending_overlay_volume)
...
def set_overlay(self, overlay, *, defer_3d=False):
    ...
    if defer_3d:
        if self._slice_overlay is None and self._view.scene is not None:
            self._add_slice_overlay()
        self._update_overlay_image()
        self._schedule_overlay_volume_update()
        return
    self._apply_overlay_volume_now()

# controllers/annotation_controller.py
self.refresh_overlay(defer_volume=True)
...
self.volume_view.set_overlay(overlay_data, defer_3d=defer_volume)
```

---

### **2025-11-30** — Overlays par label en mémoire + toggles visibles sans reupload

**Tags :** `#services/overlay_service.py`, `#controllers/annotation_controller.py`, `#views/volume_view.py`, `#views/endview_view.py`, `#overlay`, `#performance`, `#vispy`, `#qt`

**Actions effectuées :**
- `OverlayData` stocke désormais un volume alpha par label (`label_volumes`), palette séparée.
- `OverlayService.build_overlay_data` génère un alpha volume float32 pour chaque label présent (tous labels), sans filtrage par visibilité ; retourne None si aucun label.
- `AnnotationController` maintient un cache OverlayData et, pour les toggles couleurs/visibilité, réutilise les volumes en mémoire (rebuild=False) en ne poussant que palette/visibilité vers les vues ; passe `visible_labels` aux vues ; log mis à jour (nombre de labels).
- `VolumeView` gère des VolumeVisual par label (colormap 2 stops couleur du label) et ne réuploade les données que si le volume a changé ; les toggles ne modifient que `visible`; overlay 2D composé par slice en combinant les volumes visibles ; support du defer (timer 120ms) conservé mais appliqué aux volumes multi-label.
- `EndviewView` compose la slice overlay via les volumes par label et les labels visibles, sans recomposer un volume RGBA complet stocké.

**Contexte :**
Objectif d’avoir un volume par label gardé en mémoire et rendre les toggles de visibilité quasi instantanés en évitant les reuploads complets ; cache OverlayData pour réutiliser les volumes existants quand seule la visibilité ou la palette change.

**Décisions techniques :**
1. Volume par label avec colormap dédiée pour conserver la couleur sans recalcul des données ; visibilité gérée au niveau du visual.
2. Cache overlay au contrôleur pour différencier les cas “masque changé” (rebuild) vs “visibilité/couleur” (réutilisation + palette/visibilité).
3. Composition 2D à la volée par slice (combine les volumes visibles) pour rester cohérent avec la vue 3D sans stocker un RGBA volumique.

**Implémentation (extrait) :**
```python
# controllers/annotation_controller.py
if not rebuild and self._overlay_cache is not None:
    overlay_data = OverlayData(label_volumes=self._overlay_cache.label_volumes, palette=palette)
else:
    overlay_data = self.overlay_service.build_overlay_data(mask_volume, palette)
...
self.volume_view.set_overlay(overlay_data, visible_labels=visible_labels, defer_3d=defer_volume)

# views/volume_view.py (colormap par label)
return Colormap(colors=[(0,0,0,0), (r/255.0, g/255.0, b/255.0, a/255.0)])
```

---

### **2025-11-30** — Log overlay counts (mask/palette/visible)

**Tags :** `#controllers/annotation_controller.py`, `#overlay`, `#logging`

**Actions effectuées :**
- Remplacé le log d’envoi overlay pour afficher trois compteurs: labels présents dans le masque (`mask_labels`), labels dans la palette, et nombre de labels visibles (ou "all").
- Supprimé le message trompeur qui indiquait toujours `labels=2` même quand plus de labels existaient.

**Contexte :**
Le précédent log ne comptait que `overlay_data.label_volumes` (labels avec des pixels), ce qui restait souvent à 2 malgré l’ajout de nouveaux labels. Le nouveau message clarifie les volumes effectivement poussés (masque), la palette définie et l’ensemble visible, pour diagnostiquer correctement l’overlay.

**Décisions techniques :**
1. `mask_labels` = `len(overlay_data.label_volumes)` pour refléter uniquement les labels ayant des pixels.
2. `palette` = `len(palette)` pour suivre les labels définis côté modèle.
3. `visible` = `len(visible_labels)` ou `"all"` si aucun filtrage, pour éviter la confusion lorsque la visibilité est totale.

**Implémentation (extrait) :**
```python
mask_label_count = len(overlay_data.label_volumes)
palette_count = len(palette)
visible_count = len(visible_labels) if visible_labels is not None else palette_count
self.logger.info(
    "Pushing overlay to views | mask_labels=%d | palette=%d | visible=%s",
    mask_label_count,
    palette_count,
    visible_count if visible_labels is not None else "all",
)
```

---

### **2025-11-30** — Reset overlay state on NDE load

**Tags :** `#controllers/master_controller.py`, `#controllers/annotation_controller.py`, `#overlay`, `#reset`

**Actions effectuées :**
- Ajout d’une méthode `reset_overlay_state` dans `AnnotationController` pour vider le cache overlay, nettoyer les vues (Endview/Volume) et réinitialiser la fenêtre de paramètres overlay.
- Appel de `reset_overlay_state` lors du chargement d’un nouveau NDE et lors du chargement d’un overlay NPZ afin d’éviter la réutilisation d’overlays de taille incompatible.

**Contexte :**
Un overlay mis en cache sur un précédent volume causait un crash (mismatch de shapes) après chargement d’un nouveau NDE puis toggle overlay. Il fallait repartir d’un état propre avant de reconstruire les overlays.

**Décisions techniques :**
1. Centraliser le reset overlay dans le contrôleur pour éviter la duplication et garantir que cache + vues sont vidés.
2. Appeler le reset avant de réinitialiser les masques/labels lors des chargements NDE ou NPZ afin que tout nouveau build d’overlay reparte du volume courant.

---

### **2025-12-01** — Zoom molette C-scan

**Tags :** `#views/cscan_view.py`, `#views/cscan_view_corrosion.py`, `#ui`, `#zoom`, `#mvc`

**Actions effectuées :**
- Ajout d’un zoom à la molette sur CScanView avec ancre sous la souris (`ViewportAnchor.AnchorUnderMouse`) et facteur 1.15/-1.15.
- Réinitialisation du transform du QGraphicsView lors de `set_projection` pour repartir à échelle 1 après changement de projection; CscanViewCorrosion hérite automatiquement.

**Contexte :**
Besoin de zoomer sur la heatmap C-scan sans casser la sélection existante au Shift-clic. L’interaction reste gérée dans la vue (UI) et se limite au scaling du QGraphicsView.

**Décisions techniques :**
1. Gestion du wheel event via l’eventFilter pour garder la logique d’interaction encapsulée dans la vue et préserver le Shift-clic existant.
2. Reset du transform à chaque nouvelle projection afin d’éviter la persistance d’un zoom précédent qui pourrait induire une mauvaise lecture de la nouvelle image.

---

### **2025-12-01** — Chunked nnUNet inference (8 parts)

**Tags :** `#services/nnunet_service.py`, `#plugins/segmentation_hooks/segmentation_plugins/nnunetv2_plugin.py`, `#nnunet`, `#segmentation`, `#batching`

**Actions effectuées :**
- Ajout d’un paramètre `chunk_parts` (défaut 8, validé >0) dans `NnUnetService.run_inference` et passage dans `PipelineInput.config` pour piloter le découpage en Z des inférences nnUNet.
- `nnunetv2_preprocess` découpe désormais le volume en `chunk_parts` blocs contigus sur l’axe Z via `np.array_split(..., axis=1)` pour les modèles 2D et 3D, en conservant l’orientation existante et en ajoutant les métadonnées `part_index/num_parts/original_shape`.
- `nnunetv2_inference_iter` recolle les prédictions chunkées, pad si le dernier bloc est plus court, concatène, puis recadre la profondeur finale sur celle du volume d’entrée pour garantir la forme du masque.

**Contexte :**
Le temps d’inférence est réduit en limitant les appels nnUNet (8 blocs au lieu d’un par slice), en s’alignant sur le comportement Sentinel observé dans les logs. L’orientation est inchangée (pas de transpose XY supplémentaire), et le chunking s’applique aussi aux modèles 3D pour maîtriser VRAM/latence. Le paramètre reste configurable via la config du service.

**Décisions techniques :**
1. Valeur par défaut `chunk_parts=8` exposée côté service pour pouvoir désactiver/ajuster sans changer les plugins, avec garde-fou `>0`.
2. Préprocess et inference partagent la profondeur attendue (`sinp.data_array.shape[0]`) et recadrent le masque final pour éviter les divergences de shape liées aux splits ou au padding du dernier bloc.

---

### **2025-12-01** — Ajout règle de tag de branche dans AGENTS.md

**Tags :** `#AGENTS.md`, `#MEMORY.md`, `#documentation`, `#tagging`

**Actions effectuées :**
- Ajout d’une sous-section \"Branches Git\" dans le système de tagging d’`AGENTS.md` décrivant le format de tag `#branch:<nom_de_branche_git>` avec exemples (`#branch:main`, `#branch:feature/...`).
- Mise à jour des règles de tagging pour rendre obligatoire l’ajout d’un tag de branche pour chaque nouvelle entrée de mémoire (ByteRover + `MEMORY.md`).

**Contexte :**
On veut pouvoir rattacher chaque entrée de mémoire à la branche Git courante pour faciliter le debug, l’investigation d’historique et la compréhension du contexte de développement. Centraliser cette règle dans `AGENTS.md` garantit que tous les agents suivent la même convention lorsqu’ils documentent les tâches terminées.

**Décisions techniques :**
1. Utiliser un tag textuel standardisé `#branch:<nom_de_branche_git>` plutôt qu’un champ séparé, afin de rester compatible avec le système existant de tags plein-texte dans ByteRover et `MEMORY.md`.
2. Limiter la modification à `AGENTS.md` (section Système de Tagging + règles de tagging) pour ne pas casser les templates existants, tout en rendant la règle explicite et non ambiguë pour les futures entrées de mémoire.

---

### **2025-12-01** — Ajout modèles ROI/temp et préview rectangle

**Tags :** `#models/roi_model.py`, `#models/temp_mask_model.py`, `#services/annotation_service.py`, `#controllers/annotation_controller.py`, `#controllers/master_controller.py`, `#views/annotation_view.py`, `#roi`, `#overlay`, `#annotation`, `#branch:annotation`

**Actions effectuées :**
- Créé `RoiModel` (stocke ROIs typés rectangle/polygon/grow avec id, slice, points, label, threshold, persistance) et `TempMaskModel` (volume masque temporaire similaire à AnnotationModel, palette BGRA, visibilités, set_slice_mask avec option persistent/remplacement).
- Étendu `AnnotationService` avec `build_rectangle_mask` (deux coins → masque binaire) et `apply_temp_rectangle` pour appliquer une ROI rectangulaire dans `TempMaskModel`.
- Implémenté `on_annotation_rectangle_drawn` dans `AnnotationController`: normalise le rect, construit le masque rectangle via service, enregistre la ROI dans `RoiModel`, applique la preview dans `TempMaskModel`, et pousse la slice temporaire vers `AnnotationView.set_roi_overlay`. Reset overlay state nettoie aussi ROI/temp.
- `AnnotationController` prend désormais `roi_model` et `temp_mask_model` en dépendances.
- `MasterController` instancie/initialise `RoiModel` et `TempMaskModel`, les passe au contrôleur, et les reset lors du chargement NDE/overlay.
- `AnnotationView` affiche un overlay ROI dédié (nouvel item QGraphicsPixmapItem opacité 0.35, couleur label 1, conversion masque→pixmap), avec clear sécurisé.

**Contexte :**
Première implémentation de ROI rectangle avec prévisualisation séparée du volume overlay. Les ROIs sont stockées à part et les masques temporaires servent de preview avant fusion dans l’overlay principal. La vue d’annotation dispose maintenant d’un canal visuel pour ces previews.

**Décisions techniques :**
1. Séparer stockage overlay (AnnotationModel) et preview (TempMaskModel) pour éviter d’altérer le volume tant que l’utilisateur n’a pas validé.
2. Centraliser les métadonnées ROI dans `RoiModel` (type/slice/points/label/threshold/persistence) afin de supporter d’autres formes (polygon/grow) ultérieurement.
3. Utiliser une couleur de palette existante (label 1) pour le ROI overlay et un item graphique dédié pour ne pas perturber l’overlay principal.

---

### **2025-12-01** — Fix rectangle ROI crash quand le volume masque est absent

**Tags :** `#controllers/annotation_controller.py`, `#roi`, `#annotation`, `#branch:annotation`

**Actions effectuées :**
- Corrigé `on_annotation_rectangle_drawn` pour ne plus évaluer un `ndarray` dans un `or` (ambiguïté). On récupère d’abord le masque de `AnnotationModel`, sinon celui de `TempMaskModel`, et on quitte si aucun volume n’est disponible.

**Contexte :**
Le dessin rectangle plantait avec un `ValueError` (truth value of an array…) car l’expression `mask_a or mask_b` tentait d’évaluer la vérité d’un `ndarray`.

**Décisions techniques :**
1. Vérification explicite `None` pour les volumes masque avant usage, afin d’éviter l’ambiguïté des `ndarray`.

### **2025-12-01** — Label actif dans le tools panel, couleur du polygone par label

**Tags :** `#views/tools_panel.py`, `#controllers/master_controller.py`, `#controllers/annotation_controller.py`, `#models/view_state_model.py`, `#roi`, `#labels`, `#ui`, `#branch:annotation`

**Actions effectuées :**
- ToolsPanel : ajout du scroll de labels avec sélection (signal `label_selected`), `set_labels` conserve la sélection et accepte un label courant.
- ViewStateModel : nouvel attribut `active_label` + setter.
- MasterController : synchronisation des labels du tools panel avec la palette de l’annotation model, sélection du label actif, mise à jour après ajout de label via la fenêtre overlay.
- AnnotationController : les ROI utilisent `active_label` pour les masques temporaires ; la couleur du polygone temporaire est dérivée de la palette (fallback `MASK_COLORS_BGRA`). Correction du crash `clear_roi_rectangle` -> `clear_roi_rectangles`.

**Contexte :**
Permettre à l’utilisateur de choisir un label dans le tools panel (issu de la palette overlay) et d’appliquer ce label lors du dessin ROI/polygone, avec la couleur associée.

**Décisions techniques :**
1. Centraliser l’état du label actif dans `ViewStateModel` et le synchroniser avec l’UI des labels.
2. Repasser par la palette des labels pour colorer le polygone temporaire (fallback sur `MASK_COLORS_BGRA` si non défini).

---

### **2025-12-02** — Session ROI/labels (extraction métier, palettes, preview)

**Tags :** `#controllers/annotation_controller.py`, `#services/annotation_service.py`, `#models/roi_model.py`, `#models/temp_mask_model.py`, `#models/annotation_model.py`, `#models/view_state_model.py`, `#views/tools_panel.py`, `#views/annotation_view.py`, `#overlay`, `#roi`, `#labels`, `#mvc`, `#branch:annotation`

**Actions effectuées :**
- Déplacé la logique rectangle dans `AnnotationService.apply_rectangle_roi` (normalisation, masque, enregistrement ROI, application TempMaskModel) et `rebuild_temp_masks_for_slice` (rebuild complet du masque temp depuis les ROI d’une slice).
- `RoiModel` expose `rectangles_for_slice` et gère le filtrage slice/persistant ; le contrôleur ne filtre plus.
- `AnnotationModel`/`TempMaskModel` fournissent `mask_shape_hw`; `TempMaskModel.clear` remet le volume à zéro sans effacer palette/visibilités et conserve la palette à l’init ; ajout de `set_label_color`.
- Palette par défaut retirée : aucun label pré-créé, `active_label` peut être None ; `overlay_service` n’injecte la palette par défaut que si aucune n’est fournie.
- Outil labels : ToolsPanel avec scroll de labels et signal `label_selected`; MasterController synchronise la palette vers le ToolsPanel (sans labels par défaut), sélectionne le label actif à l’ajout et reflète couleurs/visibilités. Les ROI utilisent le label actif ; la couleur du polygone temporaire vient de la palette (fallback BGRA si couleur manquante).
- Vue/preview : AnnotationView gère plusieurs rectangles, masque temp colorisé par label, clears sécurisés ; refresh slice affiche masque temp + rectangles (slice ou persistants).
- Nettoyages : helpers de filtrage ROI supprimés du contrôleur ; `_normalize_rect_input` dans le service.

**Contexte :**
Refactor ROI/labels pour séparer métier (services/models) du contrôleur, retirer les labels par défaut et aligner la couleur du polygone temporaire sur le label choisi. Permet plusieurs ROI/polygones, persistance, et sélection de label via le ToolsPanel.

**Décisions techniques :**
1. Centraliser la construction/apply masque dans le service pour éviter la duplication et respecter MVC.
2. Ne pas pré-créer de labels ; l’utilisateur ajoute via Paramètres overlay, ce label devient actif et colorise la preview.
3. Conserver la palette du TempMaskModel lors des reset pour garder les couleurs cohérentes après suppression/rebuild.

---

### **2025-12-03** — Injection slice volume pour seuil ROI

**Tags :** `#controllers/annotation_controller.py`, `#controllers/master_controller.py`, `#services/annotation_service.py`, `#roi`, `#threshold`, `#overlay`, `#branch:annotation`

**Actions effectuées :**
- AnnotationController accepte un getter de volume (`get_volume`) et utilise `_slice_data` pour passer la slice brute au service lors du rebuild/apply ROI, afin de supporter le seuillage local.
- AnnotationService reçoit `slice_data` pour `rebuild_temp_masks_for_slice`/`apply_rectangle_roi`, applique un masque seuillé via `build_thresholded_mask` (normalisation locale 0-255 sur la zone rectangulaire) quand un threshold est défini.
- MasterController injecte `_current_volume` dans AnnotationController pour que les previews ROI utilisent les données NDE existantes.
- Les masques rectangles appliqués aux temp masks utilisent désormais le masque seuillé quand disponible, sinon restent inchangés.

**Contexte :**
Le preview ROI devait respecter le threshold saisi : le service normalise les valeurs de la zone rectangle et applique le seuil pour ne pousser que les pixels retenus dans le masque temporaire. Le contrôleur a besoin d’accès au volume pour fournir ces données slice-level.

**Décisions techniques :**
1. Dériver `slice_data` via un getter optionnel pour ne pas coupler le contrôleur à un volume global (fallback None sans crash).
2. Normaliser localement (min/max de la zone) avant seuil pour supporter des plages arbitraires et ne pas dépendre de l’échelle globale du volume.

---

### **2025-12-03** — Navigation ROI + appli volume

**Tags :** `#controllers/master_controller.py`, `#controllers/annotation_controller.py`, `#views/endview_view.py`, `#views/annotation_view.py`, `#views/tools_panel.py`, `#ui_mainwindow.py`, `#untitled.ui`, `#roi`, `#shortcuts`, `#navigation`, `#branch:annotation`

**Actions effectuées :**
- Ajout boutons previous/next et bouton Appliquer ROI dans l’UI (Designer + ToolsPanel wiring) + labels radio renommés (Free Hand/Box/Germ).
- MasterController enregistre des raccourcis globaux (A/D navigation slices, W appliquer ROI, Escape annuler ROI) et connecte boutons/signaux de l’annotation_view/tools_panel aux handlers de navigation/apply.
- AnnotationController ajoute `on_apply_temp_mask_requested` pour fusionner le masque temporaire (ROI/polygone) dans `AnnotationModel`, propager palette/visibilités, clear preview puis rafraîchir overlay/ROI.
- AnnotationView et EndviewView améliorent l’ergonomie : coords clampées au volume, focus sur interaction, navigation/soumission via signaux, panning au bouton droit, zoom à la molette avec ancre sous la souris (drag désactivé par défaut).
- `_navigate_slice_delta` centralise la navigation slice et supporte boutons/shortcuts.

**Contexte :**
Les ROI devaient être validées dans le volume et naviguées rapidement. Les raccourcis/boutons facilitent le flux (A/D/W/Escape), et l’application du masque temporaire construit une slice mise à jour dans le modèle avant refresh overlay. Les vues gagnent pan/zoom cohérents et évitent de sortir du volume.

**Décisions techniques :**
1. Appliquer le masque temporaire par copie de slice pour éviter les effets de bord et conserver la palette dans le modèle, puis nettoyer la preview pour ne pas laisser d’artefacts.
2. Centraliser les raccourcis côté master pour qu’ils soient actifs partout dans la fenêtre (ApplicationShortcut) et réutiliser les mêmes handlers que l’UI.
3. Clamper les coordonnées souris et supporter le panning droit pour éviter les exceptions hors volume et améliorer la navigation dans l’endview.

---

### **2025-12-03** — Renommage temp polygon en temp mask (ROI)

**Tags :** `#views/annotation_view.py`, `#services/annotation_service.py`, `#roi`, `#temp_mask`, `#naming`, `#branch:annotation`

**Actions effectuées :**
- Renommé le placeholder de polygone temporaire en masque temporaire dans `AnnotationView` (`_temp_mask_points`, `set_temp_mask`, docstrings/clear) pour refléter le mask ROI.
- Mis à jour la docstring de seuillage dans `AnnotationService` pour parler de "temp mask" au lieu de "temp polygon".

**Contexte :**
Clarifier la terminologie autour des préviews ROI : on manipule des masques temporaires appliqués depuis les ROI, pas des polygones persistants. L’objectif est d’éviter les confusions avec le modèle de masque temporaire lors du câblage contrôleur/service.

**Décisions techniques :**
1. Limiter le changement à l’API de vue et aux textes pour aligner le nommage sans modifier la logique existante.
2. Conserver la structure MVC intacte (vue uniquement pour l’UI, service pour la doc) afin d’éviter tout impact fonctionnel non souhaité.

---

### **2025-12-03** — Renommage ROI polygon → free hand

**Tags :** `#views/tools_panel.py`, `#controllers/master_controller.py`, `#controllers/annotation_controller.py`, `#models/roi_model.py`, `#roi`, `#tooling`, `#naming`, `#branch:annotation`

**Actions effectuées :**
- ToolsPanel émet désormais le mode "free_hand" (radio Free Hand), mapping interne renommé et `select_tool_mode` aligne le nouvel identifiant.
- MasterController attache le radio Designer via l’argument `free_hand_radio` et propage le mode "free_hand" au contrôleur.
- Documentation/typage alignés : `roi_type` mentionne "free_hand" au lieu de "polygon", docstrings du contrôleur mis à jour pour refléter le free-hand.

**Contexte :**
Harmoniser la terminologie avec l’UI (bouton "Free Hand") et l’état ViewStateModel, pour éviter de mélanger l’ancien nom "polygon" avec le mode main levée utilisé pour les ROI.

**Décisions techniques :**
1. Limiter le changement au nommage de mode et au wiring pour ne pas toucher aux signaux bas niveau (polygon_started, etc.) afin d’éviter une refonte plus large.
2. Conserver la logique existante et uniquement clarifier l’API/état ; aucune modification comportementale ou calculatoire.

---

### **2025-12-03** — Handlers renommés free hand

**Tags :** `#controllers/annotation_controller.py`, `#controllers/master_controller.py`, `#roi`, `#tooling`, `#naming`, `#free_hand`, `#branch:annotation`

**Actions effectuées :**
- Renommé les handlers du contrôleur d’annotation pour le dessin free hand (`on_annotation_freehand_started/point_added/completed`) et mis à jour la docstring d’application de masque temporaire.
- MasterController connecte désormais les signaux `polygon_started/point_added/completed` de la vue vers les handlers `freehand_*` (pas de changement côté signaux de la vue).

**Contexte :**
Aligner le nommage des handlers avec le mode "free_hand" tout en conservant les signaux existants de la vue pour limiter l’ampleur du refactor.

**Décisions techniques :**
1. Changer uniquement les noms de handlers et le câblage côté contrôleur pour éviter de renommer les signaux de vue (limite de surface de changement).
2. Garder le comportement identique ; aucune modification de logique fonctionnelle.

---

### **2025-12-03** — Signaux free hand (remplacement polygon_*)

**Tags :** `#views/endview_view.py`, `#controllers/master_controller.py`, `#controllers/annotation_controller.py`, `#roi`, `#signals`, `#free_hand`, `#branch:annotation`

**Actions effectuées :**
- Renommé les signaux de dessin polygon_* en freehand_* dans `EndviewView`.
- Mis à jour le câblage dans `MasterController` vers les handlers freehand du contrôleur d’annotation.
- Maintenu les handlers freehand existants (introduits précédemment) pour rester cohérents avec le mode "free_hand".

**Contexte :**
Finaliser le renommage "polygon" → "free_hand" côté signaux d’interaction, afin que l’UI et les handlers partagent la même terminologie.

**Décisions techniques :**
1. Changer uniquement les noms de signaux et leur câblage ; pas de modification des signaux émis ailleurs pour éviter des changements plus larges.
2. Conserver la structure MVC et les stubs en l’état, la logique fonctionnelle reste inchangée.

---

### **2025-12-03** — AnnotationService: paramètres free hand

**Tags :** `#services/annotation_service.py`, `#free_hand`, `#roi`, `#naming`, `#branch:annotation`

**Actions effectuées :**
- Renommé les paramètres de polygone en `free_hand_points` dans `compute_threshold` et `build_roi_mask` pour aligner la terminologie free hand.
- Supprimé les dernières occurrences du terme "polygon" dans AnnotationService.

**Contexte :**
Suite au renommage global polygon → free hand, les stubs du service devaient refléter la même terminologie pour éviter toute confusion lors des futurs appels.

**Décisions techniques :**
1. Changement limité aux signatures/nommage (stubs) sans modifier le comportement, afin de préserver les contrats existants tout en clarifiant l’API.
2. Garder les fonctions comme placeholders (return None) en attendant l’implémentation métier.

---

### **2025-12-03** — Mode ROI point renommé grow

**Tags :** `#views/tools_panel.py`, `#controllers/master_controller.py`, `#roi`, `#grow`, `#tooling`, `#naming`, `#branch:annotation`

**Actions effectuées :**
- ToolsPanel émet désormais le mode "grow" (bouton point renommé en grow côté code), mapping interne et `select_tool_mode` mis à jour.
- MasterController attache le radio Designer via l’argument `grow_radio` pour propager le mode grow au contrôleur.

**Contexte :**
Aligner la terminologie ROI de type point vers "grow" pour refléter le comportement attendu (germination) et rester cohérent avec les autres renommages.

**Décisions techniques :**
1. Changement limité au nommage/mapping des radios et du mode émis, sans toucher aux autres logiques ou signaux.
2. Conserver la structure MVC et le wiring existant ; aucune modification fonctionnelle au-delà du nommage.

---

### **2025-12-03** — ROI rectangle renommé box

**Tags :** `#views/tools_panel.py`, `#controllers/master_controller.py`, `#models/roi_model.py`, `#services/annotation_service.py`, `#roi`, `#box`, `#tooling`, `#naming`, `#branch:annotation`

**Actions effectuées :**
- ToolsPanel émet désormais le mode "box" (radio rectangle renommé côté code) et `select_tool_mode` mappe l’identifiant `box`.
- MasterController passe le radio Designer en paramètre `box_radio` pour propager le mode box au contrôleur.
- RoiModel stocke le type ROI "box" (au lieu de "rectangle") et `AnnotationService.rebuild_temp_masks_for_slice` filtre désormais sur `roi_type == "box"` pour les ROI rectangulaires.

**Contexte :**
Aligner la terminologie ROI sur "box" pour remplacer l’ancien nom "rectangle", en cohérence avec les autres renommages (free_hand, grow) sans modifier la logique géométrique.

**Décisions techniques :**
1. Renommer uniquement le type/identifiant et le mapping d’outil ; les fonctions métiers restent nommées rectangle (mask, apply) pour limiter la surface de changement.
2. Garantir que la reconstruction des masques temporaires filtre sur le nouveau type "box" afin de ne pas ignorer les ROI rectangulaires existantes dans le nouveau schéma.

---

### **2025-12-03** — ROI rectangle → box (signals, modèles, services)

**Tags :** `#views/annotation_view.py`, `#views/endview_view.py`, `#views/tools_panel.py`, `#controllers/master_controller.py`, `#controllers/annotation_controller.py`, `#models/roi_model.py`, `#services/annotation_service.py`, `#roi`, `#box`, `#naming`, `#signals`, `#branch:annotation`

**Actions effectuées :**
- Signaux de dessin rectangle renommés en `box_drawn` (EndviewView) et raccordés côté MasterController aux handlers `on_annotation_box_drawn`.
- AnnotationView : API box complète (`set_temp_box`, `set_roi_boxes`, `clear_roi_boxes`, handlers `_handle_box_*`, émission `box_drawn`).
- RoiModel : ROI type `box` et méthodes `add_box` / `boxes_for_slice` (filtre sur `roi_type == "box"`).
- AnnotationService : API box (`build_box_mask`, `apply_temp_box`, `apply_box_roi`, `_normalize_box_input`), rebuild des masques temporaires filtrant sur `roi_type == "box"` et renvoyant les boxes.
- ToolsPanel/MasterController : radio Designer passé en `box_radio` et `tool_mode_changed` émet `box`.
- AnnotationController : handler `on_annotation_box_drawn`, usage de `apply_box_roi`, rafraîchissements via `boxes_for_slice` et `set_roi_boxes`, nettoyage `clear_roi_boxes`.

**Contexte :**
Uniformiser la terminologie ROI en remplaçant "rectangle" par "box" sur tout le flux (vue, signaux, contrôleur, modèle ROI, service) après le renommage des autres modes (free_hand, grow).

**Décisions techniques :**
1. Renommer signaux/APIs/méthodes liées aux ROI rectangulaires plutôt que laisser des alias pour éviter la confusion avec les modes UI ; conserver les objets graphiques internes (QGraphicsRectItem) pour le rendu.
2. Garder les structures métier identiques (masque calculé depuis deux coins) tout en filtrant désormais sur `roi_type == "box"` pour la reconstruction des previews.

---

### **2025-12-03** — ROI box partout (remplacement rectangle/rect)

**Tags :** `#views/annotation_view.py`, `#views/endview_view.py`, `#controllers/master_controller.py`, `#controllers/annotation_controller.py`, `#models/roi_model.py`, `#services/annotation_service.py`, `#roi`, `#box`, `#signals`, `#naming`, `#branch:annotation`

**Actions effectuées :**
- Renommé le flux rectangle → box sur l’ensemble des signaux et API : `box_drawn` (Endview/AnnotationView), handlers `on_annotation_box_drawn`, `set_temp_box`/`set_roi_boxes`/`clear_roi_boxes`, gestion du start `_box_start`.
- RoiModel expose `add_box` et `boxes_for_slice` (ROI type `box`).
- AnnotationService passe en nomenclature box : `build_box_mask`, `apply_temp_box`, `apply_box_roi(box=...)`, `_normalize_box_input`, rebuild renvoie les boxes et filtre sur `roi_type == "box"`.
- MasterController raccorde `box_drawn` et ToolsPanel émet déjà `tool_mode=box`.

**Contexte :**
Poursuite du remplacement complet du terme "rectangle" pour les ROI afin d’éviter toute confusion après les renommages free_hand/grow.

**Décisions techniques :**
1. Renommer toutes les API ROI rectangulaires (signaux, modèles, services) en “box” pour supprimer les doublons de terminologie.
2. Conserver les primitives géométriques (QGraphicsRectItem, tuples x1,y1,x2,y2) mais masquer la terminologie rectangle dans les interfaces publiques.

---

### **2025-12-04** — Fusion slice + ROI persistantes pour boxes

**Tags :** `#models/roi_model.py`, `#controllers/annotation_controller.py`, `#roi`, `#box`, `#persistence`, `#mvc`, `#branch:annotation`

**Actions effectuées :**
- RoiModel.boxes_for_slice merge les ROI de la slice courante avec les ROI marquées `persistent` (sans doublon via l’`id`) au lieu d’un simple fallback, et renvoie toutes les boxes agrégées.
- AnnotationController._rebuild_slice_preview reconstruit les masques temporaires avec les ROI de la slice plus les ROI persistantes fusionnées, garantissant que les boxes persistantes restent visibles lors du dessin d’une nouvelle box sur cette slice.
- Maintien du flux de preview (set_roi_boxes / set_roi_overlay) inchangé mais alimenté avec la liste fusionnée, supprimant la disparition des copies persistantes.

**Contexte :**
Avec la persistance activée, créer une nouvelle box sur une endview peuplée par des ROI persistantes faisait disparaître la copie persistante car le rebuild utilisait un fallback exclusif. La fusion slice + persistent préserve l’affichage et le masque temporaire.

**Décisions techniques :**
1. Utiliser une fusion (slice + persistantes) plutôt qu’un fallback pour éviter d’écraser les boxes héritées de la persistance.
2. Dédupliquer par `roi.id` pour ne pas dupliquer l’affichage lorsqu’une ROI est déjà présente sur la slice courante.

---

### **2025-12-04** — Implémentation ROI grow (click gauche) et clic sans Shift pour annotation

**Tags :** `#views/endview_view.py`, `#views/annotation_view.py`, `#controllers/annotation_controller.py`, `#services/annotation_service.py`, `#models/roi_model.py`, `#roi`, `#grow`, `#threshold`, `#mvc`, `#branch:annotation`

**Actions effectuées :**
- EndviewView : clic gauche sans Shift émet les interactions d’annotation (grow, etc.), Shift+clic reste dédié à la crosshair.
- AnnotationView : suit le tool_mode pour n’autoriser le tracé de box que quand l’outil box est actif et vide les formes temporaires lors d’un changement d’outil.
- RoiModel : ajout `add_grow` pour stocker les ROI de type grow (seed, label, threshold, persistence).
- AnnotationService : normalise la slice en uint8, region growing 4-connexe à partir du seed avec seuil (threshold None → 0), expose `apply_grow_roi` et reconstruit les masques grow lors du rebuild.
- AnnotationController : handle grow sur clic (sans Shift) avec fallback de shape depuis mask/temp/slice_data, seuil par défaut 0 si absent, applique le grow et rafraîchit la preview ROI.

**Contexte :**
Le mode grow ne réagissait pas au clic car seul Shift+clic déclenchait l’annotation. Le growing devait aussi fonctionner même si le seuil n’était pas encore fixé explicitement. La reconstruction devait rejouer les ROI grow persistantes.

**Décisions techniques :**
1. Découpler crosshair (Shift+clic) et actions d’annotation (clic gauche) pour permettre grow et futurs outils sans combinaison de touches.
2. Autoriser threshold par défaut (0) et clamping des shapes/seeds pour éviter les early returns et garantir un masque grow même sur valeur minimale.
3. Inclure les ROI grow dans le rebuild temp mask pour que la persistance/navigation réaffiche les prévisualisations grow.

---

### **2025-12-04** — Apply-volume ROI rebuild/apply + points visibles pour grow

**Tags :** `#controllers/annotation_controller.py`, `#models/roi_model.py`, `#views/annotation_view.py`, `#roi`, `#grow`, `#apply_volume`, `#persistence`, `#mvc`, `#branch:annotation`

**Actions effectuées :**
- AnnotationController : si "Appliquer au volume" est coché, `recompute` reconstruit toutes les previews ROI sur toutes les slices (ROIs slice + persistantes) via `_rebuild_volume_preview`, et `Appliquer ROI` applique le masque temporaire sur tout le volume (`_apply_temp_volume`), palette synchronisée.
- Temp previews : `_rebuild_volume_preview` initialise/clear le temp mask, itère les slices, fusionne ROIs persistantes, reconstruit box/grow via AnnotationService et relance l’overlay sur la slice courante.
- Visual : les seeds grow sont affichés en points blancs (AnnotationView `set_roi_points`/`clear_roi_points`, RoiModel `seeds_for_slice`, refresh controller).
- Apply local (sans volume) inchangé, mais le nettoyage inclut désormais les points grow et le temp mask complet si apply_volume.

**Contexte :**
Avec l’option "Appliquer au volume", le recalcul et l’application ROI ne traitaient que la slice courante, ignorant les ROIs persistantes/autres slices. Les grow n’étaient pas visibles graphiquement comme les boxes.

**Décisions techniques :**
1. Introduire des chemins volume-wide pour recompute/apply : rebuild des masques temporaires sur chaque slice avant l’application, puis fusion dans l’annotation model.
2. Fusionner slice + persistentes lors des rebuilds volume pour garantir que les ROIs persistantes sont incluses partout.
3. Ajouter un rendu dédié des seeds grow (points blancs) pour une visibilité équivalente aux boxes blanches.

---

### **2025-12-04** — Cache overlay déplacé au modèle + délégation volume

**Tags :** `#models/annotation_model.py`, `#services/annotation_service.py`, `#controllers/annotation_controller.py`, `#views/annotation_view.py`, `#views/overlay_settings_view.py`, `#overlay`, `#cache`, `#roi`, `#mvc`, `#branch:annotation`

**Actions effectuées :**
- AnnotationModel : ajout d’un cache OverlayData avec getters/setters/clear, invalidation automatique sur init/reset/set_slice_mask/set_mask_volume.
- AnnotationService : ajout des helpers métier volume (`rebuild_volume_preview_from_rois`, `apply_temp_volume_to_model`, `propagate_grow_volume_from_slice`) pour reconstruire/appliquer les masques sur tout le volume.
- AnnotationController : suppression du cache local, lecture/écriture du cache via le modèle ; délégation des boucles volume à AnnotationService ; résolution des dimensions via `_resolve_volume_dimensions`.
- Vues : AnnotationView ouvre la boîte de dialogue de sauvegarde overlay ; OverlaySettingsView fournit les conversions QColor↔BGRA utilisées par le contrôleur.

**Contexte :**
Le contrôleur mélangeait cache de données, boucles volume et interactions UI, s’écartant du MVC strict. Le cache overlay devait vivre côté modèle, et les opérations volume/grow devaient être dans le service métier.

**Décisions techniques :**
1. Centraliser le cache overlay dans le modèle pour coller aux données dérivées et l’invalider dès qu’un masque change.
2. Déporter les traitements volume (rebuild/apply, grow forward/backward) dans le service pour garder le contrôleur orchestral.
3. Déléguer les interactions Qt (dialogue de sauvegarde, conversions couleur) aux vues afin de supprimer la logique UI du contrôleur.

---

### **2025-12-04** — Ajout outil erase (gomme) ROI

**Tags:** `#controllers/annotation_controller.py`, `#services/annotation_service.py`, `#views/annotation_view.py`, `#views/tools_panel.py`, `#controllers/master_controller.py`, `#roi`, `#erase`, `#cursor`, `#mvc`, `#branch:annotation`

**Actions effectuées:**
- ToolsPanel/MasterController : radio Designer `radioButton_4` branché comme outil `erase`, mapping `select_tool_mode` mis à jour pour émettre `tool_mode_changed("erase")`.
- AnnotationView : curseur cercle creux (rayon 8 px) appliqué quand l’outil erase est actif, retour au curseur flèche sinon.
- AnnotationController/AnnotationService : clic en mode erase appelle `erase_disk_on_slice` (disque rayon 8 px) qui remet à 0 les pixels du masque de la slice courante, invalide le cache via `set_slice_mask` et rafraîchit l’overlay.

**Contexte:**
Besoin d’un outil gomme dédié branché sur un nouveau radio bouton « Erase » pour effacer localement des pixels du masque existant sans passer par les ROIs box/grow.

**Décisions techniques:**
1. Rayon fixe 8 px pour aligner visuellement le curseur cercle et limiter le coût de recalcul sur clic.
2. Effacement limité à la slice courante (pas de propagation volume) pour éviter les suppressions accidentelles ; invalidation overlay gérée par `set_slice_mask` puis `refresh_overlay`.

---

### **2025-12-04** — Support label 0 + outil paint (roi/brush)

**Tags:** `#controllers/annotation_controller.py`, `#services/annotation_service.py`, `#views/annotation_view.py`, `#views/tools_panel.py`, `#views/overlay_settings_view.py`, `#models/temp_mask_model.py`, `#roi`, `#paint`, `#label0`, `#mvc`, `#branch:annotation`

**Actions effectuées:**
- OverlaySettingsView : ajout d’un bouton « Ajouter label 0 » qui crée/émet le label id=0 (gris) s’il n’existe pas.
- ToolsPanel/Master : l’outil « erase » est renommé en « paint » (`tool_mode=paint`), mapping select_tool_mode mis à jour.
- AnnotationView : curseur cercle creux pour le mode paint.
- AnnotationController : clic paint applique un disque avec le label actif (y compris 0) via `paint_disk_on_slice`; preview ROI affiche aussi les zones label 0 via un masque sentinelle 255.
- TempMaskModel : ajout d’un `coverage_volume` booléen pour mémoriser les zones à appliquer, même pour label=0; clear/init/clear_slice mis à jour; getter coverage.
- AnnotationService : nouvelle `paint_disk_on_slice`; `apply_temp_volume_to_model` applique via coverage (permet label=0) ; apply-temp local dans le contrôleur utilise coverage.

**Contexte:**
Le flux doit permettre d’effacer en utilisant un label spécial 0 comme n’importe quel ROI/outil (paint/box/grow) plutôt qu’un effacement forcé. On doit voir le label 0 en preview ROI et l’appliquer réellement au volume.

**Décisions techniques:**
1. Couverture dédiée dans TempMaskModel pour savoir où appliquer même si la valeur cible est 0 (sinon l’information serait perdue).
2. Sentinel 255 pour afficher en preview les zones label 0 (palette 255 héritant de la couleur du label 0) sans coloriser l’arrière-plan.
3. Paint brush (disque rayon 8 px) applique le label actif ; l’effacement se fait donc en sélectionnant le label 0.

---

### **2025-12-04** — Paint en temp mask + bouton label 0

**Tags:** `#controllers/annotation_controller.py`, `#services/annotation_service.py`, `#views/annotation_view.py`, `#views/tools_panel.py`, `#views/overlay_settings_view.py`, `#models/temp_mask_model.py`, `#roi`, `#paint`, `#label0`, `#apply`, `#mvc`, `#branch:annotation`

**Actions effectuées:**
- OverlaySettingsView : bouton « Ajouter label 0 » ajoute/émet le label id=0 (gris) s’il est absent.
- Paint : le radio émet `tool_mode=paint`, curseur cercle creux conservé.
- Controller : clic paint écrit un disque (rayon 8 px) dans `TempMaskModel` (avec palette/coverage), pas directement dans AnnotationModel ; preview ROI affichée via overlay temp.
- Apply : chemin `apply_temp_mask_requested` préserve les strokes paint (mask+coverage) lors du rebuild volume ROI, puis applique via coverage pour supporter le label 0 ; apply slice idem via coverage.
- TempMaskModel : coverage_volume conservé, clear/init/clear_slice mis à jour ; apply_temp_volume applique via coverage (permet label 0).
- AnnotationService : helper `build_disk_mask(shape, center, radius)` pour générer un disque binaire.

**Contexte:**
Exiger que l’outil paint fonctionne comme les autres ROI : écriture dans un masque temporaire, prévisualisation, puis application explicite. Le label 0 doit être utilisable comme n’importe quel label pour effacer.

**Décisions techniques:**
1. Coverage booléen dédié pour ne pas perdre les zones à appliquer lorsque la valeur cible est 0.
2. Merge paint + rebuild ROI en apply_volume : on sauvegarde/restaure (mask, coverage) avant/après `rebuild_volume_preview_from_rois` pour ne pas effacer les strokes paint.
3. Disque construit via `build_disk_mask` pour réutilisation et robustesse aux shapes/clamp.

---

### **2025-12-04** — Affichage dynamique du threshold

**Tags:** `#views/tools_panel.py`, `#controllers/master_controller.py`, `#ui_mainwindow.py`, `#threshold`, `#ui`, `#mvc`, `#branch:annotation`

**Actions effectuées:**
- ToolsPanel : le label threshold est injecté et stocké ; le slider met à jour le texte `Threshold : <val>` via `_on_threshold_changed`, et `set_threshold_value` met aussi à jour le label.
- MasterController : passe `label_2` (Designer) au ToolsPanel pour activer l’affichage dynamique.

**Contexte:**
L’étiquette “Threshold :” devait afficher en direct la valeur du slider pour refléter le seuil courant.

**Décisions techniques:**
1. Gérer l’UI côté ToolsPanel (pas dans le contrôleur) pour rester MVC et limiter les dépendances au Designer.
2. Bloquer les signaux lors des mises à jour programmatiques du slider et rafraîchir le label ensuite pour éviter les boucles de signaux.

---
