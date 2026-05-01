### 2025-11-26 - Align model/controller skeleton imports

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

### 2025-12-10 - Correctifs multi-sessions (overlay 3D et duplication)

**Tags:** `#controllers/master_controller.py`, `#views/volume_view.py`, `#session`, `#overlay`, `#3d`, `#state-management`, `#branch:interpolation`

**Actions effectuées:**
- `volume_view.set_volume` nettoie désormais systématiquement les visuels d’overlay (`_clear_overlay_visuals`) avant de reconstruire la scène VisPy et lorsqu’un volume vide est défini, évitant la disparition de l’overlay 3D lors d’un switch de session avec labels communs.
- `MasterController` déclenche `_after_session_switch()` juste après la création d’une session (et `save_active=True` pour snapshot de la session courante), garantissant la resynchronisation immédiate des vues/overlays après duplication.

**Contexte:**
En multi-sessions, basculer entre des sessions partageant les mêmes labels faisait disparaître l’overlay 3D jusqu’à ce que l’utilisateur toggle l’overlay. De plus, la création d’une 2e session pouvait remplacer l’état de la première faute de snapshot avant duplication.

**Décisions techniques:**
1. Nettoyer les visuels overlay avant chaque rebuild de volume pour forcer une ré-application propre de l’overlay 3D sur la scène reconstruite.
2. Sauvegarder l’état actif avant création et relancer une resynchro complète après création afin de préserver la session source et afficher immédiatement l’overlay dans la nouvelle session.

---
### 2025-12-10 - Gestion multi-sessions d’annotation (sélecteur Session)

**Tags:** `#services/annotation_session_manager.py`, `#views/session_manager_dialog.py`, `#controllers/master_controller.py`, `#ui`, `#session`, `#state-management`, `#branch:interpolation`

**Actions effectuées:**
- Créé `AnnotationSessionManager` pour snapshot/restore des modèles (`AnnotationModel`, `TempMaskModel`, `RoiModel`, `ViewStateModel`), avec stockage en mémoire, create/delete/list/switch et session par défaut.
- Ajouté `SessionManagerDialog` (QDialog) listant les sessions, création (duplique l’état courant), suppression et sélection, déclenché par l’action Qt `actionSession_selector` du menu Session.
- `MasterController` instancie le manager, connecte `actionSession_selector`, gère les callbacks de création/sélection/suppression et resynchronise les vues (overlay/cross toggles, colormaps, labels, overlay settings, refresh overlays + vues) après un switch de session.

**Contexte:**
Besoin de conserver plusieurs états d’annotation en mémoire et de basculer rapidement via le menu Session, sans recharger un NPZ. Le flux MVC reste inchangé pour une session ; on réutilise les modèles existants en les rechargeant depuis un snapshot à chaque switch.

**Décisions techniques:**
1. Snapshots profonds (np.copy/deepcopy) des volumes masques, palette/visibilité, temp mask + coverage, ROIs et view state ; le cache overlay est invalidé lors de l’application d’une session.
2. Pas de duplicata de contrôleurs/vues : on recharge simplement l’état dans les modèles existants puis on pousse vers les vues via une routine `_after_session_switch`.
3. Dialogue léger en liste + boutons (créer/supprimer/fermer) ; l’action menu Session `actionSession_selector` ouvre le dialog.

---
### 2025-12-10 - Palette défaut appliquée + contrôles C-scan/endview corrigés

**Tags:** `#models/annotation_model.py`, `#views/overlay_settings_view.py`, `#services/cscan_corrosion_service.py`, `#views/cscan_view.py`, `#views/endview_view.py`, `#ui`, `#overlay`, `#colors`, `#zoom`, `#panning`, `#branch:interpolation`

**Actions effectuées:**
- `AnnotationModel.set_mask_volume` applique désormais la palette `MASK_COLORS_BGRA` pour chaque label détecté (fallback magenta seulement si non défini) afin de conserver les couleurs par défaut lors du chargement NPZ.
- `OverlaySettingsView` associe l’ajout de label à la palette par défaut (fallback roue HSV au-delà de la palette, label 0 reste gris), alignant les couleurs créées manuellement sur les valeurs configurées.
- `CScanCorrosionService` enlève le flip + transpose de l’overlay corrosion, sauvant le NPZ directement en (Z,H,W) sans miroir Y.
- `CScanView` ajoute le pan au bouton droit (scrollbars) et garde la molette pour le zoom uniquement (consommation de l’événement).
- `EndviewView` désactive les scrollbars, intercepte la molette pour zoom-only et importe `QEvent` pour éviter le crash `NameError` ; pan existant conservé.

**Contexte:**
Les overlays NPZ et l’ajout de labels ignoraient la palette par défaut, produisant des couleurs magenta/aléatoires. Le NPZ corrosion était miroité sur Y à cause d’un flip/transpose. Les vues C-scan/endview mélangeaient molette/scroll et C-scan manquait du pan au bouton droit ; l’interception de la molette provoquait un crash faute d’import `QEvent`.

**Décisions techniques:**
1. Utiliser `MASK_COLORS_BGRA` comme palette source unique pour la découverte de labels (NPZ) et l’ajout manuel, avec magenta uniquement en secours.
2. Supprimer les opérations de flip/transpose sur l’overlay corrosion pour aligner (Z,H,W) sur le volume sans correction côté loader.
3. Harmoniser l’interaction : pan au bouton droit, molette strictement zoom sur C-scan et endview, en consommant l’événement pour éviter tout scroll ou changement de slice ; ajouter `QEvent` pour sécuriser l’eventFilter.

---

### 2025-12-05 - Rotation fallback auto pour NDE sans orientation explicite

**Tags :** `#services/nde_loader.py`, `#rotation`, `#orientation`, `#fallback`, `#domain-structure`, `#mvc`, `#branch:annotation`

**Actions effectuées :**
- Ajout de `_maybe_rotate`/`_should_apply_rotation` : la rot90 horaire n’est plus systématique, elle ne s’applique que si l’`axis_order` est auto-généré (`axis_*`) ou si le fichier Domain ne fournit pas `dataMappings` mais uniquement des `dataEncodings` (schéma 3.x legacy).
- Les chemins Public/Domain appellent `_maybe_rotate` après l’orientation par métadonnées/fallback ; log INFO détaillé quand la rotation est appliquée (shape/axis_order avant-après).

**Contexte :**
Certains NDE Public 4.0 étaient déjà correctement orientés et subissaient une rotation en trop. Les Domain 3.x (grille exposée via `dataEncodings` seulement) ou les cas sans métadonnées requièrent encore la rot90 héritée pour l’affichage.

**Décisions techniques :**
1. Ne pas appliquer de rotation si des métadonnées explicites existent (`dataMappings` Public/Domain) et un `axis_order` non `axis_*` est fourni.
2. Appliquer la rotation uniquement en fallback (axis_* généré) ou sur Domain legacy (pas de `dataMappings` mais présence de `dataEncodings`), conservant le swap/inversion des positions aligné sur la rot90 horaire existante.

---

### 2025-12-04 - Bouton nnUNet branché dans MasterController

**Tags:** `#controllers/master_controller.py`, `#nnunet`, `#pyqt6`, `#mvc`, `#ui`, `#branch:annotation`

**Actions effectuées:**
- Ajout de l’import QTimer, MASK_COLORS_BGRA et du service `NnUnetService`/`NnUnetResult`, avec instanciation `self.nnunet_service` dans `MasterController`.
- Raccordement des actions Qt `actionnnunet` et `actionSauvegarder` au contrôleur; nouvel handler `_on_run_nnunet` lance les boîtes de dialogue (modèle nnUNet + chemin .npz), déclenche `run_inference`, et applique le masque retourné sur `annotation_model` avec palette issue du mapping de labels.
- Dans `_on_run_nnunet`, callbacks succès/erreur dispatchés via `QTimer.singleShot(0, ...)`, réinitialisent l’overlay, activent la visibilité overlay, synchronisent paramètres/labels (`clear_labels`, `sync_overlay_settings`, `_sync_tools_labels`, `refresh_overlay`), et affichent messages de statut/boîte d’info.

**Contexte:**
Port du bouton nnUNet présent dans la version "copy" pour permettre le lancement d’une inférence nnUNet depuis le menu Inference tout en conservant la structure MVC existante. Le masque résultant est injecté dans `annotation_model` avec couleurs provenant de `MASK_COLORS_BGRA`, overlay forcé visible, et vue rafraîchie sans toucher au reste des contrôleurs.

**Décisions techniques:**
1. Utiliser `QTimer.singleShot` pour renvoyer l’application des résultats nnUNet dans le thread UI et éviter tout thread-safety issue lors des callbacks asynchrones.
2. Conserver `actionExporter_npz` et ajouter `actionSauvegarder` pour compatibilité UI, tout en connectant explicitement `actionnnunet` au nouveau handler afin de limiter l’impact aux changements demandés.

**Implémentation (extrait clé):**
```python
self.nnunet_service.run_inference(
    volume=volume,
    raw_volume=raw_volume,
    model_path=model_path,
    output_path=save_path,
    dataset_id=str(dataset_id) if dataset_id else "current",
    on_success=_on_success,
    on_error=_on_error,
)
```

---

### 2025-12-01 - Nettoyage palette générique et suppression logging_config

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

### 2025-12-01 - Squelettes annotation_view / annotation_service et routage ROI vers AnnotationController

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

### 2025-12-01 - Export overlay NPZ via AnnotationController

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

### 2025-11-29 - Refactor corrosion workflow (C-scan/AScan controllers)

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
### 2025-11-29 - Gestion overlay centralisée et labels dynamiques

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

### 2025-11-28 - Overlay NPZ/NPY géré via NPZOverlayService et annotation_model

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

### 2025-11-28 - Overlay NPZ: auto-transpose H/W

**Tags :** `#services/npz_overlay.py`, `#overlay`, `#mvc`

**Actions effectuées :**
- `NPZOverlayService.load` tolère un overlay dont les dimensions H/W sont inversées par rapport au volume (ex: (Z, W, H) vs (Z, H, W)) : détection (même profondeur, axes 1 et 2 permutés), transpose `(0,2,1)` et log avant de bâtir l’RGBA.
- Les autres shapes restent rejetées par une erreur.

**Contexte :**
Des NPZ/NPY fournis avec H/W échangés doivent être acceptés sans demander de retoucher les vues/contrôleur ; le service corrige l’ordre pour livrer un masque aligné au volume NDE.

**Décision technique :**
1. Vérifier la profondeur puis appliquer un transpose (0,2,1) sur mismatch H/W au lieu d’échouer, afin de conserver une orientation cohérente avec le volume attendu.

---

### 2025-11-28 - Overlay 3D poussé via contrôleur

**Tags :** `#controllers/master_controller.py`, `#views/volume_view.py`, `#overlay`, `#mvc`

**Actions effectuées :**
- `_push_overlay` envoie désormais l’overlay RGBA du service à la fois à l’Endview et à la VolumeView, et les nettoie toutes deux si le toggle overlay est désactivé.

**Contexte :**
Le volume overlay (annotations) n’apparaissait que dans l’Endview ; le contrôleur pousse maintenant le même overlay vers la vue 3D pour afficher l’annotation par-dessus le volume NDE.

**Décision technique :**
1. Mutualiser le push overlay pour les deux vues afin de garder une seule source (service overlay_rgba) et respecter le toggle overlay.

---

### 2025-11-28 - Fix compat numpy (overlay asarray copy)

**Tags :** `#services/npz_overlay.py`, `#overlay`, `#numpy`

**Actions effectuées :**
- Remplacé `np.asarray(..., copy=False)` par `np.array(..., copy=False)` lors du chargement overlay pour éviter l’erreur `asarray() got an unexpected keyword argument 'copy'` sur les versions numpy antérieures.

**Contexte :**
La compatibilité numpy nécessitait d’éviter l’argument `copy` avec `asarray`; `np.array` supporte `copy` et conserve le cast en uint8 sans duplication inutile.

---

### 2025-11-28 - Rotation 90° horaire appliquée dans SimpleNdeLoader

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
### 2025-11-28 - Orientation fix via display transforms (pas de rotation des données)

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
### 2025-11-28 - Rotations différenciées Endview/Volume

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
### 2025-11-28 - Caméra centrée sur la slice courante en 3D

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

### 2025-11-28 - Désactivation du plan de surlignage VolumeView

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
### 2025-11-27 - VolumeView coupe 3D style volume_plane

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

### 2025-11-27 - Forcer l’orientation Domain sur l’axe lengthwise

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
### 2025-11-27 - Crosshair sync gated to Shift-click

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

### 2025-11-27 - Heuristique Domain assouplie pour l’orientation

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
### 2025-11-27 - ToolsPanel signals and controller wiring

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

### 2025-11-27 - Position label + cross toggle via ViewStateModel

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
### 2025-11-27 - Menu actions wiring

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

### 2025-11-27 - NDE loading via service in controller/model

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

### 2025-11-26 - Respect UI Designer layout for controller/views

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

### 2025-11-27 - NDE loader renvoie un modèle orienté/normalisé prêt

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

### 2025-11-27 - Core viewer scaffolding for MVC UI

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

### 2025-01-27 - Refactorisation MVC complète pour les vues Endview, 3D et A-scan

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

### 2025-01-27 - Correction du bug de la croix C-Scan : ligne verticale qui ne bouge pas

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

### 2025-11-27 - VolumeView slice highlight via plane-mode Volume

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
### 2025-11-28 - Rotation slice-wise après orientation NDE

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
### 2025-11-28 - Rotation appliquée aux endviews en mémoire

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
### 2025-11-28 - Orientation caméra VolumeView alignée sur Endview

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
### 2025-11-28 - Caméra VolumeView centrée sur la slice courante

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
### 2025-11-28 - Focalisation caméra VolumeView sur l’axe Z

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
### 2025-11-28 - Référence Volume/CScan mises à jour selon ui_mainwindow

**Tags:** `#controllers/master_controller.py`, `#ui_mainwindow.py`, `#mvc`, `#views/volume_view.py`, `#views/cscan_view.py`

**Actions effectuées:**
- Inversé les références des vues dans `MasterController` pour suivre l’échange de frames dans `ui_mainwindow.py`: `cscan_view` pointe maintenant sur `frame_4` et `volume_view` sur `frame_5`.

**Contexte:**
Le fichier Designer a été modifié pour intervertir les frames (CScanView sur `frame_4`, VolumeView sur `frame_5`). Le contrôleur devait refléter ce mapping pour que les signaux et mises à jour utilisent les bons widgets.

**Décisions techniques:**
1. Mettre à jour uniquement les affectations de références sans toucher au wiring existant, afin de conserver la logique MVC tout en alignant le contrôleur avec la nouvelle structure UI.

---
### 2025-11-28 - Caméra VolumeView flippée de 180°

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
### 2025-11-28 - Flip vertical (Y) de la scène VolumeView

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
### 2025-11-28 - Flip Y appliqué aux visuels VolumeView

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
### 2025-11-28 - Flip appliqué aux mises à jour d’image de slice

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
### 2025-11-28 - Fallback d’orientation pour NDE sans métadonnées

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

### 2025-11-28 - Overlay 3D translucide dans VolumeView

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

### 2025-11-28 - Overlay non repush sur navigation de slices

**Tags :** `#controllers/master_controller.py`, `#overlay`, `#performance`, `#mvc`

**Actions effectuées :**
- Supprimé l’appel à `_push_overlay()` dans `_on_slice_changed` pour éviter de réinjecter le volume RGBA à chaque déplacement de slice.

**Contexte :**
Chaque mouvement de slider déclenchait un push complet de l’overlay RGBA vers les vues, générant des logs et un coût inutile. Les vues gèrent déjà la mise à jour de slice via `set_slice` / `set_slice_index` et l’overlay 2D/3D se met à jour localement sans recharger les données.

**Décisions techniques :**
1. Laisser `_push_overlay` uniquement sur chargement NDE, chargement overlay ou toggle overlay, car ces événements changent effectivement les données.
2. S’appuyer sur les vues (Endview/VolumeView) pour rafraîchir l’overlay slice côté rendu lors des changements d’index, évitant un repush volumique coûteux.

---

### 2025-11-28 - Log diagnostique A-Scan au chargement

**Tags :** `#controllers/master_controller.py`, `#services/ascan_service.py`, `#ascan`, `#logging`, `#mvc`

**Actions effectuées :**
- Ajout de `_log_ascan_preview` appelé juste après le chargement NDE : extrait un profil A-Scan au centre (slice, x, y) et loggue longueur, min/max/mean, head(5), marker et crosshair.

**Contexte :**
Les fichiers Domain sans métadonnées ont un A-Scan « spécial » après fallback d’orientation. Le log console permet d’inspecter rapidement le contenu du profil sans interagir dans l’UI.

**Décisions techniques :**
1. Utiliser `AScanService.build_profile` pour respecter l’axe ultrasound détecté et les positions ; log structuré seulement si profil non vide.
2. Éviter le bruit : early return si volume invalide/vide, head limité à 5 valeurs, stats simples (len/min/max/mean) pour lecture console.

---

### 2025-11-28 - Compat numpy pour log A-Scan

**Tags :** `#controllers/master_controller.py`, `#ascan`, `#logging`, `#numpy`

**Actions effectuées :**
- Supprimé l’usage de `initial=` dans `min/max/mean` pour le log A-Scan (incompatible avec certaines versions numpy). Utilisation directe de `sig.min()/max()/mean()` après vérification `size>0`.

**Contexte :**
Une exception `_mean() got an unexpected keyword argument 'initial'` apparaissait sur des environnements numpy plus anciens lors du chargement NDE. Le log diagnostic A-Scan fonctionne maintenant sans ce paramètre.

**Décisions techniques :**
1. Conserver le guard `size>0` avant calcul, ce qui rend inutile l’argument `initial`.

---

### 2025-11-28 - Logs A-Scan enrichis (axe ultrasound, brut)

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

### 2025-11-28 - Heuristique ultrasound sur axe le plus long

**Tags :** `#services/ascan_service.py`, `#ascan`, `#domain-structure`, `#orientation`, `#mvc`

**Actions effectuées :**
- Ajusté `_ultrasound_axis_index` : si aucun axe n’est nommé « ultrasound », on choisit désormais l’axe le plus long parmi les axes > 0 (non-slice) au lieu du dernier axe par défaut.

**Contexte :**
Les fichiers Domain sans métadonnées `dimensions` voyaient l’axe ultrasound sélectionné par défaut sur le dernier axe (longueur 113), alors que l’axe attendu est de longueur 276. Cette heuristique sélectionne désormais l’axe le plus long (ici 276) pour obtenir des profils A-Scan corrects.

**Décisions techniques :**
1. Considérer les axes après l’axe slice (index 0) et prendre le plus long, tri décroissant, pour refléter l’axe ultrasound probable quand il n’est pas nommé.
2. Garder le fallback `max(0, len(shape)-1)` si la shape est trop courte (<3 axes).

---

### 2025-11-28 - Logs A-Scan déplacés dans le service

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

### 2025-11-28 - Debug log A-Scan externalisé

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

### 2025-11-29 - Renommage NDE/overlay loaders et modèle

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

### 2025-11-29 - ViewStateModel centralisation & controller cleanup

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

### 2025-11-29 - Extraction du rendu overlay vers OverlayService

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

### 2025-11-29 - Pile C-scan construite dans MasterController

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

### 2025-11-29 - Service de workflow corrosion injecté

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

### 2025-11-29 - Extraction overlay vers AnnotationController

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

### 2025-11-29 - État de navigation centralisé dans ViewStateModel

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

### 2025-11-29 - Renommage OverlayDebugLogger

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

### 2025-11-29 - Overlay alpha compact + refresh sans reset

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

### 2025-11-29 - Fix asarray(copy) crash on overlay load

**Tags :** `#views/volume_view.py`, `#overlay`, `#numpy`, `#bugfix`

**Actions effectuées :**
- Remplacé `np.asarray(..., dtype=np.float32, copy=False)` par `np.array(..., dtype=np.float32, copy=False)` dans `VolumeView.set_overlay` pour supporter les versions de numpy où `asarray` n’accepte pas `copy`.

**Contexte :**
Chargement d’NPZ déclenchait `asarray() got an unexpected keyword argument 'copy'` lors de la conversion du volume alpha overlay; utiliser `np.array` supprime l’argument incompatible tout en conservant la conversion float32 sans copie inutile.

**Décisions techniques :**
1. Conversion alpha via `np.array(..., copy=False)` pour rester zéro-copy quand possible et compatible avec numpy <1.26.
2. Aucun impact sur le pipeline overlay (payload compact mask+alpha+palette) ; uniquement la conversion d’entrée côté vue 3D.

---

### 2025-11-30 - OverlayService perf: filtrage sans copie globale

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

### 2025-11-30 - Déferlement overlay 3D pour toggles rapides

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

### 2025-11-30 - Overlays par label en mémoire + toggles visibles sans reupload

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

### 2025-11-30 - Log overlay counts (mask/palette/visible)

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

### 2025-11-30 - Reset overlay state on NDE load

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

### 2025-12-01 - Zoom molette C-scan

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

### 2025-12-01 - Chunked nnUNet inference (8 parts)

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

### 2025-12-01 - Ajout règle de tag de branche dans AGENTS.md

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

### 2025-12-01 - Ajout modèles ROI/temp et préview rectangle

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

### 2025-12-01 - Fix rectangle ROI crash quand le volume masque est absent

**Tags :** `#controllers/annotation_controller.py`, `#roi`, `#annotation`, `#branch:annotation`

**Actions effectuées :**
- Corrigé `on_annotation_rectangle_drawn` pour ne plus évaluer un `ndarray` dans un `or` (ambiguïté). On récupère d’abord le masque de `AnnotationModel`, sinon celui de `TempMaskModel`, et on quitte si aucun volume n’est disponible.

**Contexte :**
Le dessin rectangle plantait avec un `ValueError` (truth value of an array…) car l’expression `mask_a or mask_b` tentait d’évaluer la vérité d’un `ndarray`.

**Décisions techniques :**
1. Vérification explicite `None` pour les volumes masque avant usage, afin d’éviter l’ambiguïté des `ndarray`.

### 2025-12-01 - Label actif dans le tools panel, couleur du polygone par label

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

### 2025-12-02 - Session ROI/labels (extraction métier, palettes, preview)

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

### 2025-12-03 - Injection slice volume pour seuil ROI

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

### 2025-12-03 - Navigation ROI + appli volume

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

### 2025-12-03 - Renommage temp polygon en temp mask (ROI)

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

### 2025-12-03 - Renommage ROI polygon → free hand

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

### 2025-12-03 - Handlers renommés free hand

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

### 2025-12-03 - Signaux free hand (remplacement polygon_*)

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

### 2025-12-03 - AnnotationService: paramètres free hand

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

### 2025-12-03 - Mode ROI point renommé grow

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

### 2025-12-03 - ROI rectangle renommé box

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

### 2025-12-03 - ROI rectangle → box (signals, modèles, services)

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

### 2025-12-03 - ROI box partout (remplacement rectangle/rect)

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

### 2025-12-04 - Fusion slice + ROI persistantes pour boxes

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

### 2025-12-04 - Implémentation ROI grow (click gauche) et clic sans Shift pour annotation

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

### 2025-12-04 - Apply-volume ROI rebuild/apply + points visibles pour grow

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

### 2025-12-04 - Cache overlay déplacé au modèle + délégation volume

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

### 2025-12-04 - Ajout outil erase (gomme) ROI

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

### 2025-12-04 - Support label 0 + outil paint (roi/brush)

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

### 2025-12-04 - Paint en temp mask + bouton label 0

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

### 2025-12-04 - Affichage dynamique du threshold

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

### 2025-12-04 - Correctif colormap corrosion (_to_rgb signature)

**Tags:** `#views/cscan_view_corrosion.py`, `#views/cscan_view.py`, `#cscan`, `#colormap`, `#bugfix`, `#branch:annotation`

**Actions effectuées:**
- Ajusté `_to_rgb` dans `CscanViewCorrosion` pour accepter un argument LUT optionnel (aligné sur la signature de `CScanView`), en ignorant ce paramètre et en continuant d’utiliser le LUT corrosion interne.
- La génération d’image corrosion redevient compatible avec `_render_pixmap` qui passe désormais la LUT en paramètre.

**Contexte:**
Après l’ajout de colormaps configurables, `CScanView._render_pixmap` appelle `_to_rgb(data, value_range, lut)`. La surcharge dans `CscanViewCorrosion` ne prenait que deux arguments, provoquant un `TypeError` lors de l’analyse corrosion. La mise à jour aligne la signature pour empêcher l’erreur tout en conservant le LUT dédié corrosion.

**Décisions techniques:**
1. Ne pas utiliser le LUT passé en argument pour la corrosion : la palette reste figée via le LUT interne afin d’assurer un rendu cohérent de la distance corrosion.
2. Signature alignée (ajout d’un paramètre optionnel) pour éviter toute régression future si `_render_pixmap` évolue encore sur la base CScanView.

---

### 2025-12-05 - Nouveau split flaw/noflaw basé sur masque courant

**Tags :** `#services/split_service.py`, `#controllers/master_controller.py`, `#services/endview_export.py`, `#mvc`, `#ui`, `#branch:annotation`

**Actions effectuées :**
- Créé `services/split_service.py` (classe `SplitFlawNoflawService`) qui exporte systématiquement les endviews RGB/uint8 via `EndviewExportService` puis trie chaque slice en flaw/noflaw selon le mask_volume courant, en copiant les endviews vers flaw/noflaw et en écrivant les masques dans gtmask/flaw ou gtmask/noflaw.
- Remplacé l’ancien `services/split_flaw_noflaw.py` par ce nouveau service et supprimé le fichier legacy.
- Rebranché `actionSplit_flaw_noflaw` dans `MasterController` pour appeler `split_flaw_noflaw_service.split_endviews`, en construisant le dossier racine `[nom_du_nde]` sous le répertoire choisi par l’utilisateur.

**Contexte :**
Le split doit désormais s’appuyer sur le modèle NDE déjà chargé et le masque d’annotation courant, sans script externe ni analyse de fichiers masks_binary. La structure de sortie est toujours `endviews_rgb24/complete|flaw|noflaw|gtmask/...` et `endviews_uint8/...` sous un dossier nommé d’après le fichier .nde.

**Décisions techniques :**
1. Exporter systématiquement les endviews pour garantir cohérence avant le split (pas de skip conditionnel).
2. Bucket flaw dès qu’un pixel de masque est non nul ; sauve le masque slice dans gtmask et copie l’endview correspondante depuis `complete`.
3. Le nom du dossier racine est dérivé du chemin .nde (metadata path ou paramètre) afin d’obtenir `[nom du nde]/endviews_*`.

---

### 2025-12-05 - Menu actions endviews + split flaw-noflaw reliés aux services

**Tags :** `#services/endview_export.py`, `#services/split_flaw_noflaw.py`, `#controllers/master_controller.py`, `#mvc`, `#ui`, `#branch:annotation`

**Actions effectuées :**
- Simplifié `EndviewExportService` pour charger via `NdeLoader` ou un `NdeModel` existant, normaliser localement et exporter endviews RGB/uint8 avec transformations optionnelles sans dépendances externes.
- Ajusté `split_flaw_noflaw_gui`/`split_from_npz_file` pour accepter `nde_loader`/`nde_model` et utiliser `export_endviews_gui` avec ces dépendances, couvrant l’export automatique avant le split.
- Relié `actionExporter_endviews` et `actionSplit_flaw_noflaw` dans `MasterController` avec handlers UI (QFileDialog) appelant les services, exportant sous `endviews_rgb24/complete` et `endviews_uint8/complete` puis lançant le split.

**Contexte :**
On supprime toute logique métier hors services : export et split sont encapsulés dans leurs scripts, le contrôleur ne fait que collecter les chemins et déclencher les services. L’export réutilise le volume déjà chargé si fourni, sinon charge via `NdeLoader`.

**Décisions techniques :**
1. Préférer `nde_model` existant pour éviter de recharger le .nde ; fallback `NdeLoader` si chemin fourni.
2. Normalisation par min/max du volume dans `EndviewExportService` pour produire des PNG cohérents même sans métadonnées.
3. Les handlers UI créent/réutilisent les dossiers `endviews_rgb24/complete` et `endviews_uint8/complete`, puis affichent les messages de statut via QMessageBox/statusbar.

---

### 2025-12-05 - Affichage du nom NDE et de l’endview dans le ToolsPanel

**Tags:** `#controllers/master_controller.py`, `#views/tools_panel.py`, `#ui_mainwindow.py`, `#untitled.ui`, `#ui`, `#mvc`, `#endview`, `#nde`, `#branch:annotation`

**Actions effectuées:**
- ToolsPanel accepte désormais les labels NDE/endview du Designer et expose `set_nde_name` / `set_endview_name` avec placeholders "-" .
- MasterController passe les deux labels au ToolsPanel, stocke `_nde_path`, met à jour le nom du fichier NDE après chargement et rafraîchit le nom d’endview à chaque changement de slice (`endview_{slice_idx*1500:012d}.png`).
- ui_mainwindow.py/untitled.ui initialisent les libellés à "NDE: -" et "Endview: -" pour l’état vide.

**Contexte:**
Afficher dans le panneau d’outils quel fichier NDE est ouvert et quelle endview (slice) est affichée, en respectant le naming utilisé pour les exports endview.

**Décisions techniques:**
1. Mettre la logique d’affichage dans la vue (ToolsPanel) pour rester MVC : le contrôleur ne fait que pousser les valeurs.
2. Recalcul de l’identifiant endview sur `_on_slice_changed` via la formule d’export (slice_idx*1500) afin de rester cohérent avec les noms de fichiers générés par les services d’export/split.

---

### 2025-12-10 - Cache sessions et projections C-scan

**Tags :** `#services/annotation_session_manager.py`, `#controllers/cscan_controller.py`, `#controllers/master_controller.py`, `#overlay`, `#cscan`, `#caching`, `#session`, `#branch:interpolation`

**Actions effectuées :**
- Stocké `overlay_cache` dans `AnnotationSessionState` et réappliqué lors des switches; `_apply` réassigne `mask_volume`, `temp_mask_volume`, `coverage` par référence (pas de copie) pour accélérer les changements de session.
- Ajouté `reset_for_new_dataset` pour vider les sessions lors du chargement d’un nouveau NDE.
- Mis en cache la projection C-scan standard dans `CScanController` (projection + range liés au shape du volume) avec invalidation sur reset corrosion.
- Dans `_refresh_views`, usage de `refresh_overlay(rebuild=False)` pour réutiliser l’overlay caché plutôt que de le reconstruire.

**Contexte :**
Les switches de session étaient lents car chaque passage recréait l’overlay et recalculait la projection C-scan. En mémorisant l’`overlay_cache` par session et en réutilisant les projections standard lorsque le shape ne change pas, le switch devient principalement un rerendu des vues. La réinitialisation des sessions sur nouveau NDE évite des incohérences de shapes.

**Décisions techniques :**
1. Réaffecter directement les volumes lors du switch de session pour éviter les copies coûteuses; conserver néanmoins une copie lors du snapshot pour figer l’état sauvegardé.
2. Lier le cache C-scan au `volume.shape` et l’invalider lors d’un reset corrosion pour rester cohérent avec les changements de volume.

---

### 2025-12-10 - Optimisation analyse corrosion (interpolation vectorisée)

**Tags :** `#services/cscan_corrosion_service.py`, `#services/distance_measurement.py`, `#corrosion`, `#performance`, `#numpy`, `#branch:interpolation`

**Actions effectuées :**
- Vectorisé le recalcul de carte de distance interpolée BW/FW dans `_build_interpolated_distance_map` en utilisant `np.nonzero` + `np.bincount` par slice (suppression des boucles X), avec facteur métrique appliqué en une fois.
- Désactivé les logs de performance détaillés (`ENABLE_PERF_LOGS=False`) dans `DistanceMeasurementService` pour réduire la surcharge CPU/IO lors de l’analyse corrosion.

**Contexte :**
L’analyse corrosion était lente à cause de boucles imbriquées (Z/X) pour l’interpolation et de logs détaillés. La nouvelle implémentation calcule les moyennes Y par X pour les deux classes en une passe vectorisée, puis la distance verticale, ce qui réduit fortement le temps CPU. Les logs de perf sont coupés par défaut pour ne pas pénaliser le runtime.

**Décisions techniques :**
1. Utiliser `bincount` pour sommer/compter les positions Y par X et dériver les moyennes, car cette approche est O(n) et évite les boucles Python sur X.
2. Laisser le facteur métrique appliqué en fin de calcul (scale) pour conserver un seul chemin de calcul et éviter les multiplications dans la boucle.

---

### 2025-12-10 - Hotfix corrosion: distance_results missing after vectorization

**Tags :** `#services/cscan_corrosion_service.py`, `#corrosion`, `#bugfix`, `#branch:interpolation`

**Actions effectuées :**
- Initialisé `distance_results` à un dict vide dans `run_analysis` après le calcul vectorisé de `distance_map`, pour éviter NameError et permettre le passage à `_build_front_back_overlay` (qui gère un dict vide en produisant un overlay vide).

**Contexte :**
Le refactor vectorisé supprimait l’ancienne structure distance_results; un NameError se produisait lors de l’appel à `_build_front_back_overlay`. Le correctif rétablit une valeur par défaut cohérente (dict vide) pour maintenir le pipeline sans plantage.

**Décisions techniques :**
1. Fournir un `distance_results` vide pour compatibilité avec la signature existante, en assumant un overlay nul si aucune donnée détaillée n’est disponible.

---

### 2025-12-10 - Overlay corrosion aligné sur l’interpolation BW/FW

**Tags :** `#services/cscan_corrosion_service.py`, `#corrosion`, `#overlay`, `#performance`, `#branch:interpolation`

**Actions effectuées :**
- Construction de l’overlay corrosion à partir du mask_stack en reliant les points BW/FW par X croissant (`_build_overlay_from_masks`), au lieu de mapper tout le masque; l’interpolation BW/FW se base maintenant sur cet overlay de lignes.
- L’interpolation de carte de distances réutilise cet overlay de lignes (colors A/B), évitant d’avoir le même NPZ que l’original dans la nouvelle session.

**Contexte :**
La nouvelle session héritait d’un overlay identique au masque brut. Désormais l’overlay corrosion est constitué uniquement des lignes front/back dérivées du mask_stack, ce qui alimente l’interpolation et différencie clairement la session interpolée.

**Décisions techniques :**
1. Générer un overlay de lignes via `np.nonzero` + tri par X et tracé OpenCV, plutôt que de réaffecter tout le masque.
2. Utiliser cet overlay de lignes comme base pour la carte interpolée afin de rester cohérent entre NPZ interpolé et C-scan interpolé.

---

### 2025-12-10 - Overlay slice update sans rebuild 3D

**Tags :** `#controllers/annotation_controller.py`, `#views/volume_view.py`, `#services/overlay_service.py`, `#models/annotation_model.py`, `#overlay`, `#performance`, `#mvc`, `#branch:interpolation`

**Actions effectuées :**
- Ajouté `OverlayService.update_overlay_slice` pour réutiliser le cache et ne recalculer qu’une slice, en copiant uniquement les labels concernés et en purgeant les labels vides.
- Étendu `AnnotationModel.set_slice_mask(invalidate_cache=False)` et `AnnotationController.refresh_overlay(changed_slice)` qui calcule les `changed_labels` par comparaison cache/nouveau, puis pousse l’overlay sans invalider le cache sur les mises à jour mono-slice.
- `VolumeView.set_overlay` accepte `changed_slice/changed_labels` et limite les uploads VisPy aux labels concernés (déferables via timer), tout en continuant à retomber sur le rebuild complet si forme/profondeur incohérente.

**Contexte :**
La modification d’une slice endview déclenchait un upload 3D complet de l’overlay, ce qui ralentissait fortement l’UI. Le flux s’appuie maintenant sur le cache d’overlay pour mettre à jour uniquement la slice éditée, en propulsant des mises à jour ciblées vers la vue 3D et la slice 2D sans reconstruire tout le volume.

**Décisions techniques :**
1. Conserver le cache lors des écritures slice-only et ne l’invalider que pour les opérations volume (apply_volume), afin d’éviter un rebuild systématique.
2. Détecter les labels réellement modifiés sur la slice (`changed_labels`) en comparant cache/nouveau pour réduire les uploads GPU ; fallback rebuild complet si le cache est absent ou incompatible.
3. Utiliser des uploads partiels dans `VolumeView` (gating par `labels_to_push` + timer) pour éviter de réuploader les volumes overlay inchangés tout en gardant un chemin de secours en cas de mismatch de shape.

---

### 2025-12-11 - Toggle tools panel, purge ROI non persistantes, threshold par défaut 50

**Tags :** `#controllers/master_controller.py`, `#controllers/annotation_controller.py`, `#models/roi_model.py`, `#models/view_state_model.py`, `#ui`, `#roi`, `#threshold`, `#tools_panel`, `#mvc`, `#branch:interpolation`

**Actions effectuées :**
- Rendu l’action menu Affichage > Toggle tools panel cochable et reliée à `_on_toggle_tools_panel`, avec synchro de l’état via `dockWidget_2.visibilityChanged` et initialisation du check depuis la visibilité du dock.
- Ajouté `RoiModel.clear_non_persistent()` et appelé après `on_apply_temp_mask_requested` lorsque la persistance ROI est décochée, pour purger les ROIs non persistantes et nettoyer l’aperçu.
- Défini le threshold par défaut à 50 dans `ViewStateModel` et poussé cette valeur dans le ToolsPanel lors du wiring via `set_threshold_value` pour afficher 50 dès l’ouverture.

**Contexte :**
Le menu Affichage devait réellement masquer/afficher le ToolsPanel et refléter l’état du dock. Les ROIs temporaires ne devaient pas rester après application quand la persistance est désactivée. Le workflow ROI/grow manquait d’une valeur initiale cohérente pour le seuil, d’où la valeur par défaut à 50 synchronisée modèle/vue.

**Décisions techniques :**
1. S’appuyer sur `visibilityChanged` pour maintenir QAction et dock en cohérence afin d’éviter des inversions d’état après toggles multiples.
2. Considérer les ROIs non persistantes comme purement temporaires et les purger juste après application pour éviter les ré-applications involontaires.
3. Centraliser la valeur par défaut du threshold dans le modèle et la pousser à l’UI au setup pour conserver la cohérence modèle/vue sans émettre de signaux initiaux.

---

### 2025-12-11 - Suppression de labels via Overlay Settings

**Tags :** `#views/overlay_settings_view.py`, `#controllers/annotation_controller.py`, `#controllers/master_controller.py`, `#models/annotation_model.py`, `#models/temp_mask_model.py`, `#models/roi_model.py`, `#ui`, `#labels`, `#overlay`, `#roi`, `#mvc`, `#branch:interpolation`

**Actions effectuées :**
- Ajouté un bouton « Supprimer » par ligne de label dans Overlay Settings avec signal `label_deleted`; la ligne est retirée de la liste puis l’événement est propagé.
- Implémenté la suppression de label côté modèles (`AnnotationModel.remove_label`, `TempMaskModel.remove_label`, `RoiModel.remove_label`) en purgeant palette/visibilité et en remplaçant les voxels du label par 0 (masque principal et temporaire, coverage mis à jour).
- Ajouté `AnnotationController.on_label_deleted` pour orchestrer la suppression (masques, temp mask, ROIs), réinitialiser l’`active_label` si besoin, nettoyer les previews ROI et rafraîchir l’overlay; connecté dans `MasterController` pour resynchroniser le ToolsPanel.

**Contexte :**
Le besoin était de retirer proprement des labels depuis la fenêtre Overlay Settings : supprimer le label doit aussi effacer son empreinte dans les masques et ROIs afin d’éviter une réapparition dans l’overlay ou les previews. Le ToolsPanel doit refléter la liste à jour des labels.

**Décisions techniques :**
1. Purger le masque principal et temporaire lors de la suppression pour éviter des volumes overlay fantômes; invalider le cache overlay implicitement via la remise à zéro.
2. Supprimer les ROIs associées au label pour empêcher toute réinjection lors des reconstructions ROI/volume.
3. Propager l’événement via un signal dédié (`label_deleted`) et resynchroniser le ToolsPanel/active_label pour garder la cohérence modèle/vue sans dépendre d’un rebuild manuel de la liste UI.

---

### 2025-12-11 - Redimension endview par scaling (déformation visuelle)

**Tags :** `#views/endview_view.py`, `#views/endview_resize_dialog.py`, `#controllers/master_controller.py`, `#ui`, `#resize`, `#mvc`, `#branch:annotation`

**Actions effectuées :**
- Ajouté `set_display_size`/`get_display_size` dans `EndviewView` pour stocker une taille d’affichage cible et appliquer un transform de scène (scale_x/scale_y) calculé depuis la sceneRect, combiné au zoom utilisateur (`_zoom_factor`), avec recentrage après `resetTransform`.
- L’action menu `actionResize_endview` ouvre toujours `EndviewResizeDialog` et appelle `set_display_size` sur l’annotation view; le dialog conserve largeur/hauteur avec option carré.

**Contexte :**
Besoin de déformer visuellement l’endview 2D (étirer/compresser) sans changer les données ni les modèles. Le scaling est appliqué sur la vue pour étirer l’image et les overlays tout en gardant la correspondance des coordonnées scène/volume.

**Décisions techniques :**
1. Utiliser le transform du `QGraphicsView` (scale_x/scale_y depuis la taille cible vs sceneRect) plutôt qu’un simple resize de widget afin de déformer l’image sans resampler les données.
2. Conserver un `_zoom_factor` séparé pour composer zoom utilisateur et déformation, évitant des sauts lors des zooms après redimensionnement.
3. Recentre la vue après `resetTransform` pour éviter les sauts visuels lors de la réapplication du scale.

---

### 2025-12-11 - Affichage distance locale sur C-scan

**Tags :** `#views/cscan_view.py`, `#controllers/cscan_controller.py`, `#cscan`, `#distance`, `#ui`, `#mvc`, `#branch:annotation`

**Actions effectuées :**
- Ajouté une mise à jour du bandeau CScanView pour afficher le point courant (Z, X) et la valeur du pixel sélectionné; format en mm si une échelle est fournie, sinon en pixels, avec fallback '-' sur NaN.
- Introduit un paramètre `value_scale_mm` dans `set_projection` (vue standard/corrosion) et conversion affichée `mm (px)` quand disponible; status rafraîchi à chaque `_update_cursor`.
- Calculé dans `CScanController` la résolution mm/px de l’axe ultrasound à partir des positions NDE (axis "Ultrasound" si présent, sinon axis_order[1]) via la médiane des diff absolues, puis injectée dans la vue corrosion.

**Contexte :**
Le besoin est d’afficher la distance calculée au pixel cliqué sur le C-scan (shift+clic), en privilégiant les millimètres lorsque la résolution NDE est disponible. La projection corrosion reste en pixels mais le facteur mm/px (axe ultrasound) est appliqué pour l’affichage, tandis que la vue standard continue d’afficher la valeur brute.

**Décisions techniques :**
1. Utiliser l’axe ultrasound issu de `metadata.positions`/`axis_order` comme référence pour le facteur mm/px; fallback sur axis_order[1] pour conserver une correspondance avec l’axe vertical (Y) de la carte Z×X.
2. Séparer l’échelle d’affichage (`value_scale_mm`) de la projection pour ne pas modifier les données ni le LUT; l’échelle est optionnelle afin de conserver le comportement pixels quand aucune résolution fiable n’est fournie.
3. Formater le statut en `Z=.. · X=.. · dist=..` avec conversion mm et valeur pixel entre parenthèses pour transparence, et ignorer les valeurs non finies pour éviter des affichages incohérents.

---

### 2025-12-11 - Taille pinceau Paint + Entrée applique tous les masques

**Tags :** `#views/tools_panel.py`, `#views/annotation_view.py`, `#controllers/annotation_controller.py`, `#controllers/master_controller.py`, `#ui_mainwindow.py`, `#roi`, `#paint`, `#shortcuts`, `#mvc`, `#branch:annotation`

**Actions effectuées :**
- Ajouté un slider `horizontalSlider_3` (label_6) dans ToolsPanel pour régler le rayon du pinceau paint; signal `paint_size_changed` propagé au ViewStateModel puis à AnnotationView qui régénère le curseur et au contrôleur qui l’utilise pour le disk mask.
- Intégré la taille du pinceau dans ViewStateModel (`paint_radius`), avec setter dédié; AnnotationController récupère ce rayon pour `build_disk_mask` et initialise la vue au démarrage.
- Ajouté un raccourci global Enter/Return dans MasterController qui appelle `on_apply_all_temp_masks_requested`, forçant l’application des masques temporaires sur tout le volume puis restaurant l’état apply_volume.

**Contexte :**
Le mode paint devait permettre de choisir la grosseur du pinceau et un raccourci clavier pour appliquer simultanément tous les masques temporaires sur l’ensemble des slices/endviews. Le slider existant (label_6/horizontalSlider_3) a été câblé dans le pipeline ToolsPanel → ViewStateModel → AnnotationController → AnnotationView, et Enter/Return déclenche maintenant une application globale des masques.

**Décisions techniques :**
1. Stocker le rayon dans le modèle de vue pour que contrôleur et vue partagent la même source de vérité; rayon borné à [1,50] via le slider, valeur initiale 8.
2. Invalider/reconstruire le curseur cercle dans AnnotationView à chaque changement de rayon pour refléter visuellement la taille choisie.
3. Implémenter l’application globale en activant temporairement `apply_volume` lors de l’appel Enter, afin de réutiliser la logique existante d’application volume puis de restaurer l’état précédent sans modifier l’UI.

---

### 2025-12-11 - Seuil ROI jusqu’à 255

**Tags :** `#views/tools_panel.py`, `#threshold`, `#ui`, `#mvc`, `#branch:annotation`

**Actions effectuées :**
- Configuré le slider de threshold (horizontalSlider) dans ToolsPanel pour couvrir 0–255, avec valeur initiale 50, et connexion du signal après réglage des bornes.

**Contexte :**
Le seuil devait pouvoir monter jusqu’à 255 pour le mode ROI/paint. Le slider n’avait pas de bornes explicites; on les fixe pour aligner avec la plage 8 bits.

**Décisions techniques :**
1. Bornage explicite min=0, max=255 et valeur de départ 50 afin d’éviter des valeurs par défaut Qt (0–99) insuffisantes.
2. Le branchement du signal `valueChanged` reste identique, le label continue d’afficher la valeur sélectionnée.

---

### 2025-12-12 - Options d'export NPZ (rotation + miroir)

**Tags :** `#views/overlay_export_dialog.py`, `#controllers/annotation_controller.py`, `#services/overlay_export.py`, `#overlay`, `#npz`, `#mvc`, `#branch:annotation`

**Actions effectuées :**
- Ajouté `OverlayExportDialog` avec combobox rotation 0/90/180/270 et checkbox miroir vertical (gauche/droite), renvoyant `OverlayExportOptions` (mirror + rotation).
- Étendu `OverlayExport.save_npz` pour accepter `rotation_degrees` (validation 0/90/180/270) et appliquer `np.rot90` slice-wise sur axes (H,W) avant un éventuel `np.flip` sur l’axe W.
- `AnnotationController.save_overlay_via_dialog` ouvre désormais le dialog d’options, récupère rotation/miroir et les passe au service d’export NPZ après validation de shape.

**Contexte :**
Le menu Overlay > Exporter .npz doit proposer des transformations simples avant écriture (miroir vertical, rotations multiples de 90°) sans modifier le workflow existant. La rotation est appliquée sur chaque slice (axes H/W) puis le flip éventuellement, après les validations de shape.

**Décisions techniques :**
1. Rotation slice-wise via `np.rot90(..., axes=(1,2))` pour rester cohérent avec un volume (Z,H,W) et conserver l’ordre des slices.
2. Ordre des transformations: rotation puis miroir pour un comportement déterministe; erreur si la rotation demandée n’est pas dans {0,90,180,270}.

---

### 2026-01-05 - Affichage pièce 3D corrosion (volume solide)

**Tags :** `#services/cscan_corrosion_service.py`, `#controllers/master_controller.py`, `#views/piece3d_view.py`, `#ascan_debug_log.txt`, `#corrosion`, `#3d-visualization`, `#vispy`, `#pyqt6`, `#numpy`, `#mvc`, `#branch:annotation`

**Actions effectuées :**
- Étendu l'analyse corrosion pour produire des volumes solides brut + interpolé et les exposer dans `CorrosionAnalysisResult`/workflow.
- Ajouté `Piece3DView` (VisPy iso) avec colormap métal, overlays désactivés et seuil iso configuré.
- Ouverture d’une fenêtre flottante dans `MasterController` pour afficher la pièce 3D et basculer brut/interpolé.
- Mis à jour une entrée de timing dans `ascan_debug_log.txt`.

**Contexte :**
L’objectif est de visualiser la pièce corrosion en 3D à partir des masques (frontwall/backwall) et permettre la comparaison entre volumes bruts et interpolés dans une vue dédiée, sans overlays 2D.

**Décisions techniques :**
1. Construire un volume solide en remplissant entre les extrêmes des classes A/B par colonne X et slice Z pour obtenir un volume 0/1 en float32.
2. Privilégier l’interpolé quand disponible, avec un bouton de bascule pour comparer au brut.
3. Dériver la vue de `VolumeView` en mode iso, colormap métal, overlays désactivés et depth test activé pour un rendu solide.

---
### 2026-01-08 - Refonte complète du nde_loader.py avec support multi-types

**Tags :** `#services/nde_loader.py`, `#utils/extract_data_from_nde.py`, `#utils/nde_versions_helper.py`, `#utils/sectorial_nde.py`, `#refactoring`, `#mvc`, `#branch:main`

**Actions effectuées :**
- Refonte complète de `services/nde_loader.py` (~1100 lignes) pour utiliser 100% du code des fichiers utils
- Import et intégration de `nde_versions_helper.py` : classes `Unistatus_NDE_3_0_0`, `Unistatus_NDE_4_0_0_Dev`, fonction `get_unistatus()`
- Import et intégration de `sectorial_nde.py` : classe `SectorialScanNDE`, types `SScanSliceInfo`, fonctions de conversion cartésienne
- Implémentation des classes d'extraction par type de données :
  - `NDEGroupData` : classe de base
  - `NDEGroupDataZeroDegUT` : UT standard 0°
  - `NDEGroupDataSectorialScan` : scans sectoriels avec reconstruction cartésienne
  - `NDEGroupDataTFM` : Total Focusing Method (transpose z,y,x → z,x,y)
  - `NDEGroupDataFMC` : Full Matrix Capture (reshape pulser/receiver)
- Ajout de `NDEDataTypeCheck` : détection automatique du type (is_sectorial_scan, is_tfm, is_fmc)
- Ajout de `_reorder_axes_by_metadata()` : réordonnancement des axes selon uCoordinateOrientation
- Ajout de `_rotate_clockwise()` : rotation 90° horaire finale sur chaque slice

**Contexte :**
L'ancien `nde_loader.py` était un loader minimal. Le nouveau utilise intégralement le code d'extraction existant dans utils/ pour supporter tous les types de données NDE (zero_deg, sectorial, tfm, fmc) tout en maintenant la compatibilité avec `NdeModel`.

**Décisions techniques :**
1. **Stratégie d'orientation** : Si `uCoordinateOrientation` existe ("around" ou "length"), U est l'axe slice. Sinon, l'axe avec le plus d'éléments (U ou V) devient l'axe slice.
2. **Rotation 90° CW** : Appliquée systématiquement à la fin du pipeline via `np.rot90(data, k=-1, axes=(1, 2))` pour orientation display-ready.
3. **Ordre du pipeline** : Chargement → Détection type → Extraction → Réordonnancement axes → Rotation 90° → NdeModel

---### **2026-01-09** — Seuil ROI robuste et labels persistants NDE/NPZ

**Tags :** `#services/annotation_service.py`, `#models/annotation_model.py`, `#controllers/annotation_controller.py`, `#controllers/master_controller.py`, `#roi`, `#threshold`, `#overlay`, `#labels`, `#mvc`, `#branch:annotation`

**Actions effectuees :**
- Rendu le threshold ROI rectangle robuste aux outliers via normalisation percentile (1/99) + clipping dans `build_thresholded_mask`, avec filtrage NaN/inf.
- Ajoute l'option `preserve_labels` a `AnnotationModel.set_mask_volume` pour garder palette/visibilite et fusionner les classes manquantes d'un NPZ.
- Ajoute `preserve_labels` a `AnnotationController.reset_overlay_state` et adapte le chargement NDE/NPZ pour conserver les labels du tools panel et resynchroniser l'overlay settings.

**Contexte :**
Un pixel aberrant dans une ROI avec threshold faible faisait disparaitre la selection. Les labels disparaissaient du tools panel lors du chargement d'un NDE alors qu'ils doivent persister, et le NPZ doit seulement ajouter ses classes manquantes.

**Decisions techniques :**
1. Normaliser la ROI avec percentiles 1/99 plutot que min/max pour rester relatif tout en limitant l'impact des outliers.
2. Preserver les labels lors des resets NDE/NPZ (masques/ROI vides) et re-synchroniser la fenetre overlay plutot que tout vider.
3. Ne pas modifier le flow nnUNet.

---
### 2026-01-14 - Optimisation du grow et ajout du mode Box-Grow

**Tags :** `#services/annotation_service.py`, `#ui_mainwindow.py`, `#untitled.ui`, `#region-growing`, `#ui`, `#pyqt6`, `#numpy`, `#skimage`, `#scipy`, `#mvc`, `#branch:annotation`

**Actions effectuées :**
- Optimise le grow mono-seed avec `skimage.morphology.flood` et un masque valide combinant threshold, restriction et blocage.
- Optimise le grow multi-seeds via `scipy.ndimage.label` en sélectionnant les composantes connectées touchées par les seeds.
- Ajoute l’option UI "Box-Grow" (radio) dans `ui_mainwindow.py` et `untitled.ui`.

**Contexte :**
Les opérations de grow étaient effectuées via un BFS Python. Le changement remplace cette logique par des opérations vectorisées (flood + label) pour gagner en performance, tout en respectant le threshold, la restriction et le blocage des pixels. L’UI expose un mode Box-Grow pour l’utilisateur.

**Décisions techniques :**
1. Construire un `valid_mask` unique (threshold + restriction + blocage) pour centraliser les conditions et réduire les branches.
2. Utiliser `flood` pour le seed unique et `label` pour regrouper efficacement les régions multi-seeds.
3. Ajouter un radio button dédié pour rendre le mode Box-Grow explicite dans la vue.

---
### 2026-01-13 - Blocage des pixels deja masques pour ROI

**Tags:** `#annotation_controller.py`, `#annotation_service.py`, `#roi`, `#blocage`, `#apply-volume`, `#temp-mask`, `#mvc`, `#numpy`, `#branch:annotation`

**Actions effectuees:**
- Ajoute un helper `_build_blocked_mask` (annotation + temp) et l'utilise pour box/grow/line/paint ainsi que apply-volume/rebuild.
- Etend AnnotationService pour accepter `blocked_mask`/`blocked_mask_provider` et filtrer box (apres threshold) + grow/line (barriere).
- Maintient `restriction_mask` et applique le blocage uniquement pour `label != 0`.

**Contexte:**
Eviter d'ecraser des pixels deja masques lors des ROIs, tout en laissant le label 0 effacer librement; le blocage combine l'annotation existante et, selon le cas, la couverture temporaire.

**Decisions techniques:**
1. Le blocage n'est applique que pour `label != 0` afin de conserver le comportement d'effacement du label 0.
2. `include_temp=False` lors des rebuild/apply-volume pour ne pas auto-bloquer la reconstruction; `include_temp=True` en edition interactive.
3. Le mask est normalise cote service pour garantir un comportement coherent entre box/grow/line et les propagations volume.

---
### 2026-01-13 - Zone de restriction ROI globale (Alt drag)

**Tags:** `#views/annotation_view.py`, `#controllers/annotation_controller.py`, `#services/annotation_service.py`, `#models/view_state_model.py`, `#controllers/master_controller.py`, `#roi`, `#restriction`, `#grow`, `#line`, `#box`, `#paint`, `#mvc`, `#pyqt6`, `#branch:annotation`

**Actions effectuees:**
- Ajoute un rectangle de restriction global dans la vue avec overlay en pointilles et edition Alt+drag (deplacement + resize).
- Stocke la restriction dans ViewStateModel et l'initialise a pleine endview au chargement volume via MasterController.
- Applique le mask de restriction a tous les modes ROI (grow/line/box/paint) et aux rebuild/propagations volume; les pixels hors zone sont ignores.
- Filtre les seeds/ROIs et clippe les boxes/disk masks pour respecter la zone pendant la preview.

**Contexte:**
Besoin d'une zone de dessin globale, editable directement dans l'endview, pour empecher toute segmentation en dehors de la zone.

**Decisions techniques:**
1. Utiliser un rectangle global (full-frame par defaut) pour rester simple et rapide a manipuler.
2. Forcer le clipping au niveau service (region growing) et controller (box/paint) pour garantir l'effet sur tous les modes.
3. Interaction Alt+drag pour ne pas ajouter un nouveau mode d'outil.

---
### 2026-01-13 - Ajout ROI line (ligne libre)

**Tags:** `#views/tools_panel.py`, `#controllers/master_controller.py`, `#views/annotation_view.py`, `#controllers/annotation_controller.py`, `#models/roi_model.py`, `#services/annotation_service.py`, `#roi`, `#line`, `#grow`, `#mvc`, `#pyqt6`, `#branch:annotation`

**Actions effectuees:**
- Ajoute le mode outil "line" dans ToolsPanel et le wiring du radio Designer via MasterController (tool_mode="line").
- AnnotationView: capture press-drag-release, preview de la ligne via QGraphicsPathItem et emission de `line_drawn`.
- RoiModel: ajout du type `line` et stockage des points; AnnotationService: rasterise la polyline en graines puis applique un grow multi-seeds (rebuild/propagation volume inclus).
- AnnotationController: applique la ROI line avec seuil/label/persistance/apply-volume puis rafraichit la preview ROI.

**Contexte:**
Besoin d'un mode ROI line similaire au grow: tracer une ligne libre pour semer plusieurs graines et lancer la germination sur les pixels touches.

**Decisions techniques:**
1. Reutiliser la logique grow en multi-graines via la rasterisation de la polyline (Bresenham) pour garder un comportement coherent avec le seuil.
2. Garder le flux MVC: dessin/preview en vue, orchestration dans le controleur, logique grow/rasterisation dans le service.
3. Integrer le mode line aux chemins apply-volume et rebuild ROI pour conserver persistance et preview.

---
### 2026-01-12 - Range d'application au volume et decouplage ROI box/persistance

**Tags :** `#views/nde_settings_view.py`, `#controllers/master_controller.py`, `#controllers/annotation_controller.py`, `#models/view_state_model.py`, `#services/annotation_service.py`, `#apply-volume`, `#roi`, `#settings`, `#mvc`, `#branch:annotation`

**Actions effectuees :**
- Ajoute un range "Appliquer au volume (de/a)" dans la fenetre Parametres via `NdeSettingsView`, avec signaux et setters de bornes/valeurs.
- Ajoute `apply_volume_start/end` dans `ViewStateModel` et synchronise le range depuis `MasterController` en forcant l'inclusion de la slice courante.
- Limite l'application au volume au range (rebuild/apply/propagate grow) et applique la ROI box sur le range quand apply-volume est actif, sans depender de la persistance.

**Contexte :**
Le checkbox "Appliquer au volume" devait etre independant de la persistance des ROI et permettre d'appliquer uniquement un intervalle de slices (incluant la slice courante) au lieu du volume entier.

**Decisions techniques :**
1. Garde le range dans le modele de vue pour le partager entre UI et logique, avec clamp aux bornes volume.
2. Applique le range au niveau service pour la reconstruction et l'application des masques temporaires.
3. Pour la ROI box, appliquer le masque sur le range sans rendre les ROI persistantes.

---
### 2026-01-14 - Consolidation des helpers NDE dans nde_loader

**Tags :** `#services/nde_loader.py`, `#utils/nde_versions_helper.py`, `#utils/sectorial_nde.py`, `#utils/extract_data_from_nde.py`, `#refactoring`, `#nde_loader`, `#mvc`, `#numpy`, `#hdf5`, `#scipy`, `#branch:annotation`

**Actions effectuées :**
- Intégré les helpers de version (`Unistatus_*`, `get_unistatus`) et le traitement sectoriel (`SectorialScanNDE`, `SScanSliceInfo`, `s_scan_to_cartesian_image_extremes_fast`) directement dans `services/nde_loader.py`.
- Nettoyé les imports inutilisés liés aux helpers sectoriels non appelés.
- Supprimé les fichiers utils redondants/legacy (`utils/nde_versions_helper.py`, `utils/sectorial_nde.py`, `utils/extract_data_from_nde.py`).

**Contexte :**
Consolidation demandée pour centraliser le pipeline NDE, réduire les dépendances utils non utilisées et préserver le comportement existant dans un seul service.

**Décisions techniques :**
1. Conserver la logique existante en la déplaçant telle quelle dans `services/nde_loader.py` pour éviter toute régression fonctionnelle.
2. Supprimer les helpers non utilisés et les modules utils obsolètes pour clarifier la surface API.

---
### 2026-01-14 - Optimisation du region growing

**Tags :** `#services/annotation_service.py`, `#region-growing`, `#numpy`, `#skimage`, `#scipy`, `#mvc`, `#branch:annotation`

**Actions effectuées :**
- Optimise le grow mono-seed avec `skimage.morphology.flood` et un masque valide combinant threshold, restriction et blocage.
- Optimise le grow multi-seeds via `scipy.ndimage.label` en sélectionnant les composantes connectées touchées par les seeds.

**Contexte :**
Les opérations de grow étaient effectuées via un BFS Python. Le changement remplace cette logique par des opérations vectorisées (flood + label) pour gagner en performance, tout en respectant le threshold, la restriction et le blocage des pixels.

**Décisions techniques :**
1. Construire un `valid_mask` unique (threshold + restriction + blocage) pour centraliser les conditions et réduire les branches.
2. Utiliser `flood` pour le seed unique et `label` pour regrouper efficacement les régions multi-seeds.

---
### 2026-01-16 - Overlay 3D refactor vers masque uint8 + LUT

**Tags :** `#controllers/annotation_controller.py`, `#models/overlay_data.py`, `#services/overlay_service.py`, `#views/endview_view.py`, `#views/volume_view.py`, `#overlay`, `#volume-rendering`, `#vispy`, `#mvc`, `#optimization`, `#branch:annotation`

**Actions effectuées :**
- Remplacé `OverlayData` pour transporter `mask_volume` (uint8, Z/H/W) et marqué `label_volumes` comme déprécié.
- Simplifié `OverlayService` : plus de volumes alpha par label, `build_overlay_data` renvoie le masque brut + palette, `update_overlay_slice` délègue à la reconstruction simple.
- Refondu l’overlay dans `VolumeView` et `EndviewView` : LUT RGBA par palette/visibilités, overlay 2D via LUT, overlay 3D via un seul `VolumeVisual` basé sur le masque uint8 (cmap + clim).
- Ajusté `AnnotationController` : cache reconstruit avec `mask_volume`, suppression de l’optimisation `changed_labels`, log basé sur la palette.

**Contexte :**
Réduction de la mémoire CPU/GPU et simplification du pipeline overlay 3D en évitant la génération de volumes alpha par label, tout en conservant palette/visibilités.

**Décisions techniques :**
1. Utiliser un masque uint8 unique + LUT pour toutes les vues afin de réduire la mémoire et accélérer les mises à jour.
2. Centraliser l’overlay 3D sur un seul `VolumeVisual` avec colormap discrète et updates différées.
3. Accepter un rendu discret (moins de texture alpha) au profit de la stabilité et des performances.

---
### 2026-01-16 - Slice indicator 3D: plan RGBA constant

**Tags :** `#views/volume_view.py`, `#vispy`, `#volume-rendering`, `#overlay`, `#mvc`, `#branch:annotation`

**Actions effectuées :**
- Remplacé l’image de slice (colormap) par un plan RGBA constant (1x1 pixel noir translucide) étiré sur toute la slice.
- Désactivé le colormap sur `self._slice_image` et forcé `depth_test=False` pour garder l’indicateur visible au-dessus des overlays.
- Ajusté l’ordre de rendu (`order=11`) et le transform pour scaler la texture 1x1 à la taille complète (width/height).

**Contexte :**
Le slice indicator devait rester visible même avec l’overlay 3D et ne pas dépendre du colormap de base.

**Décisions techniques :**
1. Utiliser un pixel RGBA constant + transform de scale pour un indicateur stable et léger.
2. Couper le depth test pour garantir la visibilité par-dessus l’overlay.

---
### 2026-01-20 - Heuristique axe de coupe NDE

**Tags :** `#services/nde_loader.py`, `#nde_loader`, `#mvc`, `#branch:annotation`

**Actions effectuees :**
- Ajoute une heuristique pour choisir l'axe de coupe: V si V > 2x U, sinon U par defaut.
- Ajuste les messages de debug pour expliciter le choix (V >> U ou preference par defaut).

**Contexte :**
Le choix automatique de l'axe de coupe devait mieux distinguer les cas ou V domine nettement, tout en conservant une preference stable quand les dimensions sont proches.

**Decisions techniques :**
1. Utiliser un seuil 2x pour preferer V quand l'ecart est significatif.
2. Conserver U comme choix par defaut pour respecter la structure des fichiers.

---
### 2026-01-25 - Effacement label 0 cible via Parametres

**Tags :** `#views/nde_settings_view.py`, `#controllers/master_controller.py`, `#models/view_state_model.py`, `#controllers/annotation_controller.py`, `#services/annotation_service.py`, `#settings`, `#label0`, `#roi`, `#apply-volume`, `#mvc`, `#branch:annotation`

**Actions effectuees :**
- Ajoute un menu deroulant "Effacement label 0" dans Parametres, alimente par les labels existants et synchro avec l'etat.
- Stocke la cible d'effacement du label 0 dans ViewStateModel et la met a jour via MasterController.
- Applique la restriction d'effacement au label 0 pour paint/grow/line/box et les rebuilds ROI (incluant apply-volume) via un blocked mask par label.

**Contexte :**
Le besoin est de pouvoir effacer uniquement un label specifique avec le label 0, tout en conservant l'option "Tous" pour le comportement historique.

**Decisions techniques :**
1. Utiliser un blocked mask dedie au label 0 pour ne laisser passer que le label cible (None = aucun filtre).
2. Passer la cible via Parametres pour rester coherents avec les autres reglages (colormap, apply-volume).

---
### 2025-02-14 - Prefixe/suffixe pour split flaw/noflaw

**Tags :** `#services/split_service.py`, `#controllers/master_controller.py`, `#split_flaw_noflaw`, `#export`, `#ui`, `#mvc`, `#pyqt6`, `#branch:annotation`

**Actions effectuees :**
- Ajoute des parametres optionnels de prefixe/suffixe au service de split et les applique aux fichiers flaw/noflaw et gtmask sans modifier les endviews complete.
- Ajoute deux prompts (prefixe, suffixe) dans MasterController et transmet les valeurs au service.

**Contexte :**
Besoin de personnaliser le nom des images exportees pour le dataset, tout en conservant les noms sources des endviews complete pour la copie.

**Decisions techniques :**
1. Appliquer prefixe/suffixe uniquement aux sorties flaw/noflaw + gtmask pour conserver la correspondance avec les fichiers complete.
2. Utiliser QInputDialog pour une saisie simple et garder des valeurs vides par defaut.

---
### 2026-01-26 - Controle opacite overlay endview/3D

**Tags :** `#views/overlay_settings_view.py`, `#views/endview_view.py`, `#views/volume_view.py`, `#controllers/annotation_controller.py`, `#controllers/master_controller.py`, `#models/view_state_model.py`, `#overlay`, `#opacity`, `#ui`, `#3d`, `#mvc`, `#branch:annotation`

**Actions effectuees :**
- Ajoute un slider d'opacite (0-100%) dans OverlaySettingsView avec signal et synchro UI.
- Stocke l'opacite globale dans ViewStateModel et la propage via AnnotationController/MasterController (init + switch session).
- Applique l'opacite aux overlays Endview (QGraphicsPixmapItem) et 3D (LUT alpha + overlay slice).

**Contexte :**
Besoin de controler l'opacite des labels overlay dans les vues endview et 3D directement depuis les parametres overlay.

**Decisions techniques :**
1. Centraliser l'opacite dans ViewStateModel et la reappliquer a l'ouverture des parametres et aux switches de session.
2. Utiliser un facteur global multiplie sur l'alpha de la palette pour garder les alphas par label tout en offrant un controle absolu; valeur par defaut 0.4 pour conserver le rendu precedent.

---
### 2026-01-26 - Seuil ROI box sans percentiles (normalisation slice complete)

**Tags :** `#services/annotation_service.py`, `#roi`, `#threshold`, `#box`, `#mvc`, `#numpy`, `#branch:annotation`

**Actions effectuees :**
- Retire le clipping percentile (1/99) dans `build_thresholded_mask` pour la ROI box.
- Aligne le seuillage box sur la normalisation de la slice complete [0-255], et applique le threshold uniquement dans la box.

**Contexte :**
Le threshold devait etre applique de facon fidele aux pixels selectionnes, sans lissage/percentile local de la zone ROI.

**Decisions techniques :**
1. Utiliser `_normalize_slice_to_uint8` comme grow/line pour un seuil global cohérent.
2. Conserver un fallback simple (masque box brut) si la normalisation echoue.

---

### 2026-01-27 - Pruning lignes fines pour ROI + reglage UI

**Tags :** `#services/annotation_service.py`, `#controllers/annotation_controller.py`, `#controllers/master_controller.py`, `#models/view_state_model.py`, `#views/nde_settings_view.py`, `#roi`, `#opencv`, `#ui`, `#mvc`, `#branch:annotation`

**Actions effectuees :**
- Ajoute un parametre global `roi_thin_line_max_width` dans `ViewStateModel` et sa mise a jour via `MasterController` depuis les parametres NDE.
- Expose un `QSpinBox` dans `NdeSettingsView` pour regler la largeur max (px) et emet un signal `roi_thin_line_width_changed` synchronise a l’ouverture.
- Propage `thin_line_max_width` dans les flows grow/line/volume via `AnnotationController` jusqu’au service.
- Integre `_prune_thin_lines` dans `AnnotationService` (morpho open via OpenCV) et remplace le labeling SciPy par `cv2.connectedComponents` pour accelerer la segmentation.

**Contexte :**
Besoin de filtrer automatiquement les traits trop fins lors des ROI grow/line pour eviter la propagation sur des artefacts, tout en gardant un reglage simple dans les parametres.

**Decisions techniques :**
1. Utiliser une ouverture morphologique horizontale (kernel largeur = max_width+1) pour supprimer les lignes trop fines sans casser les regions larges.
2. Basculer sur `cv2.connectedComponents` (connectivite 4) pour maintenir le comportement de labeling tout en reduisant le cout de calcul.

---
### 2026-01-27 - Percentiles conditionnels pour seuil ROI box

**Tags :** `#services/annotation_service.py`, `#controllers/annotation_controller.py`, `#roi`, `#threshold`, `#box`, `#percentiles`, `#mvc`, `#branch:annotation`

**Actions effectuees :**
- Reintroduit le clipping percentile (1/99) pour la ROI box via un flag `use_box_percentiles` dans `build_thresholded_mask`.
- Propage le flag dans `apply_box_roi`, `apply_box_roi_to_range`, `rebuild_temp_masks_for_slice` et `rebuild_volume_preview_from_rois` pour couvrir preview slice + volume.
- Lie l'option a `view_state_model.threshold_auto` (checkbox "Box percentiles") dans `AnnotationController`.

**Contexte :**
Le percentile de la ROI box avait ete retire. Il est maintenant reintroduit uniquement quand l'utilisateur coche "Box percentiles", afin de conserver le comportement global par defaut.

**Decisions techniques :**
1. Garder la normalisation slice complete quand l'option est decochee pour ne pas modifier le seuillage existant.
2. Limiter l'option aux ROIs box (grow/line inchanges) et reutiliser `threshold_auto` pour eviter un nouvel etat UI.

---
### 2026-01-30 - Endview resize reset + pan scaling

**Tags :** `#controllers/master_controller.py`, `#views/endview_resize_dialog.py`, `#views/endview_view.py`, `#ui`, `#endview`, `#zoom`, `#pan`

**Actions effectuees :**
- Ajoute un bouton "Par defaut" dans le dialogue de resize et expose `wants_reset`.
- MasterController applique un reset de taille d'affichage quand demande, sinon applique la taille choisie.
- EndviewView memorise les tailles min/max par defaut, corrige le pan en fonction du zoom, et ajoute `reset_display_size` pour restaurer taille/zoom/pan.

**Contexte :**
Besoin d'un retour a la taille/zoom par defaut sans forcer le resize du widget, et d'un panning coherent quand le zoom est actif.

**Decisions techniques :**
1. Piloter le reset via le dialogue et une methode `reset_display_size` cote vue.
2. Eviter de fixer la taille du widget et ajuster le pan en utilisant l'echelle de transformation.

---

### 2026-01-30 - Ajustement pinceau paint et curseur fixe
**Tags :** `#branch:annotation`, `#annotation_service.py`, `#annotation_controller.py`, `#annotation_view.py`, `#paint`, `#brush`, `#cursor`

**Actions effectu�es :**
- Autorise un rayon effectif 0 dans build_disk_mask pour obtenir 1 pixel au centre.
- Mappe la taille du slider vers un rayon effectif size-1 pour que size 2 donne 5 pixels et l'augmentation reste naturelle.
- Remplace le curseur paint par une croix fixe 13x13 pour garder une visibilite constante.

**Contexte :**
Le pinceau min devait produire 1 pixel, le size 2 devait redevenir 5 pixels, et le curseur devait rester visible sans varier avec la taille.

**Decisions techniques :**
1. Decaler le rayon effectif (size-1) au niveau du controller pour conserver le slider min a 1 sans changer l'API du modele.
2. Utiliser un curseur en croix fixe dans la vue pour dissocier l'affichage du curseur de la taille reelle du pinceau.

---
### 2026-01-30 - Endview zoom ancre et pan centré

**Tags :** `#views/endview_view.py`, `#endview`, `#zoom`, `#pan`, `#qt`, `#ui`, `#branch:annotation`

**Actions effectuées :**
- Remplace l’ancrage de transformation par un rendu sans scrollbars et conserve un centre de pan en coordonnées scène.
- Applique le zoom autour du curseur (molette vue + widget) en recalculant le centre cible et en mémorisant `_pan_center_scene`.
- Ajoute une chaîne de transformation dédiée (scale d’affichage + zoom) et met à jour la padding de scène pour garder du pan disponible après resize.
- Ajuste `reset_display_size` pour réinitialiser le centre de pan et restaurer l’état de transformation sans sauts.

**Contexte :**
Le zoom/pan devenait instable avec les scrollbars désactivées et perdait le centrage après resize ou reset. Il fallait conserver un centre logique tout en autorisant le zoom ancré sur le curseur.

**Décisions techniques :**
1. Centraliser l’état de pan via `_pan_center_scene` plutôt que les scrollbars, pour un comportement déterministe.
2. Séparer l’échelle d’affichage (fit) du facteur de zoom, puis recalculer le centre et la sceneRect à chaque interaction.

---
### 2026-01-30 - Toggle overlay applique sur Endview
**Tags :** `#branch:annotation`, `#controllers/annotation_controller.py`, `#overlay`, `#endview`, `#toggle`

**Actions effectuees :**
- Efface overlay 2D quand le toggle est desactive pour rester coherent avec la 3D.
- Conditionne le push de overlay Endview a show_overlay pour eviter un affichage residuel.

**Contexte :**
Le toggle overlay masquait seulement la 3D et la Endview restait affichee.

**Decisions techniques :**
1. Nettoyer les deux vues quand show_overlay est faux pour eviter des overlays orphelins.
2. Ne pousser overlay 2D que si show_overlay est actif.

---
### 2026-01-30 - Alignement des contours Endview/ROI
**Tags :** `#branch:annotation`, `#views/annotation_view.py`, `#views/endview_view.py`, `#roi`, `#restriction`, `#crosshair`, `#ui`

**Actions effectuées :**
- Uniformise l'épaisseur des traits (1 px cosmetic) pour contour de restriction, ROI box/line et croix.
- Ajuste les rectangles (restriction/ROI box) pour inclure le pixel de droite et du bas.
- Aligne la ROI line sur le centre des pixels via un offset 0.5.

**Contexte :**
Le contour et les ROI n'étaient pas alignés sur la grille de pixels et l'épaisseur variait avec le zoom.

**Décisions techniques :**
1. Utiliser des pens « cosmetic » pour garder une épaisseur constante à 1 px.
2. Étendre largeur/hauteur des rectangles de 1 px pour inclure le bord droit/bas.
2. Décaler la ligne temporaire de 0.5 px pour coller aux centres de pixels.

---
### 2026-02-02 - Parametres corrosion + selection labels
**Tags :** `#branch:annotation`, `#controllers/master_controller.py`, `#controllers/cscan_controller.py`, `#models/view_state_model.py`, `#services/cscan_corrosion_service.py`, `#services/corrosion_label_service.py`, `#views/corrosion_settings_view.py`, `#ui_mainwindow.py`, `#untitled.ui`, `#corrosion`, `#labels`, `#mvc`, `#ui`

**Actions effectuees :**
- Ajoute un dialogue Parametres corrosion avec 2 combos de labels et signaux (nouvelle vue).
- Branche le menu Analyse > Parametres corrosion et synchronise la selection avec l'etat.
- Ajoute l'etat des labels corrosion (A/B) dans ViewStateModel et la selection par defaut sur les deux premiers labels.
- Fait passer la paire selectionnee dans le workflow corrosion et ajoute des validations (labels distincts, >0, presents, voxels).
- Deplace la logique de normalisation des labels dans un service dedie pour respecter MVC.

**Contexte :**
L'analyse corrosion devait fonctionner avec n'importe quel nombre de labels, en laissant l'utilisateur choisir 2 labels.

**Decisions techniques :**
1. Isoler la normalisation de la paire dans un service pur (CorrosionLabelService) pour garder le controleur leger.
2. Valider la paire dans le workflow corrosion afin d'eviter des analyses incoherentes.
---
### 2026-02-03 - Ancrage 3D corrosion sur centre de masse
**Tags :** `#branch:annotation`, `#services/cscan_corrosion_service.py`, `#controllers/master_controller.py`, `#views/piece3d_view.py`, `#corrosion`, `#3d`, `#pivot`, `#anchor`

**Actions effectu?es :**
- Calcule le centre de masse du volume solide 0/1 (brut puis fallback interpol?) et le propage dans le workflow corrosion.
- Stocke l'ancrage dans le contr?leur et l'applique ? la vue Piece 3D sans recalcul lors du toggle brut/interpol?.
- Centre la cam?ra de `Piece3DView` sur cet ancrage en mode ancrage volume, avec flip XY coh?rent.

**Contexte :**
Besoin d'un pivot 3D align? sur la pi?ce corrosion bas? sur la paire de labels s?lectionn?e.

**D?cisions techniques :**
1. Calculer l'ancrage ? partir du volume brut quand disponible et le conserver pour les bascules de volume.
2. Appliquer le flip XY au point d'ancrage pour rester coh?rent avec la transformation visuelle VisPy.

### 2026-02-05 - Croix C-scan cosmetique rouge
**Tags :** #branch:annotation, #views/cscan_view.py, #cscan, #crosshair, #ui, #zoom, #corrosion, #endview

**Actions effectuees :**
- Passe la croix C-scan en rouge et en pen cosmetic 1 px.
- Rend l'epaisseur constante au zoom, identique a l'Endview, pour la vue standard et la corrosion (heritage CScanView).

**Contexte :**
La croix de position C-scan devait rester cosmetique (largeur constante) et rouge, comme l'Endview, y compris en vue corrosion.

**Decisions techniques :**
1. Aligner le style sur l'Endview en utilisant un pen cosmetic 1 px.
2. Appliquer le changement dans CScanView pour couvrir aussi CscanViewCorrosion.

### 2026-02-05 - Mesure A-scan corrosion alignee sur C-scan
**Tags :** `#branch:annotation`, `#controllers/ascan_controller.py`, `#controllers/master_controller.py`, `#services/ascan_service.py`, `#views/ascan_view_corrosion.py`, `#ascan`, `#corrosion`, `#mvc`, `#ui`, `#measurement`, `#pyqtgraph`

**Actions effectuees :**
- Ajoute la vue A-scan corrosion avec overlay (deux lignes verticales paralleles, ligne de mesure horizontale, label).
- Integre un stack A-scan standard/corrosion et branche les signaux/visibilites via AScanController.
- Aligne la valeur affichee sur la distance_map corrosion (C-scan) et descend le texte.
- Deplace le calcul des indices/distance corrosion dans AScanService pour respecter MVC.

**Contexte :**
Besoin d'afficher la distance entre labels sur l'A-scan en mode corrosion et d'eviter les ecarts de 1 px avec le C-scan.

**Decisions techniques :**
1. Utiliser la distance_map corrosion comme source unique pour le label A-scan afin d'etre identique au C-scan.
2. Laisser l'overlay comme aide visuelle et centraliser le calcul (indices + distance) dans le service.

### 2026-02-06 - Corrosion overlay garde labels source
**Tags :** `#branch:annotation`, `#services/cscan_corrosion_service.py`, `#corrosion`, `#labels`, `#palette`, `#session`, `#mvc`

**Actions effectuees :**
- Supprime le mapping des IDs corrosion vers 5/6 et conserve label_a/label_b comme IDs d'overlay.
- Transmet la palette source depuis le workflow et construit overlay_palette avec ces couleurs (fallback MASK_COLORS_BGRA).
- Aligne la generation des overlays corrosion sur les IDs d'origine.

**Contexte :**
La nouvelle session corrosion devait reutiliser les memes numeros de labels et les memes couleurs que la session de depart.

**Decisions techniques :**
1. Centraliser la logique dans CScanCorrosionService pour respecter MVC et eviter une reconversion en controller.
2. Utiliser la palette de la session source avec fallback quand une couleur manque.
### 2026-02-06 - Peaks corrosion pour A-scan et overlay
**Tags :** `#branch:annotation`, `#services/distance_measurement.py`, `#services/cscan_corrosion_service.py`, `#services/ascan_service.py`, `#controllers/ascan_controller.py`, `#controllers/master_controller.py`, `#models/view_state_model.py`, `#corrosion`, `#ascan`, `#cscan`, `#overlay`, `#measurement`, `#peaks`

**Actions effectuees :**
- Ajoute un calcul vectorise des index Y max (peak maps A/B) en plus de la distance_map.
- Propage les peak maps dans le workflow corrosion et le ViewStateModel pour l'A-scan.
- Fait prioriser l'A-scan sur les peak maps et construit l'overlay corrosion depuis ces peaks.

**Contexte :**
Les lignes A-scan/C-scan ne suivaient pas toujours les vrais pics car l'overlay utilisait une moyenne Y par X.

**Decisions techniques :**
1. Utiliser des peak maps ZxX calculees sur le volume brut pour aligner A-scan et overlay.
2. Conserver la distance_map pour la valeur affichee et garder un fallback sur l'overlay si besoin.

### 2026-02-09 - Refactor UI en docks modularises avec ADS
**Tags :** `#branch:annotation`, `#.gitignore`, `#annotation.ui`, `#ascan.ui`, `#controllers/master_controller.py`, `#cscan.ui`, `#requirements.in`, `#requirements.txt`, `#toolspanel.ui`, `#ui_annotation.py`, `#ui_ascan.py`, `#ui_cscan.py`, `#ui_mainwindow.py`, `#ui_toolspanel.py`, `#ui_volume.py`, `#untitled.ui`, `#volume.ui`, `#mvc`, `#ui`, `#docking`, `#ads`, `#pyqt6`, `#qt-designer`

**Actions effectuees :**
- Decoupe l'UI monolithique en fichiers .ui dedies pour annotation, cscan, ascan, volume et tools panel, puis regenere les wrappers `ui_*.py`.
- Rebranche `MasterController` pour instancier les vues depuis ces UI separees et conserver les controlleurs existants (Annotation/CScan/AScan) sans deplacer la logique metier.
- Integre `PyQt6Ads` comme couche d'orchestration de layout (`CDockManager` + `CDockWidget`) et migre le wiring des docks vers ADS.
- Adapte le panneau tools au modele ADS: binding des widgets Designer via `attach_designer_widgets`, toggle menu via `viewToggled`/`toggleView`.
- Fixe le layout par defaut demande: Tools a gauche, Annotation a sa droite, puis Volume/AScan/CScan empiles de haut en bas a droite.
- Active les features par defaut du dock Tools pour le rendre deplacable comme les autres docks.
- Ajoute `PyQt6Ads` dans `requirements.in` et `requirements.txt`.

**Contexte :**
Le refactor UI visait a abandonner les splitters du MainWindow au profit d'une architecture full dock widget, plus flexible pour le placement et la manipulation des vues.

**Decisions techniques :**
1. Garder les vues en PyQt6/Designer pures et limiter ADS a la couche d'assemblage dans `MasterController`.
2. Preserver l'architecture MVC: aucun transfert de logique metier vers les vues, les controlleurs existants restent les points d'orchestration.
3. Conserver les stacks corrosion A-scan/C-scan dans les conteneurs de vue pour maintenir le comportement fonctionnel existant.



### 2026-02-09 - Refactor MVC des docks ADS et delegation des controllers
**Tags :** `#branch:annotation`, `#controllers/master_controller.py`, `#controllers/dock_layout_controller.py`, `#controllers/annotation_controller.py`, `#controllers/cscan_controller.py`, `#controllers/ascan_controller.py`, `#mvc`, `#ads`, `#docking`, `#orchestration`

**Actions effectuees :**
- Ajoute `DockLayoutController` pour centraliser l'assemblage ADS des docks et des stacks corrosion (A-scan/C-scan), avec binding propre du toggle Tools.
- Allege `MasterController` en supprimant la construction ADS inline et en deleguant le wiring dock/layout, les toggles, et plusieurs handlers vers les controllers respectifs.
- Etend `AnnotationController` avec `on_slice_changed`, `set_cross_visible`, `on_annotation_point_selected` et `on_annotation_drag_update` pour sortir la logique d'interaction Endview du master.
- Etend `CScanController` avec `set_colormap` et `on_crosshair_changed` pour encapsuler la synchro slice/crosshair C-scan et le point de rafraichissement A-scan.
- Etend `AScanController` avec `on_position_changed` pour encapsuler la resolution curseur A-scan vers point Endview/slice.

**Contexte :**
Suite au split UI en vues dediees et a la migration vers des docks ADS, il fallait eviter les duplications dans `MasterController` et renforcer la separation MVC en confiant la logique d'interaction aux controllers specialises.

**Decisions techniques :**
1. ADS reste une couche d'orchestration de layout seulement via `DockLayoutController`, sans importer ADS dans les vues metier.
2. `MasterController` conserve le role de coordination globale, tandis que les comportements de chaque vue sont deplaces dans `AnnotationController`, `CScanController` et `AScanController`.


### 2026-02-09 - Stabilisation OpenGL du dock Volume en mode flottant
**Tags :** `#branch:annotation`, `#controllers/dock_layout_controller.py`, `#main.py`, `#views/volume_view.py`, `#mvc`, `#ads`, `#vispy`, `#opengl`, `#docking`

**Actions effectuees :**
- Ajoute l'attribut Qt `AA_ShareOpenGLContexts` avant l'instanciation de `QApplication` pour stabiliser le partage de contexte OpenGL entre fenetres dockees/flottantes.
- Connecte `volume_dock.topLevelChanged` dans `DockLayoutController` pour detecter les transitions dock <-> flottant du dock Volume.
- Ajoute `notify_dock_topology_changed` dans `VolumeView` avec timer dedie pour relancer un rebuild de scene apres reparenting.
- Rebuild la scene VisPy via `_build_scene()`, restaure la slice courante, et force un `canvas.update()` apres changement de topologie.

**Contexte :**
Un crash VisPy/OpenGL (`GL_INVALID_VALUE` pendant `SceneCanvas.on_draw`) apparaissait lorsque la vue Volume etait detachee en fenetre flottante ADS. Le symptome indique un etat GL invalide/stale apres changement de parent et de surface de rendu.

**Decisions techniques :**
1. Traiter le probleme a la source (partage de contextes Qt) et au niveau vue (reconstruction explicite de la scene) pour couvrir les cas de reparenting ADS.
2. Garder la logique de docking dans `DockLayoutController` et la logique GL dans `VolumeView` pour rester conforme a MVC.



### 2026-02-10 - Refactor Endview MVC et session corrosion preservee
**Tags :** `#branch:annotation`, `#controllers/annotation_controller.py`, `#controllers/ascan_controller.py`, `#controllers/dock_layout_controller.py`, `#controllers/endview_controller.py`, `#controllers/master_controller.py`, `#views/endview_view_corrosion.py`, `#mvc`, `#endview`, `#corrosion`, `#sessions`, `#stackedlayout`

**Actions effectuees :**
- Ajout de `EndviewViewCorrosion` heritee de `EndviewView` avec rendu de lignes corrosion cosmetiques et stack Endview standard/corrosion dans `DockLayoutController`.
- Creation de `EndviewController` pour centraliser le mode standard/corrosion, la slice, la crosshair, la colormap, et les interactions point/drag.
- Delegation des responsabilites Endview depuis `MasterController` vers `EndviewController` et adaptation de `AScanController` via callback `set_endview_crosshair`.
- Synchronisation de l overlay annotation vers la vue corrosion dans `AnnotationController` et suppression de methodes redondantes point/drag/cross de ce controller.
- Protection de la session source pendant l analyse corrosion: snapshot pre-analyse, restauration de la session d origine, creation de la session corrosion active sans ecraser la session initiale.

**Contexte :**
Les vues basculaient en mode corrosion dans une nouvelle session mais le retour a la session d origine conservait un etat de vue corrosion. En parallele, la logique Endview etait dispersee entre controllers, ce qui rendait la separation MVC moins claire.

**Decisions techniques :**
1. Introduire un controller dedie Endview pour isoler l orchestration des vues Endview et reduire le couplage avec `MasterController`.
2. Capturer et restaurer explicitement l etat de session pre-corrosion pour garantir qu un retour a la session d origine restaure bien les vues standard.



### 2026-02-10 - Dual endview U V et slice orthogonale synchronisee
**Tags :** `#branch:annotation`, `#controllers/annotation_controller.py`, `#controllers/dock_layout_controller.py`, `#controllers/endview_controller.py`, `#controllers/master_controller.py`, `#models/view_state_model.py`, `#services/annotation_axis_service.py`, `#views/volume_view.py`, `#ucoordinate.ui`, `#ui_ucoordinate.py`, `#ui_vcoordinate.py`, `#vcoordinate.ui`, `#mvc`, `#orthogonal-view`, `#overlay-sync`

**Actions effectuees :**
- Ajout d un layout dock dual U Coordinate et V Coordinate avec nouveaux fichiers UI et alias de compatibilite dans `DockLayoutController`.
- Ajout du service `AnnotationAxisService` pour choix du mode d axe annotation `Auto` `UCoordinate` `VCoordinate`, application du mode par permutation des axes du modele, generation des titres de docks, transposition du volume secondaire et transposition de l overlay secondaire.
- Extension de `ViewStateModel` avec etat de navigation secondaire `secondary_slice`, bornes min max et helpers de clamp et set.
- Extension de `VolumeView` avec slider secondaire, signal `secondary_slice_changed`, et visualisation 3D du plan orthogonal secondaire via une ligne rectangle synchronisee.
- Extension de `EndviewController` pour piloter aussi la vue secondaire sur volume, slice, crosshair, colormap, visibilite du crosshair et resize.
- Mise a jour de `MasterController` pour demander le mode d axe au chargement NDE, initialiser les bornes de slice secondaire, connecter les signaux secondaires, synchroniser slice secondaire depuis interactions endview cscan ascan, et pousser etat secondaire vers les vues.
- Mise a jour de `AnnotationController` pour propager overlay, reset et opacite vers la vue secondaire et y appliquer une version transposee de l overlay.

**Contexte :**
Besoin de supporter deux endviews synchronisees avec une coupe orthogonale secondaire lisible en temps reel, tout en gardant une architecture MVC stricte et une source unique d etat de navigation.

**Decisions techniques :**
1. Garder la vue secondaire en lecture seule pour eviter de dupliquer la logique metier d annotation et limiter les regressions d edition.
2. Appliquer le changement de plan d annotation au chargement via permutation du volume du modele et mise a jour de `axis_order`, afin de conserver un pipeline aval coherent.
3. Centraliser l index de slice secondaire dans `ViewStateModel` et synchroniser toutes les vues a partir de cet etat partage.



### 2026-02-11 - Profil corrosion applique a la vue transversale read-only
**Tags :** `#branch:annotation`, `#controllers/annotation_controller.py`, `#controllers/dock_layout_controller.py`, `#controllers/endview_controller.py`, `#controllers/master_controller.py`, `#mvc`, `#corrosion`, `#endview`, `#read-only`, `#overlay`, `#pyqt`

**Actions effectuees :**
- Ajout d un stack corrosion secondaire dans `DockLayoutController` pour la vue V read-only avec une instance dediee de `EndviewViewCorrosion`.
- Extension de `EndviewController` avec gestion du stack secondaire standard/corrosion et synchronisation secondaire du volume, slice, crosshair, colormap et taille d affichage.
- Extension de `AnnotationController` pour pousser l overlay transpose vers la vue secondaire corrosion et propager clear/opacite sur cette vue.
- Mise a jour de `MasterController` pour injecter les nouvelles references secondaire corrosion, connecter les signaux de slice et brancher les nouveaux parametres des controllers.

**Contexte :**
Apres l analyse corrosion, la vue annotation corrosion affichait un profil interpole propre, mais la vue transversale read-only affichait un rendu masque moins lisible. Le besoin etait d aligner le rendu profil de la vue read-only sur celui de la corrosion annotation sans casser la separation MVC.

**Decisions techniques :**
1. Reutiliser `EndviewViewCorrosion` pour la vue secondaire afin de conserver un rendu de lignes cosmetiques coherent entre les deux endviews.
2. Conserver la transposition d overlay dans `AnnotationAxisService` et diffuser ce meme overlay secondaire vers les vues read-only standard et corrosion.
3. Maintenir la separation MVC en limitant la composition UI a `DockLayoutController` et l orchestration d etat aux controllers (`MasterController`, `EndviewController`, `AnnotationController`).



### 2026-02-11 - Interaction camera VolumeView: ancrage clic droit et recentrage au double clic
**Tags :** `#branch:annotation`, `#views/volume_view.py`, `#views/piece3d_view.py`, `#vispy`, `#camera`, `#anchor`, `#double-click`

**Actions effectuees :**
- Ajout de `_AnchorMoveTurntableCamera` dans `VolumeView` pour remplacer le zoom au clic droit par un deplacement du centre camera (point d ancrage) en XY.
- Correction de compatibilite VisPy avec import explicite `PerspectiveCamera` depuis `vispy.scene.cameras.perspective` pour eviter l exception runtime sur les evenements souris.
- Ajout d un `eventFilter` sur le canvas 3D pour recentrer la camera uniquement au double clic gauche.
- Suppression du recentrage automatique dans `set_slice_index` via un flag `_recenter_on_slice_change` desactive par defaut.
- Preservation du comportement de `Piece3DView` en forcant `_recenter_on_slice_change = True`.

**Contexte :**
Le besoin etait de changer la navigation dans la vue volume: ne plus zoomer au clic droit, deplacer l ancrage manuellement, et eviter le recentrage automatique lors des changements d index. Un crash VisPy est apparu apres le premier patch a cause d un appel de classe camera non resolu dans la version locale.

**Decisions techniques :**
1. Garder la logique d interaction dans la couche View (`VolumeView`) pour respecter MVC et ne pas polluer les controllers.
2. Deriver `TurntableCamera` au lieu de rebind des events externes pour conserver les comportements natifs VisPy (rotation, modificateurs) avec un changement localise.
3. Activer le recentrage automatique uniquement dans `Piece3DView` afin de ne pas regresser les flux corrosion dependants de ce recentrage.



### 2026-02-13 - Couleur du contour ROI box adaptee au colormap endview
**Tags :** `#branch:annotation`, `#views/annotation_view.py`, `#roi`, `#box`, `#colormap`, `#omniscan`, `#gris`, `#pyqt`

**Actions effectuees :**
- Surcharge de `set_colormap` dans `AnnotationView` pour recalculer la couleur des contours ROI a chaque changement de palette.
- Ajout de `_update_roi_outline_color` avec mapping explicite `OmniScan` vers noir et fallback blanc pour `Gris` et les autres palettes.
- Application de la couleur sur les stylos ROI persistants (`_roi_pen`, `_roi_point_pen`) et re-application immediate sur les items deja affiches.
- Synchronisation de la box temporaire de selection (`_temp_box_item`) sur la meme couleur que les ROI persistantes.
- Initialisation de la couleur au demarrage via appel de `_update_roi_outline_color` en fin de constructeur.

**Contexte :**
Demande utilisateur de rendre la lisibilite des ROI box dependante du colormap actif dans l endview: contour blanc en mode Gris et noir en mode OmniScan, y compris pendant la selection en cours.

**Decisions techniques :**
1. Conserver la logique dans la couche View (`AnnotationView`) car il s agit d un comportement purement visuel.
2. Reutiliser l etat existant `_colormap_name` mis a jour par `EndviewView.set_colormap` pour eviter tout couplage controller supplementaire.
3. Appliquer un fallback blanc pour toute palette non OmniScan afin de maintenir une valeur sure par defaut.



### 2026-02-13 - ROI free hand polygonale alignee sur le flux box
**Tags :** `#branch:annotation`, `#controllers/annotation_controller.py`, `#models/roi_model.py`, `#services/annotation_service.py`, `#views/annotation_view.py`, `#roi`, `#free_hand`, `#polygon`, `#threshold`, `#apply_volume`, `#mvc`, `#opencv`

**Actions effectuees :**
- Ajout du mode d interaction `free_hand` dans `AnnotationView` avec capture souris press/move/release, preview temporaire du trace et emission des signaux `freehand_started`, `freehand_point_added`, `freehand_completed`.
- Ajout de `add_free_hand` dans `RoiModel` pour persister les ROI polygonales avec slice, points, label, threshold et persistance.
- Ajout dans `AnnotationService` de `build_free_hand_mask` (remplissage polygonal via `cv2.fillPoly`) avec normalisation et clamp des points.
- Ajout de `apply_free_hand_roi` et `apply_free_hand_roi_to_range` pour appliquer la ROI free hand comme la box (threshold, restriction mask, blocked mask, palette, mode volume).
- Integration des ROI `free_hand` dans `rebuild_temp_masks_for_slice` pour que recompute/apply reconstruisent correctement les previews temporaires.
- Implementation de `on_annotation_freehand_completed` dans `AnnotationController` avec orchestration complete mono-slice ou plage volume selon `apply_volume` et `roi_persistence`.

**Contexte :**
Demande utilisateur d implementer une zone ROI free hand qui se comporte comme la ROI box mais avec une forme polygonale dessinee a la main. Le diff staged montre que le wiring UI existait mais la logique free hand etait encore en stubs dans le controller/service/model.

**Decisions techniques :**
1. Conserver la separation MVC stricte : capture gesture et preview dans la View, stockage metadata dans le Model, calcul/application des masques dans le Service, orchestration dans le Controller.
2. Reutiliser les memes regles fonctionnelles que `box` (threshold, restriction, blocage label, apply_volume, persistance) pour garantir un comportement coherent entre outils ROI.
3. Utiliser un remplissage polygonal robuste (`cv2.fillPoly`) pour produire un masque binaire ferme et compatible avec la pipeline existante de temp mask.



### 2026-02-13 - Couleur free hand ROI alignee sur le colormap endview
**Tags :** `#branch:annotation`, `#views/annotation_view.py`, `#MEMORY.md`, `#roi`, `#free_hand`, `#colormap`, `#omniscan`, `#gris`, `#pyqt`

**Actions effectuees :**
- Extension de `_update_roi_outline_color` pour appliquer la couleur dynamique aussi au stylo du trace temporaire free hand (`_temp_line_item`).
- Conservation de la regle de contraste existante: `OmniScan` en noir, `Gris` (et fallback) en blanc.
- Maintien du comportement dynamique sur changement de colormap via `set_colormap` deja surcharge dans `AnnotationView`.

**Contexte :**
Apres la mise a jour de la ROI box, la demande etait d avoir le meme comportement visuel pour la free hand ROI pendant la selection, afin d uniformiser la lisibilite de tous les contours temporaires selon la palette active.

**Decisions techniques :**
1. Centraliser la mise a jour dans `_update_roi_outline_color` pour eviter une logique dupliquee entre box et free hand.
2. Reutiliser le meme mapping `omniscan -> noir` et fallback blanc pour garantir un rendu coherent sur les outils ROI.
3. Limiter le patch a la couche View (`AnnotationView`) car le changement est purement UI et ne touche ni service ni model.


### 2026-02-17 - Réorganisation du layout ADS en grille 2x2
**Tags :** `#branch:annotation`, `#dock_layout_controller.py`, `#ads`, `#splitter`, `#qtimer`

**Actions effectuees :**
- Ajout de `QTimer.singleShot(0, self._apply_default_splitter_sizes)` pour appliquer les tailles apres l'initialisation de la hierarchie de docks.
- Activation de `EqualSplitOnInsertion` via `_configure_dock_manager()` pour stabiliser le comportement des splits imbriques.
- Recomposition du layout par defaut en grille droite 2x2 : `[V-Coord | Volume]` au-dessus de `[A-Scan | C-Scan]`.
- Separation des constantes de split en trois groupes (`root`, `right_top`, `right_bottom`) et mise a jour de `_apply_default_splitter_sizes()`.

**Contexte :**
Ajuster la disposition initiale ADS pour obtenir une structure visuelle plus previsible, avec des proportions fixes des le demarrage et un decoupage explicite des zones droite haute/basse.

**Decisions techniques :**
1. Utiliser `QTimer.singleShot(0, ...)` pour differer l'application des tailles apres construction complete des `dockAreaWidget`, afin d'eviter des tailles ignorees au premier rendu.
2. Conserver des ratios `1:1` sur les splits internes droite haute/basse pour une symetrie lisible, tout en imposant `20/50/30` sur le split racine pour prioriser la zone centrale.

### 2026-02-20 - Persistance robuste de la disposition des docks ADS
**Tags :** `#branch:annotation`, `#controllers/dock_layout_controller.py`, `#controllers/master_controller.py`, `#ads`, `#qsettings`, `#dock-layout`, `#state-persistence`

**Actions effectuees :**
- Ajout de la persistance du layout ADS via QSettings dans DockLayoutController avec saveState/restoreState versionnes.
- Initialisation du layout au demarrage avec tentative de restauration puis fallback automatique sur les proportions par defaut.
- Connexion de QApplication.aboutToQuit dans MasterController pour sauvegarder systematiquement la disposition a la fermeture.
- Correction du bug de restauration partielle en attribuant un objectName unique et stable a chaque dock ADS.
- Invalidation des anciens snapshots de layout incompatibles (version 2) et purge defensive des etats invalides/corrompus.

**Contexte :**
La demande etait de conserver la disposition des docks entre deux sessions. Un bug est apparu apres deplacement du C-Scan: au redemarrage, seuls Tools et C-Scan restaient visibles. L analyse des changements staged montrait des objectName dupliques (DockWidget) qui rendaient restoreState ambigu.

**Decisions techniques :**
1. Utiliser CDockManager.saveState/restoreState avec QSettings cote controller de layout pour respecter MVC et centraliser la logique UI docking.
2. Utiliser des objectName explicites par dock (dock_tools, dock_ucoordinate, dock_vcoordinate, dock_cscan, dock_ascan, dock_volume) pour garantir une restauration deterministe.
3. Versionner le state (2) afin d ignorer les snapshots precedents incompatibles et repartir proprement sans intervention manuelle.
4. Purger la cle de state si le type persiste est invalide ou si restoreState echoue, pour eviter les redemarrages casses repetes.

### 2026-02-20 - Sync slice secondaire vers croix C-scan
**Tags :** `#branch:annotation`, `#controllers/master_controller.py`, `#secondary-slice`, `#cscan`, `#ascan`, `#endview`, `#crosshair`, `#mvc`

**Actions effectuees :**
- Etendu `_on_secondary_slice_changed` pour recuperer le point courant et calculer un `current_y` de secours (centre vertical) si aucun point n'est actif.
- Ajoute un garde pour conserver le chemin existant (`_sync_secondary_endview_state`) quand `current_x` est deja aligne avec la slice secondaire.
- Branche la synchro de changement X via `_update_ascan_trace(point=(clamped, current_y))` afin de propager la mise a jour a l'Endview principal, au C-scan et a l'A-scan par le pipeline centralise.
- Met a jour le label de position outils avec le point effectivement synchronise (ou fallback local).

**Contexte :**
Le changement de slice sur la 2e Endview ne deplacait pas la ligne verticale de la croix dans le C-scan. Le flux secondaire mettait a jour la vue secondaire et le slider 3D, mais ne passait pas par le chemin de synchro crosshair C-scan.

**Decisions techniques :**
1. Reutiliser `_update_ascan_trace` comme point d'orchestration pour eviter de dupliquer la logique de propagation du crosshair entre vues.
2. Conserver `_sync_secondary_endview_state` pour le cas no-op (`current_x == clamped`) afin de limiter les refresh inutiles.
3. Garder un fallback `current_y` au centre quand aucun point actif n'est disponible, pour garantir une synchro deterministe.

### 2026-02-20 - Edition ancree du profil corrosion dans Endview avec commit Apply ROI
**Tags :** `#branch:annotation`, `#controllers/corrosion_profile_controller.py`, `#controllers/endview_controller.py`, `#controllers/master_controller.py`, `#services/corrosion_profile_edit_service.py`, `#services/cscan_corrosion_service.py`, `#views/endview_view_corrosion.py`, `#corrosion`, `#endview`, `#anchors`, `#apply-roi`, `#mvc`

**Actions effectuees :**
- Ajout de `CorrosionProfileEditService` pour gerer le contexte des peak maps A/B, les points d ancrage, le drag, l ajout d ancrage au double clic, le preview overlay temporaire et le commit avec recalcul de projection.
- Ajout de `CorrosionProfileController` pour orchestrer l edition en mode corrosion (selection label A/B, drag start/move/end, double clic, synchronisation des ancrages, commit des edits sur Apply ROI).
- Extension de `EndviewController` avec une API d integration profile editing (binding des signaux de la vue corrosion, push de l overlay preview, set/clear des anchor points).
- Extension de `EndviewViewCorrosion` avec les signaux `profile_drag_*` et `profile_double_clicked`, un `eventFilter` souris sans modifieurs et le rendu graphique des anchor points.
- Integration dans `MasterController` : instanciation du service/controller, reroutage de `apply_roi_requested` vers un commit conditionnel, reset du service aux transitions de workflow corrosion, synchro des ancrages aux changements de slice et refresh overlay.
- Extension de `CScanCorrosionService` avec `interpolate_peak_map_1d` et des wrappers publics pour rebuild overlay/distance map, puis application de l interpolation des peak maps avant le calcul de projection corrosion.

**Contexte :**
Diff staged du 2026-02-20 sur 6 fichiers (`controllers/corrosion_profile_controller.py`, `controllers/endview_controller.py`, `controllers/master_controller.py`, `services/corrosion_profile_edit_service.py`, `services/cscan_corrosion_service.py`, `views/endview_view_corrosion.py`) avec 1003 insertions et 7 suppressions. Le besoin etait de permettre une edition interactive du profil corrosion (lignes A/B) directement dans l Endview, avec validation explicite via Apply ROI et recalcul coherent de l overlay/projection.

**Decisions techniques :**
1. Garder la capture d interaction et le rendu des ancrages dans la View, l orchestration dans les Controllers, et l edition/recalcul des donnees dans les Services pour respecter le MVC strict.
2. Utiliser un etat `pending edits` dans le service d edition et rediriger Apply ROI vers un commit uniquement en mode corrosion, afin de conserver le comportement ROI existant hors corrosion.
3. Interpoler les trous (`-1`) des peak maps avant commit/projection pour stabiliser les lignes et limiter les artefacts dans l overlay et la carte de distance.
4. Exposer des wrappers publics dans `CScanCorrosionService` plutot que d appeler des methodes privees depuis les composants d edition.


### 2026-02-23 - Mode Mod masque ancre (edition locale + reactivation double-clic)
**Tags :** `#branch:annotation`, `#controllers/mask_modification_controller.py`, `#controllers/master_controller.py`, `#services/mask_modification_service.py`, `#views/annotation_view.py`, `#views/tools_panel.py`, `#toolspanel.ui`, `#ui_toolspanel.py`, `#mvc`, `#pyqt6`, `#opencv`, `#mask-editing`

**Actions effectuees :**
- Ajout du mode outil `mod` dans le panneau d'outils (UI Qt + wrapper `ToolsPanel`) avec emission de `tool_mode_changed("mod")`.
- Integration d'un controleur dedie `MaskModificationController` pour orchestrer selection de composant, drag d'ancrages, preview overlay pending, apply/cancel et synchro UI.
- Integration d'un service metier `MaskModificationService` pour gerer l'etat pending, extraction de contour de composant, ancrages, deformation/drag et commit/cancel des masques.
- Extension de `AnnotationView` avec signaux d'interaction `mod_*`, rendu des points d'ancrage, et capture d'evenements souris en mode `mod`.
- Cablage dans `MasterController` (instanciation, connexions des signaux, reset sur changements d'etat/session/corrosion, apply/cancel via flux existant).
- Ajustements iteratifs du comportement d'ancrage: densite, suppression de doublons, restrictions de drag, puis reactivation du double-clic pour insertion d'ancrage sur segment apres deformation.

**Contexte :**
Objectif: etendre a l'annotation masque le paradigme d'edition ancree du profil corrosion. Les masques etant des polygones complexes, le flux a ete adapte pour permettre une edition locale controlee avec previsualisation avant application globale. Un probleme d'usage apparaissait apres agrandissement du polygone: de nouveaux espaces visuels ne permettaient pas toujours de recreer des ancrages par double-clic.

**Decisions techniques :**
1. Isoler la logique d'edition masque dans un service/controleur dedies pour preserver la separation MVC et eviter de surcharger `AnnotationController`.
2. Conserver un workflow pending non destructif (preview + commit/cancel) pour securiser l'edition interactive avant application finale au volume masque.
3. Baser l'ajout d'ancrage au double-clic sur la projection sur segment de contour (pas seulement sur sommets existants) afin de pouvoir recreer des ancrages dans les zones etirees apres drag.


### 2026-02-23 - Resize multi-vues et Piece3D basee distance avec switch legacy
**Tags :** `#branch:annotation`, `#controllers/master_controller.py`, `#controllers/cscan_controller.py`, `#services/cscan_corrosion_service.py`, `#views/cscan_view.py`, `#views/volume_view.py`, `#views/piece3d_view.py`, `#resize`, `#cscan`, `#volume3d`, `#piece3d`, `#distance-map`, `#prism`, `#vispy`, `#mvc`

**Actions effectuees :**
- Extension du resize Affichage pour appliquer la deformation visuelle aussi a C-scan et a VolumeView, avec reset/apply centralises dans MasterController.
- Ajout dans CScanView et CScanController de get/set/reset display size pour harmoniser le comportement avec Endview.
- Ajout dans VolumeView de la deformation XY basee display size (factors X/Y), propagation dans les transforms VisPy et rescale de camera pour conserver la navigation.
- Remplacement de la geometrie Piece3D par un volume prismatique construit uniquement depuis distance_map et interpolated_distance_map.
- Conservation du pipeline legacy BW/FW en parallele (raw/interpole) et propagation des deux familles de volumes dans les resultats corrosion/workflow.
- Refactor de Piece3DView avec menu contextuel Geometrie pour switcher entre Prisme distance et Volume BW/FW, tout en conservant le toggle brut/interpole.

**Contexte :**
Le rendu Piece3D base sur la peinture BW/FW donnait une impression de shift geometrique liee aux masques traces. L objectif est de representer la piece depuis les distances mesurees (logique inspection), tout en gardant une comparaison directe avec le rendu legacy via un switch de source.

**Decisions techniques :**
1. Definir la geometrie par defaut de Piece3D sur un prisme distance (hauteur derivee de max distance, remplissage colonne par colonne, NaN converti a 0) pour decoupler la forme du trace masque.
2. Garder les volumes legacy BW/FW dans le workflow et exposer un switch contextuel dans Piece3DView pour valider visuellement les ecarts et limiter le risque de regression terrain.
3. Conserver le bouton brut/interpole independamment de la source geometrique et un calcul d ancre tolerant (premier volume non vide) pour stabiliser le recentrage camera.

### 2026-02-23 - Reinitialisation manuelle des docks ADS depuis le menu Affichage
**Tags :** `#branch:annotation`, `#controllers/dock_layout_controller.py`, `#controllers/master_controller.py`, `#ui_mainwindow.py`, `#untitled.ui`, `#ads`, `#dock-layout`, `#qsettings`, `#qt-actions`, `#mvc`

**Actions effectuees :**
- Ajout dans `DockLayoutController` d un snapshot `_default_layout_state` capture apres construction du layout par defaut.
- Ajout de `reset_layout_to_default()` pour purger l etat persiste, restaurer la disposition par defaut, rouvrir les docks fermes et resynchroniser l action du panneau outils.
- Cablage dans `MasterController` de l action menu `actionR_initialisation_docks` vers `dock_layout_controller.reset_layout_to_default`.
- Mise a jour de `ui_mainwindow.py` et `untitled.ui` pour declarer et exposer l action `Reinitialisation docks` dans le menu Affichage.

**Contexte :**
Les changements staged du 2026-02-23 introduisent un moyen explicite de revenir a une disposition docks saine apres des manipulations utilisateur, sans dependre d une fermeture/reouverture de l application ni d un nettoyage manuel de `QSettings`.

**Decisions techniques :**
1. Reutiliser la mecanique ADS `saveState/restoreState` plutot que reconstruire les docks pour garder un reset rapide et coherent avec la persistance existante.
2. Purger d abord l etat persiste (`_clear_saved_layout_state`) pour eviter qu une restauration ulterieure recharge un layout stale.
3. Garder la logique strictement UI/controleur (`DockLayoutController` + `MasterController` + actions Qt) afin de respecter le perimetre MVC sans impact Model.

### 2026-02-24 - Resolution mm centralisee et interpolation dual-axis corrosion
**Tags :** `#branch:annotation`, `#controllers/cscan_controller.py`, `#models/nde_model.py`, `#services/ascan_service.py`, `#services/corrosion_profile_edit_service.py`, `#services/cscan_corrosion_service.py`, `#services/cscan_service.py`, `#services/nde_loader.py`, `#views/ascan_view_corrosion.py`, `#views/corrosion_settings_view.py`, `#resolution-mm`, `#interpolation`, `#corrosion`, `#mvc`

**Actions effectuees :**
- Remplacement des lectures de resolution ultrasound dans `CScanController` et `AScanService` par `NdeModel.get_axis_resolution_mm(...)`, avec fallback d index d axe.
- Centralisation de la logique de calcul de pas axe dans `NdeModel` (metadata `axis_resolutions_mm`, fallback `status_info`, fallback positions), avec helpers prives robustes.
- Suppression de `CScanService.compute_ultrasound_resolution_mm` et retrait de l import `NdeModel` devenu inutile dans ce service.
- Ajout dans `NdeLoader` d un pre-calcul `axis_resolutions_mm` en mm/px lors du chargement, avec fallback sur derivee des positions.
- Passage du commit de profil corrosion et du pipeline corrosion vers `interpolate_peak_map_1d_dual_axis`, puis reconstruction explicite de `distance_map` apres interpolation.
- Ajout de `_build_distance_map_from_peak_maps` dans `CScanCorrosionService` pour recalculer une carte distance cohérente depuis les peak maps A/B.
- Harmonisation de l extraction des resolutions dans `CorrosionWorkflowService` vers `get_axis_resolution_mm` avec defaults explicites.
- Simplification de l etiquette de mesure dans `AScanViewCorrosion` (priorite mm, puis positions, puis px) et nettoyage formatage import dans `CorrosionSettingsView`.

**Contexte :**
Les changements staged du 2026-02-24 couvrent 9 fichiers modifies autour du workflow corrosion et de la mesure A-scan. L objectif est d eviter les divergences de conversion px/mm entre composants, fiabiliser la resolution des axes a partir des metadonnees loader, et reduire les artefacts de trous sur les cartes de pics via une interpolation sur deux axes.

**Decisions techniques :**
1. Placer la source unique de resolution axe dans le Model (`NdeModel`) pour respecter MVC et eviter la duplication service/controller.
2. Materialiser `axis_resolutions_mm` des le loader pour minimiser les heuristiques tardives et garder un fallback robuste quand les metadonnees sont partielles.
3. Appliquer une interpolation dual-axis (axe principal puis transpose) avant calcul distance/projection afin d ameliorer la continuite spatiale des cartes corrosion.
4. Recalculer la `distance_map` a partir des peak maps interpolees plutot que reutiliser la carte pre-interpolation pour garantir la coherence des resultats affiches.



### 2026-02-24 - Chargement overlay NPZ tolerant au changement d axe U/V
**Tags :** `#branch:annotation`, `#services/overlay_loader.py`, `#overlay`, `#npz`, `#ucoord`, `#vcoord`, `#transpose`, `#mvc` 

**Actions effectuees :**
- Refactor du chargement overlay pour deleguer l alignement de shape a `_align_to_target_shape(...)`.
- Conservation du comportement existant pour les overlays deja alignes `(Z,H,W)` et pour le swap legacy H/W via transpose `(0,2,1)`.
- Ajout d une tolerance supplementaire U/V via transpose `(2,1,0)` pour permettre l ouverture d un overlay exporte en VCoordinate depuis une session UCoordinate (et inversement).
- Ajout d un logging explicite de la permutation appliquee et d un message d erreur enrichi listant les permutations testees.
- Validation locale avec un test inline sur trois cas (`identity`, `swap_hw`, `swap_uv`) confirme en `ALL_OK`.

**Contexte :**
Le chargement NPZ echouait quand l orientation d export overlay etait differente entre UCoordinate et VCoordinate, car seul un swap H/W etait tolere. Les changements staged du 2026-02-24 ciblent ce point de robustesse sans changer l architecture MVC existante.

**Decisions techniques :**
1. Garder la logique d alignement strictement dans `OverlayLoader` (service) pour conserver un point unique de verite au chargement.
2. Appliquer des permutations deterministes et bornees (`(0,2,1)` puis `(2,1,0)`) afin d eviter des heuristiques ambigues.
3. Echouer explicitement quand aucune permutation ne matche, avec un message actionnable (shape source/cible + permutations tentees).

### 2026-02-24 - Mode ROI Peak freehand avec choix 1er/2e pic et germination verticale
**Tags :** `#branch:annotation`, `#controllers/annotation_controller.py`, `#controllers/master_controller.py`, `#models/roi_model.py`, `#models/view_state_model.py`, `#services/annotation_service.py`, `#views/annotation_view.py`, `#views/nde_settings_view.py`, `#views/tools_panel.py`, `#toolspanel.ui`, `#ui_toolspanel.py`, `#roi`, `#peak`, `#ascan`, `#threshold`, `#settings`, `#ui`, `#mvc`

**Actions effectuees :**
- Activation du nouveau mode outil `Peak` via `radioButton_7` et propagation du `tool_mode` dans le panel outils, la vue annotation et les controleurs.
- Ajout dans `Fichier > Parametres` d un selecteur `Premier pic`/`Deuxieme pic`, relie au `ViewStateModel` par signal dedie.
- Extension du `ViewStateModel` avec `roi_peak_prefer_second` pour centraliser la preference de pic.
- Ajout du type ROI `peak` dans `RoiModel` pour conserver et reconstruire ce mode comme les autres ROIs.
- Integration dans `AnnotationService` de `build_ascan_max_mask` (meme logique 1er/2e pic de la branche ascan-mode), `apply_peak_roi`, `apply_peak_roi_to_range` et support de rebuild slice/volume.
- Implementation du pipeline mode Peak: zone freehand -> un pic A-scan par colonne -> germination verticale uniquement dans la ROI.
- Correction du comportement threshold: la detection des pics est independante du threshold; le threshold controle seulement la germination verticale, en echelle normalisee 0..255.
- Validation locale: compilation Python des fichiers modifies et smoke tests runtime sur le service de peak ROI.

**Contexte :**
Le besoin etait d ajouter un mode ROI `Peak` combinant la selection freehand, la logique A-scan 1er/2e pic et une expansion type grow uniquement verticale. Un probleme utilisateur a ensuite montre que rien n etait selectionne sauf a threshold 0; la cause etait une comparaison du threshold slider avec des valeurs de slice non normalisees.

**Decisions techniques :**
1. Garder toute la logique de selection peak/germination dans `AnnotationService` pour respecter MVC et eviter du metier dans View/Controller.
2. Piloter la preference 1er/2e pic depuis `NdeSettingsView` vers `ViewStateModel` pour un comportement coherent entre dessin local, recompute et apply volume.
3. Utiliser le threshold uniquement pour la pousse verticale (masque normalise 0..255) et conserver le seed peak meme si la pousse est nulle, afin d aligner le comportement avec l attente operateur.





### 2026-02-25 - Force carre VolumeView en cube avec persistance inter-session
**Tags :** `#branch:annotation`, `#controllers/master_controller.py`, `#views/endview_resize_dialog.py`, `#views/volume_view.py`, `#resize`, `#force_carre`, `#volume_view`, `#vispy`, `#3d`, `#cube`, `#session`, `#mvc`

**Actions effectuees :**
- Propagation du flag `force_carre` depuis le dialog resize vers `MasterController`, puis vers `VolumeView.set_display_size(..., force_square=...)`.
- Extension de `EndviewResizeDialog` avec `is_square_locked()` et synchronisation bidirectionnelle largeur/hauteur sous option carree via un guard anti-boucle.
- Ajout dans `VolumeView` d un etat `force_square` et d un facteur de profondeur (`_display_depth_scale`) pour contraindre les dimensions affichees en cube (X=Y=Z).
- Application du mode force carre sur la camera 3D (vue orthographique `fov=0`, orientation verrouillee, recentrage et ranges 3D coerents).
- Application du scale Z dans les transforms 3D (volume principal, overlay volume, plans/lignes, image de slice et overlay de slice).
- Ajustement du rescale camera lors des changements de display size pour prendre en compte le ratio Z en plus de X/Y.
- Persistance du mode cube lors des rebuilds de scene (`set_volume`, changement de session) en reappliquant le preset force carre si actif.

**Contexte :**
Le mode force carre corrigeait bien la deformation XY mais ne garantissait pas un vrai cube 3D et pouvait se perdre lors d un changement de session, car la scene volume etait reconstruite sans reappliquer la contrainte force carre.

**Decisions techniques :**
1. Faire transiter explicitement l intention utilisateur `force carre` du dialog jusqu a `VolumeView` plutot que deduire cet etat implicitement.
2. Forcer un cube via un scale profondeur derive de la taille affichee XY (`side/depth`) pour aligner le rendu 3D avec l attente operateur.
3. Utiliser une projection orthographique uniquement en mode force carre, et restaurer la projection camera par defaut hors de ce mode.
4. Reappliquer le preset force carre apres reconstruction de scene pour garantir un comportement stable au changement de session.


### 2026-02-26 - Mode mod aligne sur pipeline temp standard et source slice unique
**Tags :** `#branch:annotation`, `#controllers/mask_modification_controller.py`, `#controllers/master_controller.py`, `#models/temp_mask_model.py`, `#services/mask_modification_service.py`, `#views/annotation_view.py`, `#mod`, `#roi`, `#temp-mask`, `#overlay`, `#mvc`

**Actions effectuees :**
- Injection de `TempMaskModel` dans `MaskModificationController` et remplacement du rafraichissement overlay global par un refresh ROI cible par slice.
- Stockage du preview `mod` dans le pipeline temporaire standard via `TempMaskModel.set_slice_data(...)`, avec suivi `_mod_preview_base`, `_mod_preview_current` et `_mod_preview_slices`.
- Ajout d un rebase du baseline preview `mod` pour conserver les modifications externes du temp mask quand d autres outils ecrivent sur la meme slice.
- Suppression du clear automatique de la zone `mod` au changement de mode (controller + view), pour garder la zone tant qu il n y a pas apply/delete explicite.
- Changement du raccourci `W` vers `_apply_roi_non_corrosion` pour appliquer via le flux standard (`commit_pending_edits` puis `on_apply_temp_mask_requested`).
- Centralisation de la suppression ROI dans `MasterController._on_roi_delete_requested` afin de nettoyer aussi l etat pending `mod` de facon coherente.
- Refactor `MaskModificationService` vers un mode slice-first: retrait des volumes pending/base internes, `start_drag`/`add_anchor_on_contour` prennent `slice_mask`, `drag_to` retourne la slice mise a jour, `commit` retourne les slices dirty.
- Ajout de `TempMaskModel.set_slice_data(...)` pour remplacer explicitement masque + coverage d une slice temporaire.

**Contexte :**
L objectif etait d aligner le mode `mod` sur le comportement des autres outils ROI: conserver un temporaire visible en sortie de mode, appliquer/supprimer via le pipeline standard, et eviter les refresh overlay globaux couteux lors des bascules d outil.

**Decisions techniques :**
1. Utiliser `TempMaskModel` comme source unique de verite du temporaire visible, y compris pour `mod`, afin d uniformiser apply/delete/rebuild.
2. Garder la logique metier `mod` dans le service mais la limiter a la slice active pour reduire les copies de volume et les risques de divergence.
3. Faire passer les actions utilisateur (`W`, delete ROI) par les memes points d entree `MasterController` pour garantir un comportement homogene entre outils.
4. Preserver la zone `mod` hors du mode `mod` et ne la nettoyer que sur action explicite (apply/cancel/delete), comme attendu cote ROI.

### 2026-02-27 - Optimisation peak ROI et commit corrosion avec recalcul projection
**Tags :** `#branch:annotation`, `#controllers/annotation_controller.py`, `#services/annotation_service.py`, `#services/corrosion_profile_edit_service.py`, `#services/cscan_corrosion_service.py`, `#peak`, `#corrosion`, `#performance`, `#projection`, `#mvc`

**Actions effectuees :**
- Simplification de `on_apply_temp_mask_requested` en mode apply-volume pour appliquer directement `TempMaskModel` sans rebuild ROI global, avec garde `has_pending`.
- Vectorisation du coeur peak dans `build_ascan_max_mask` via tri global `(x,y)` et regroupement par colonnes, en conservant les regles de choix 1er/2e pic.
- Factorisation du pipeline peak dans `_compute_peak_roi_mask` et ajout de `precomputed_poly_mask` pour reutiliser le polygone lors des applications sur plage de slices.
- Optimisation du commit corrosion en conservant le recalcul projection C-scan, mais en calculant la distance map directement depuis les peak maps (`build_distance_map_from_peak_maps`) plutot que via `overlay -> build_interpolated_distance_map`.
- Ajout de `CScanCorrosionService.build_distance_map_from_peak_maps` comme wrapper public pour reutiliser la logique metier existante et eviter les chemins de calcul redondants.

**Contexte :**
Le mode peak et le commit des modifications de profil corrosion etaient perçus comme lents. Le besoin etait de maintenir le recalcul C-scan apres edition corrosion tout en supprimant les etapes de recalcul inutiles ou redondantes.

**Decisions techniques :**
1. Conserver le recalcul de projection corrosion au apply, mais baser la distance map sur les peak maps interpolees pour reduire le cout CPU.
2. Garder la parite fonctionnelle du mode peak (tie-break, `prefer_second_peak`) tout en limitant les boucles Python par colonne.
3. Eviter la reconstruction du polygone peak sur chaque slice en apply-range pour reduire le temps cumule en volume.
4. En mode apply-volume annotation, appliquer strictement ce qui est deja present dans `TempMaskModel` afin d eliminer le rebuild ROI global au moment du apply.

### 2026-02-27 - Parametres Peak sans position et germination verticale min/max
**Tags :** `#branch:annotation`, `#controllers/annotation_controller.py`, `#controllers/master_controller.py`, `#models/view_state_model.py`, `#services/annotation_service.py`, `#views/nde_settings_view.py`, `#peak`, `#roi`, `#mvc`

**Actions effectuees :**
- Ajoute un mode de selection Peak Plus fort sans position dans les Parametres ROI Peak.
- Ajoute deux parametres de germination verticale Peak : minimum et maximum (0 = illimite).
- Propage ces parametres du ViewStateModel vers MasterController et AnnotationController pour tous les flux Peak (slice courante, range volume, recompute).
- Etend AnnotationService pour supporter ignore_peak_position dans la selection du pic par colonne.
- Etend _build_vertical_peak_growth_mask pour limiter la longueur verticale par colonne (max) et supprimer les segments trop courts (min).
- Verifie la validite des changements par compilation Python ciblee sur les fichiers modifies.

**Contexte :**
L'objectif etait d'ameliorer le mode Peak ROI: d'abord permettre de prendre le pic le plus fort sans contrainte de position, puis controler la germination verticale avec des bornes min/max pour reduire le bruit (pixels isoles) et caper la croissance. Une demande d'application globale des ROI a aussi ete clarifiee: le raccourci Enter applique deja a tout le volume.

**Decisions techniques :**
1. Conserver une separation MVC stricte : View (widgets/signaux), Model (etat), Controller (orchestration), Service (logique Peak).
2. Garder la retro-compatibilite via des valeurs par defaut (min=1, max=0) qui preservent le comportement historique.
3. Appliquer max sur la longueur verticale totale du segment par colonne en restant centre autour du seed quand possible.
4. Appliquer min comme filtre final de segment vertical: tout segment plus court est ignore.
5. Ne pas changer le workflow d'application globale par bouton: Enter reste l'action globale pour eviter un cout de rafraichissement overlay inutile sur les edits slice uniques.

### 2026-03-10 - Chargement NDE avec detection de transformations et options Hilbert lissage
**Tags :** `#branch:annotation`, `#controllers/master_controller.py`, `#models/nde_model.py`, `#services/annotation_axis_service.py`, `#services/ascan_service.py`, `#services/nde_loader.py`, `#services/nde_signal_processing_service.py`, `#views/nde_open_options_dialog.py`, `#nde`, `#ascan`, `#hilbert`, `#smoothing`, `#rectification`, `#mvc`, `#legacy`

**Actions effectuees :**
- Ajoute `NdeSignalProcessingService` pour lire `rectification`, `digitalBandPassFilter`, `signalSource`, `smoothingFilter` et `averagingFactor` depuis les metadonnees NDE, avec fallback legacy `groups[].paut` pour les fichiers 3.x.
- Implemente deux traitements generaux par defaut sur l axe ultrasound : enveloppe de Hilbert et lissage 1D, puis prepare des fonctions vides pour les futurs algorithmes specifiques par type de rectification et de filtre numerique.
- Etend `NdeModel` pour conserver le volume source, un volume traite optionnel, leurs variantes normalisees et le choix de signal actif sans ecraser l acquisition d origine.
- Adapte `AnnotationAxisService` et `AScanService` pour suivre correctement le volume actif lors des transpositions U/V et de l affichage des profils A-scan.
- Enrichit `NdeLoader` avec `group_id`, `path`, `signal_transform_info` et l etat de traitement choisi, puis branche `MasterController` sur un nouveau dialogue d ouverture `NdeOpenOptionsDialog` qui combine le choix `Auto`/`UCoordinate`/`VCoordinate` et l application optionnelle de Hilbert/lissage.
- Preselectionne Hilbert plus lissage general quand `rectification = None` et `digitalBandPassFilter.filterType = None`, puis verifie le comportement sur un NDE reel 3.3.0 corrosion dont les champs sont stockes sous `groups[].paut`, ainsi que par compilation Python ciblee des fichiers modifies.

**Contexte :**
Le besoin etait de detecter au chargement si un fichier NDE contient des A-scans deja transformes ou un signal RF brut, puis de permettre a l ouverture de conserver les donnees telles qu enregistrees ou d appliquer un pipeline generique Hilbert plus lissage. Un fichier reel de corrosion en version 3.3.0 a montre que certaines metadonnees attendues n etaient pas sous `groups[].processes` mais sous `groups[].paut`, ce qui expliquait les valeurs `Unknown` dans le dialogue initial.

**Decisions techniques :**
1. Centraliser la detection des transformations et le pipeline signal dans un service dedie afin de garder le loader, le modele et le controleur simples et conformes au MVC.
2. Preserver toujours le volume source et exposer un volume traite separe dans `NdeModel` plutot que de muter les donnees chargees.
3. Regrouper le choix du plan d annotation et le choix de traitement du signal dans un unique dialogue d ouverture pour limiter les allers-retours UI.
4. Considerer `rectification = None` et `digitalBandPassFilter.filterType = None` comme le signal brut de reference, avec Hilbert plus lissage coches par defaut mais desactivables par l utilisateur.
5. Supporter explicitement le schema legacy 3.x via `groups[].paut` pour lire les metadonnees reelles du fichier au lieu d inferer un etat a partir de `processes` vide.

### 2026-03-13 - Split endviews avec signal actif et suffixe de traitement
**Tags :** `#branch:annotation`, `#controllers/master_controller.py`, `#services/endview_export.py`, `#services/split_service.py`, `#nde`, `#endview`, `#split`, `#hilbert`, `#smoothing`, `#mvc`

**Actions effectuees :**
- Recupere dans `MasterController` la selection courante de traitement du signal stockee dans les metadonnees NDE et la convertit en `NdeSignalProcessingOptions` avant le lancement du split flaw/noflaw.
- Etend `SplitFlawNoflawService.split_endviews` pour accepter des options de traitement, detecter si le signal actif doit etre utilise et suffixer le dossier de sortie avec les traitements appliques (`_hilbert`, `_lissage`, `_hilbert+lissage`).
- Fait propager `use_active_signal` vers `EndviewExportService` pour exporter les endviews a partir du volume actif traite au lieu du volume source quand un traitement est demande.
- Fait utiliser les bornes min/max du signal actif pour la normalisation d export, avec fallback sur les metadonnees ou les min/max calcules sur le volume si necessaire.
- Ajoute le type de signal utilise au resume final et aux logs d export pour rendre le workflow de split deterministe et tracable.

**Contexte :**
Le besoin etait d aligner l export et le split des endviews avec le choix de traitement du signal fait a l ouverture du NDE. Sans cette propagation, un utilisateur pouvait visualiser un volume traite (Hilbert, lissage) mais exporter et separer les endviews a partir du volume source, ce qui cassait la coherence entre affichage, noms de dossiers et donnees produites.

**Decisions techniques :**
1. Lire la selection de traitement depuis `nde_model.metadata["signal_processing_selection"]` dans le controleur pour conserver l orchestration au niveau Controller et eviter de faire dependre directement les services de details UI.
2. Reutiliser le concept de volume actif deja expose par `NdeModel` via `get_active_raw_volume()` et `get_active_min_max()` au lieu de recalculer un pipeline de traitement pendant l export.
3. Encoder le traitement applique dans le nom du dossier de sortie afin de separer clairement les exports issus du signal source et ceux issus du signal traite.
4. Garder un fallback robuste sur les min/max metadata puis sur les bornes calculees du volume pour eviter les regressions sur des fichiers NDE incomplets ou anciens.

### 2026-03-14 - Preview des masks temporaires sur les deux endviews
**Tags :** `#branch:annotation`, `#controllers/annotation_controller.py`, `#controllers/master_controller.py`, `#services/annotation_axis_service.py`, `#annotation`, `#endview`, `#tempmask`, `#overlay`, `#orthogonal`, `#mvc`

**Actions effectuees :**
- Ajoute dans `AnnotationAxisService` un helper `build_temp_preview_slice()` pour normaliser la construction du mask de preview temporaire a partir de `slice_mask`, `coverage` et de la palette des labels.
- Ajoute dans `AnnotationAxisService` un helper `build_secondary_temp_preview_slice()` pour extraire et transposer la preview temporaire orthogonale correspondant a la `secondary_slice`.
- Refactorise `AnnotationController.refresh_roi_overlay_for_slice()` et `_rebuild_slice_preview()` pour reutiliser le helper metier au lieu de dupliquer la logique locale de composition du mask temporaire.
- Ajoute `AnnotationController.refresh_secondary_roi_overlay()` pour pousser ou nettoyer la preview temporaire sur la vue secondaire en lecture seule, sans y projeter les ROI boxes ni les seeds.
- Branche `MasterController._on_secondary_slice_changed()` pour recalculer la preview orthogonale lorsque la tranche secondaire change, puis valide la syntaxe par `python -m py_compile` sur les trois fichiers modifies.

**Contexte :**
Le mask temporaire avant application etait visible uniquement dans l endview principal utilise pour annoter. Avec le dual endview U/V deja en place, le besoin etait de rendre la preview temporaire lisible aussi dans la vue orthogonale secondaire afin de suivre la propagation spatiale du masque sans dupliquer la logique d interaction ni casser le statut read-only de cette vue.

**Decisions techniques :**
1. Centraliser la construction des previews temporaires dans `AnnotationAxisService` pour conserver `AnnotationController` comme orchestration et eviter de dupliquer la logique de composition du mask.
2. Garder la vue secondaire en lecture seule et n y afficher que le mask temp transpose, sans ROI boxes ni seeds, afin de limiter le scope et les regressions d interaction.
3. Rafraichir explicitement la preview secondaire lors des changements de `secondary_slice` en plus des mises a jour du temp mask pour garder la coherence entre navigation orthogonale et overlay temporaire.

### 2026-03-14 - Labels persistants nommes et Reflector 100
**Tags :** `#branch:annotation`, `#config/constants.py`, `#models/annotation_model.py`, `#models/temp_mask_model.py`, `#models/view_state_model.py`, `#services/annotation_session_manager.py`, `#controllers/annotation_controller.py`, `#controllers/master_controller.py`, `#controllers/corrosion_profile_controller.py`, `#views/overlay_settings_view.py`, `#views/tools_panel.py`, `#views/nde_settings_view.py`, `#views/corrosion_settings_view.py`, `#labels`, `#overlay`, `#ui`, `#npz`, `#mvc`

**Actions effectuees :**
- Etend les labels persistants a `0`, `1`, `2`, `3` et `100`, avec re-injection automatique dans `AnnotationModel`, `TempMaskModel`, la restauration de session et les flux corrosion.
- Ajoute dans `constants.py` une nomenclature d affichage des labels (`Erase`, `Paint`, `Frontwall`, `Backwall`, `Reflector`) et un helper `format_label_text()` pour dissocier nom UI et id de classe reel.
- Met a jour `OverlaySettingsView`, `ToolsPanel`, `NdeSettingsView` et `CorrosionSettingsView` pour afficher les alias utilisateur sans modifier les ids ecrits dans l overlay ou le NPZ.
- Ajoute le label persistant `100` nomme `Reflector` avec une couleur par defaut dediee dans les palettes BGR/RGB/BGRA.
- Corrige l allocation de nouveaux labels pour repartir a `4` et ignorer le label persistant `100`, afin d eviter un saut artificiel vers `101`.
- Bloque la suppression des labels persistants et verifie la syntaxe par `python -m py_compile` sur les fichiers modifies.

**Contexte :**
Le besoin etait d ameliorer l experience utilisateur en nommant clairement certains labels metier (`Erase`, `Paint`, `Frontwall`, `Backwall`, `Reflector`) tout en conservant strictement les valeurs de classes numeriques appliquees au masque et exportees dans le NPZ. Un nouveau label persistant `100` devait aussi etre expose pour les reflecteurs, sans perturber le workflow d ajout des labels libres qui doit continuer a commencer a `4`.

**Decisions techniques :**
1. Separer completement le texte affiche a l utilisateur de l id de classe reel pour garantir que les exports NPZ et les traitements restent inchanges.
2. Ajouter `100` a la liste des labels persistants et le proteger comme les labels metier `0/1/2/3`, plutot que de le traiter comme un label libre special.
3. Faire calculer le prochain label libre a partir de `4` en ignorant les ids reserves, afin de garder un workflow d ajout stable meme si `100` est toujours present.

### 2026-03-16 - Undo/redo des annotations appliquees
**Tags :** `#branch:annotation`, `#controllers/annotation_controller.py`, `#controllers/master_controller.py`, `#controllers/corrosion_profile_controller.py`, `#models/applied_annotation_history_model.py`, `#annotation`, `#undo`, `#redo`, `#shortcuts`, `#mvc`

**Actions effectuees :**
- Ajoute `AppliedAnnotationHistoryModel` avec une pile `undo` et une pile `redo`, en stockant des copies defensives des slices avant et apres chaque `Apply`.
- Etend `AnnotationController.on_apply_temp_mask_requested()` pour capturer les slices modifiees en mode slice et apply-volume, puis ajoute les handlers `on_undo_last_applied_annotation_requested()` et `on_redo_last_applied_annotation_requested()`.
- Ajoute dans `MasterController` les raccourcis globaux `Ctrl+Z` et `Ctrl+Shift+Z` avec messages de status dedies pour annuler ou reappliquer la derniere annotation deja committee.
- Invalide l historique des annotations appliquees lors des flux qui remplacent le masque permanent ou le rendent incoherent (`reset_overlay_state`, suppression destructive de label, switch de session, commit corrosion).
- Verifie la syntaxe via `python -m py_compile` sur les fichiers modifies.

**Contexte :**
Le besoin etait d annuler la derniere annotation seulement apres son application au masque permanent, sans interagir avec les masks temporaires qui disposent deja de leur propre bouton d annulation dans l UI. La demande a ensuite ete etendue a un redo clavier pour reappliquer l annotation annulee avec `Ctrl+Shift+Z`.

**Decisions techniques :**
1. Conserver l historique au niveau des slices effectivement modifiees plutot qu au niveau d un snapshot complet de session pour limiter la memoire et rester cible sur le masque applique.
2. Stocker pour chaque action les etats avant et apres application afin de rendre `redo` strictement symetrique a `undo` sans recalcul metier supplementaire.
3. Vider la pile `redo` a chaque nouvel `Apply` et invalider tout l historique lorsque le masque permanent est remplace par un autre workflow, afin d eviter de reappliquer un etat obsolete.

### 2026-03-16 - Labels libres renommes en BW echo avec suffixe id
**Tags :** `#branch:annotation`, `#config/constants.py`, `#labels`, `#ui`

**Actions effectuees :**
- Modifie `format_label_text()` pour afficher les labels libres a partir de `USER_LABEL_START` sous la forme `BW echo N (id)` au lieu de `Label id`.
- Conserve le suffixe numerique entre parentheses pour garder la correspondance visuelle avec l id de classe reel.
- Laisse inchanges les ids internes des labels et les alias des labels persistants deja definis (`Erase`, `Paint`, `Frontwall`, `Backwall`, `Reflector`).

**Contexte :**
Le besoin etait de renommer les prochains labels ajoutes en `BW echo 1`, `BW echo 2`, etc., tout en conservant les ids numeriques reels utilises dans les masques, l overlay et les exports. L utilisateur a explicitement demande de garder le suffixe avec l id.

**Decisions techniques :**
1. Centraliser le renommage uniquement dans `format_label_text()` pour propager le nouveau libelle a toutes les vues deja branchees sur ce helper sans toucher au flux MVC d ajout de labels.
2. Calculer l index utilisateur a partir de `USER_LABEL_START` afin de garder `4 -> BW echo 1 (4)`, `5 -> BW echo 2 (5)`, etc.
3. Preserver l id de classe dans le texte affiche pour faciliter le lien entre UI, masque interne et workflows existants.

### 2026-03-18 - ToolsPanel sur doubles coordonnees et toggles d affichage menus
**Tags :** `#branch:annotation`, `#controllers/dock_layout_controller.py`, `#controllers/master_controller.py`, `#views/tools_panel.py`, `#toolspanel.ui`, `#ui_toolspanel.py`, `#ui_mainwindow.py`, `#untitled.ui`, `#mvc`, `#ui`, `#signals-and-slots`, `#dock-layout`

**Actions effectuees :**
- Remplace dans le `ToolsPanel` le flux de navigation mono-slice par deux controles dedies `U-Coordinate` et `V-Coordinate`, chacun avec slider + spinbox synchronises, exposes via `slice_changed` et `secondary_slice_changed`.
- Supprime le bouton `goto`, les boutons `previous/next` et les radios de type ROI cote panneau, puis remplace la selection d outil par un `QComboBox` mappe proprement vers les `tool_mode` metier (`free_hand`, `box`, `grow`, `line`, `paint`, `mod`, `peak`).
- Rend les checkboxes `overlay` et `cross` optionnelles dans `ToolsPanel` pour supporter leur deplacement vers le menu `Affichage`, tout en conservant une API de synchro sans reemission.
- Etend `MasterController` pour brancher le nouveau contrat de `ToolsPanel`, synchroniser les deux endviews avec les bornes/valeurs des spinboxes, pousser les labels d axes U/V, et reutiliser les etats `tool_mode`, `overlay` et `cross` apres chargement ou switch de session.
- Generalise `DockLayoutController` avec `bind_dock_toggle_action()` afin de reutiliser le pattern de `Toggle tools panel` pour `ucoord`, `vcoord`, `A-Scan`, `C-Scan` et `Volume`, avec synchronisation bidirectionnelle entre actions menu et etat ADS reel.
- Ajoute dans le menu `Affichage` les toggles `cross` et `overlay` comme sources d etat UI, synchronises avec le modele de vue et le panneau d outils.
- Verifie la syntaxe via `python -m py_compile` sur `controllers/master_controller.py`, `controllers/dock_layout_controller.py` et `views/tools_panel.py`.

**Contexte :**
Le panneau d outils ne correspondait plus au Designer apres remplacement du `goto` par deux spinboxes de coordonnees, suppression des boutons `previous/next`, et remplacement des radios ROI par un combo box. En parallele, l utilisateur a ajoute dans le menu `Affichage` des toggles par vue et voulait le meme comportement robuste que `Toggle tools panel`, sans melanger logique metier et vue.

**Decisions techniques :**
1. Garder `ToolsPanel` strictement vue en lui faisant seulement exposer les nouveaux signaux UI et des setters de synchronisation, toute l orchestration restant dans `MasterController`.
2. Centraliser la logique de binding `action <-> dock ADS` dans `DockLayoutController` plutot que de dupliquer un handler par dock, pour garantir une synchro menu/etat uniforme.
3. Considerer `cross` et `overlay` comme etats du `ViewStateModel` synchronises a la fois vers le menu et vers le panneau, afin d eviter les divergences quand l utilisateur agit depuis plusieurs points d entree UI.

### 2026-03-18 - Defaut corrosion sur Frontwall et Backwall
**Tags :** `#branch:annotation`, `#services/corrosion_label_service.py`, `#corrosion`, `#labels`, `#defaults`, `#mvc`

**Actions effectuees :**
- Ajoute dans `CorrosionLabelService` des constantes internes pour preferer `Frontwall (2)` et `Backwall (3)` comme paire par defaut.
- Modifie `normalize_pair()` pour retourner explicitement cette paire quand aucun choix utilisateur valide n est encore memorise et que les deux labels sont disponibles.
- Conserve le fallback generique existant sur les deux premiers labels disponibles si `Frontwall` ou `Backwall` manque, afin de ne pas casser les jeux de labels non standards.

**Contexte :**
L analyse corrosion selectionnait par defaut les deux premiers labels disponibles, ce qui favorisait souvent `Paint` puis `Frontwall` a cause de l ordre de palette. Le besoin etait de demarrer naturellement entre `Frontwall` et `Backwall`, qui representent le cas d usage principal pour la mesure corrosion.

**Decisions techniques :**
1. Garder la regle de choix par defaut dans `CorrosionLabelService` plutot que dans le controleur pour conserver un service pur et reutilisable.
2. Preferer explicitement les ids metier `2/3` seulement au moment ou aucune selection valide n existe, afin de ne pas ecraser un choix manuel deja present.
3. Preserver un fallback generique sur les labels disponibles pour rester compatible avec des annotations corrosion construites sur d autres paires de classes.

### 2026-03-18 - ToolsPanel colormap opacite et scroll global
**Tags :** `#branch:annotation`, `#controllers/master_controller.py`, `#views/tools_panel.py`, `#views/nde_settings_view.py`, `#toolspanel.ui`, `#ui_toolspanel.py`, `#ui`, `#opacity`, `#colormap`, `#scrollarea`, `#mvc`

**Actions effectuees :**
- Branche dans `ToolsPanel` le nouveau combo `Colormap` sur la colormap `Endview + 3D`, avec normalisation `gray/gris -> Gris` et `omniscan -> OmniScan`.
- Branche le slider et la spinbox d opacite overlay du panneau sur le pipeline deja existant `ViewStateModel -> AnnotationController -> Endview/Volume`, avec synchro bidirectionnelle entre `ToolsPanel`, `OverlaySettingsView` et `NdeSettingsView`.
- Laisse les controles d opacite NDE visibles mais desactives tant qu aucun backend d opacite de l image NDE n existe dans les vues et le modele.
- Reorganise `toolspanel.ui` autour d une `QScrollArea` principale pour faire defiler tout le panneau avant toute compression, ajoute des contraintes/minimum sizes sur les layouts et regenere `ui_toolspanel.py`.

**Contexte :**
L utilisateur avait ajoute dans le Designer un combo colormap et deux couples slider/spinbox pour les opacites overlay/NDE, puis constatait que le panneau d outils ecrasait les widgets quand le dock devenait trop petit. Le besoin etait de rebrancher les nouveaux controles sur les reglages existants sans dupliquer la logique metier, puis de rendre le panneau scrollable plutot que compressible.

**Decisions techniques :**
1. Reutiliser `MasterController` comme point d orchestration unique pour la synchro colormap/opacite entre surfaces UI, afin de garder `ToolsPanel` strictement vue et de ne pas dupliquer les handlers existants d overlay.
2. Normaliser les noms de colormap au niveau du controleur pour absorber les libelles Designer (`gray`, `omniscan`) sans imposer un format unique aux widgets.
3. Preferer une `QScrollArea` globale sur le `ToolsPanel` avec layouts en `SetMinimumSize` plutot que de laisser les frames enfants se faire ecraser verticalement quand le dock est reduit.

### 2026-03-19 - Support A-scan corrosion et no-data noir
**Tags :** `#branch:annotation`, `#controllers/corrosion_profile_controller.py`, `#controllers/master_controller.py`, `#models/view_state_model.py`, `#services/corrosion_profile_edit_service.py`, `#services/cscan_corrosion_service.py`, `#services/distance_measurement.py`, `#views/cscan_view_corrosion.py`, `#corrosion`, `#ascan`, `#support-map`, `#heatmap`, `#mvc`

**Actions effectuees :**
- Ajoute dans `CScanCorrosionService` un calcul `ascan_support_map (Z,X)` derive du volume brut actif, avec `build_fillable_support_mask()` pour autoriser le pontage de petits manques A-scan via `MAX_INTERPOLATION_GAP_PX` sans confondre absence de support et trous du mask.
- Modifie `DistanceMeasurementService.measure_distance_and_peaks_vectorized()` pour invalider pics et distances sur les colonnes sans support A-scan, meme si un mask corrosion y passe.
- Propage `corrosion_ascan_support_map` dans `ViewStateModel` et `MasterController`, puis le reutilise dans `CorrosionProfileController` et `CorrosionProfileEditService` afin que preview/commit des edits respectent les memes zones non supportees.
- Conserve les `NaN` dans la projection corrosion et rend explicitement les zones sans donnees en noir dans `CscanViewCorrosion`.
- Valide le comportement sur `Flexoform Demopipe.nde` : environ 40.94% des colonnes `(Z,X)` ont un profil A-scan entierement nul, et un test synthetique confirme qu une colonne sans support ne produit plus de distance meme avec un faux mask.

**Contexte :**
Le premier correctif de gap d interpolation utilisait les trous du `peak_map`/du mask comme proxy de manque A-scan. L analyse sur `Flexoform Demopipe` a montre que ce proxy est faux : des masques peuvent exister sur des colonnes ou le profil A-scan est entierement nul. Il fallait donc baser la coupure et l interpolation sur le support reel du signal, pas sur la geometrie du mask.

**Decisions techniques :**
1. Deriver le support A-scan depuis le volume brut actif (`get_active_raw_volume`) plutot que depuis le volume normalise, afin d eviter qu une normalisation masque les colonnes remplies a zero.
2. Considerer `MAX_INTERPOLATION_GAP_PX` comme un seuil de pontage des zones sans support A-scan uniquement ; avec `0`, les trous de support restent ouverts mais les trous de mask sur support valide peuvent encore etre interpoles.
3. Stocker explicitement le `support_map` dans le modele de vue corrosion pour garder un comportement coherent entre analyse initiale, preview d edition et commit du profil.

### 2026-03-20 - Heatmap corrosion robuste aux outliers
**Tags :** `#branch:annotation`, `#services/cscan_corrosion_service.py`, `#corrosion`, `#heatmap`, `#outlier`, `#percentile`, `#visualization`

**Actions effectuees :**
- Ajoute `HEATMAP_UPPER_PERCENTILE = 99.0` et une helper `compute_display_value_range()` dans `CScanCorrosionService` pour calculer une plage d affichage robuste a partir des distances finies.
- Remplace les `min/max` bruts par cette helper dans `run_analysis()` pour la projection corrosion principale et la projection interpolee.
- Fait passer `compute_corrosion_projection()` par la meme logique quand aucun `value_range` n est fourni, ce qui aligne aussi le recalcul du heatmap lors du commit du profil corrosion.
- Ajoute un fallback pour le cas degenerate ou le percentile haut retombe sur `vmin`, afin qu un outlier isole ne reintroduise pas le `max` brut dans l echelle de couleurs.

**Contexte :**
Certaines analyses corrosion selectionnent des pics aberrants qui produisent des distances tres grandes par rapport au reste de la carte. Avec une normalisation `min/max` brute, ces outliers ecrasent la dynamique du heatmap et la plupart des pixels retombent en rouge, ce qui rend la carte peu lisible.

**Decisions techniques :**
1. Corriger le probleme au niveau du service corrosion plutot que dans la vue, afin de centraliser la logique de plage d affichage et de garder le pipeline MVC coherent.
2. Conserver `vmin` sur le minimum reel et ne clipper que la borne haute par percentile, pour ne pas masquer artificiellement les petites distances potentiellement critiques.
3. Ne pas modifier la `distance_map` elle-meme : seules les couleurs sont saturees, ce qui preserve les valeurs brutes pour les mesures et le crosshair.

### 2026-03-23 - Export manuel du C-scan corrosion en NPZ et PNG
**Tags :** `#branch:annotation`, `#controllers/cscan_controller.py`, `#controllers/master_controller.py`, `#services/cscan_corrosion_service.py`, `#views/cscan_view.py`, `#views/cscan_view_corrosion.py`, `#corrosion`, `#cscan`, `#export`, `#png`, `#npz`, `#pyqt6`, `#mvc`

**Actions effectuees :**
- Ajoute dans `views/cscan_view.py` un point d extension de l en-tete, puis dans `views/cscan_view_corrosion.py` un bouton `Exporter` et un signal `export_requested`.
- Branche `controllers/master_controller.py` sur ce signal pour ouvrir l explorateur Windows via `QFileDialog.getExistingDirectory()` avec comme dossier initial le dossier du NDE ouvert.
- Ajoute dans `controllers/cscan_controller.py` une methode `export_corrosion_projection()` qui verifie qu une analyse corrosion est active et delegue l export au service.
- Etend `services/cscan_corrosion_service.py` pour sauvegarder le C-scan corrosion affiche sous `<nde>_cscan.npz` et generer en plus `<nde>_cscan.png` a partir de la projection 2D et de la palette corrosion.
- Fait remonter les deux chemins exportes au `MasterController` pour affichage dans le message de statut, sans introduire d auto-sauvegarde.

**Contexte :**
L utilisateur voulait un export manuel uniquement depuis la vue C-scan corrosion, avec choix explicite du dossier via l explorateur Windows. Le besoin a ensuite ete etendu pour produire non seulement le `NPZ` contenant la projection, mais aussi un `PNG` visuel directement exploitable, tout en conservant le dossier du NDE comme valeur par defaut et sans melanger rendu, orchestration UI et logique de sauvegarde.

**Decisions techniques :**
1. Garder la vue corrosion strictement limitee a l UI en n y ajoutant qu un bouton et un signal, toute l orchestration du dialogue Windows restant dans `MasterController`.
2. Centraliser la validation de l etat corrosion actif dans `CScanController` pour eviter qu un export soit declenche sans projection disponible.
3. Generer le `PNG` dans `CScanCorrosionService` a partir de la meme palette rouge-orange-jaune-bleu que la vue corrosion, afin d obtenir un rendu exporte coherent avec ce que l utilisateur voit a l ecran.

### 2026-03-23 - Sauvegarde .session individuelle par session d annotation
**Tags :** `#branch:annotation`, `#controllers/master_controller.py`, `#services/annotation_session_manager.py`, `#services/project_persistence.py`, `#ui_mainwindow.py`, `#untitled.ui`, `#session`, `#overlay`, `#persistence`, `#npz`, `#pyqt6`, `#mvc`

**Actions effectuees :**
- Ajoute `services/project_persistence.py` pour lire et ecrire des fichiers `.session` compresses contenant le chemin du NDE, le mode d annotation, les options de processing et un dump des sessions.
- Etend `services/annotation_session_manager.py` avec `build_dump()` et `restore_dump()` afin de serialiser et restaurer l etat courant du selecteur de sessions.
- Separe `Fichier > Sauvegarder` et `Ctrl+S` de `Overlay > Exporter .npz` dans `controllers/master_controller.py` ; l export NPZ reste autonome tandis que la sauvegarde ecrit des fichiers `.session`.
- Ajoute `Fichier > Ouvrir une session` dans `ui_mainwindow.py` et `untitled.ui`, puis refactorise le chargement NDE via `_load_nde_file()` pour rejouer le meme pipeline lors d une restauration de session.
- Fait sauvegarder une session par fichier, avec un nom deterministe `<nde>_<nom_session_normalise>_<id_session>.session`, de sorte que les trois sessions visibles dans le Session Selector produisent trois fichiers distincts.

**Contexte :**
Le bouton Sauvegarder reutilisait le flux d export NPZ alors que l utilisateur voulait un format `.session` distinct, contenant l overlay en memoire et une reference vers le NDE non modifie. Le besoin a ensuite ete precise pour les sessions multiples : chaque session visible dans le selector doit etre sauvegardee individuellement avec son propre ID et son nom, afin qu ouvrir un fichier `.session` ne recharge qu une seule session plutot que tout le gestionnaire.

**Decisions techniques :**
1. Conserver une separation stricte entre export `.npz` et sauvegarde `.session` pour garantir qu un export ne soit jamais ecrase ou modifie par `Ctrl+S`.
2. Sauvegarder une seule session par fichier `.session`, meme si plusieurs sessions sont presentes en memoire, afin d aligner la persistance sur les IDs exposes dans le Session Selector et sur le comportement attendu a l ouverture.
3. Normaliser le nom de session pour le filesystem Windows et centraliser le rechargement NDE dans `_load_nde_file()` pour eviter deux pipelines divergents entre ouverture manuelle et restauration de session.

### 2026-03-23 - Mesure corrosion au centre des plateaux satures
**Tags :** `#branch:annotation`, `#services/peak_plateau.py`, `#services/ascan_service.py`, `#services/distance_measurement.py`, `#corrosion`, `#ascan`, `#cscan`, `#peak`, `#plateau`, `#saturation`, `#measurement`, `#mvc`

**Actions effectuees :**
- Ajoute `services/peak_plateau.py` pour centraliser la selection d un pic sur plateau sature en prenant le milieu discret entre le premier et le dernier sample au maximum.
- Modifie `services/distance_measurement.py` pour produire `peak_map_a` et `peak_map_b` a partir du centre des plateaux max plutot que du premier max rencontre dans chaque colonne.
- Modifie `services/ascan_service.py` pour appliquer la meme logique dans le fallback A-scan corrosion quand les indices sont resolus depuis le signal local.
- Verifie le correctif sur des cas synthetiques avec plateau sature et avec pic unique afin de confirmer que le cas nominal reste stable.

**Contexte :**
Lors de l analyse corrosion, certains A-scan sont satures au sommet du peak, ce qui cree un plateau avec plusieurs valeurs maximales identiques. La mesure FW/BW se calait alors sur le bord du plateau, produisant un positionnement visuel et une distance moins representatifs du vrai centre du sommet. Le diff staged etait vide au moment du brew ; l entree documente le correctif local applique sur les services corrosion.

**Decisions techniques :**
1. Corriger la selection du pic dans la couche service plutot que dans la vue afin que l A-scan, les peak maps, l overlay et la distance FW/BW partagent la meme logique metier.
2. Introduire un helper dedie `peak_plateau.py` pour eviter de dupliquer la logique de centre de plateau entre `DistanceMeasurementService` et `AScanService`.
3. Pour les plateaux de largeur paire, utiliser le milieu discret superieur afin d eviter de retomber sur le premier maximum historique et de deplacer effectivement la mesure vers le centre du plateau.

### 2026-03-25 - Workflow de session individuel avec autosave temporaire et refactor MVC
**Tags :** `#branch:annotation`, `#controllers/master_controller.py`, `#controllers/session_workspace_controller.py`, `#services/annotation_session_manager.py`, `#controllers/annotation_controller.py`, `#controllers/corrosion_profile_controller.py`, `#views/session_manager_dialog.py`, `#ui_mainwindow.py`, `#untitled.ui`, `#session`, `#autosave`, `#persistence`, `#mvc`, `#pyqt6`, `#dirty-state`

**Actions effectuees :**
- Renomme `Sauvegarder` en `Enregistrer`, ajoute `Enregistrer sous` dans le menu Fichier et branche le workflow de sauvegarde sur la session active uniquement.
- Ajoute l ouverture de `.session` mono-session avec rattachement du chemin charge a la session restauree, de sorte que `Enregistrer` reecrive ensuite le bon fichier.
- Introduit un suivi `dirty` par session, des confirmations `Enregistrer / Ne pas enregistrer / Annuler` avant quitter, ouvrir un autre NDE, ouvrir une autre session et supprimer une session.
- Ajoute un autosave temporaire debounced des sessions modifiees vers le dossier temp, base sur le dump complet d une session unique persistable.
- Extrait la logique de workflow session hors de `MasterController` dans `controllers/session_workspace_controller.py`, en laissant `ProjectPersistence` comme couche bas niveau de lecture/ecriture de fichier.
- Etend `AnnotationSessionManager` avec la creation de session vide, le renommage, les helpers `build_session_dump()` / `build_active_dump()` et le nom par defaut `New session`.
- Revoit `SessionManagerDialog` pour separer `Creer` et `Dupliquer`, supprimer l affichage de l ID hardcode, et permettre la creation de sessions vides.
- Fait remonter les mutations de session depuis l application d annotation, l undo/redo et le commit du profil corrosion afin de marquer correctement la session courante comme modifiee.
- Fait creer au pipeline corrosion une nouvelle session derivee suffixee `corrosion` au lieu d ecraser silencieusement la session source.

**Contexte :**
Le workflow precedent melangeait encore orchestration UI, persistance de session, suivi des modifications non enregistrees et autosave dans `MasterController`. En parallele, la sauvegarde devait etre rebranchee pour enregistrer une seule session a la fois, ouvrir directement `Enregistrer sous` si la session n avait pas encore de fichier, et rendre le selecteur de sessions plus propre avec creation vide, duplication explicite et noms alignes sur les fichiers `.session`.

**Decisions techniques :**
1. Centraliser le workflow document/session dans un `SessionWorkspaceController` dedie plutot que de charger `AnnotationSessionManager` ou `ProjectPersistence` avec des responsabilites UI, afin de conserver une separation MVC defendable.
2. Reutiliser le dump complet d une session persistable pour l autosave temporaire et pour les confirmations de fermeture, plutot que le systeme undo/redo qui ne contient que des slices appliquees et ne represente pas tout l etat de session.
3. Faire du nom de fichier `.session` le nom de session visible et sauvegarder une seule session par fichier, afin d eliminer les noms hardcodes/IDs du selecteur et d aligner le comportement utilisateur avec l ouverture/sauvegarde document par document.

### 2026-03-25 - Regle agent locale alignee sur le workflow memory-first
**Tags :** `#branch:annotation`, `#.agent/rules/agent-context.md`, `#agent`, `#memory-first`, `#brew`, `#workflow`

**Actions effectuees :**
- Ajoute `.agent/rules/agent-context.md` comme regle locale always-on reprenant le workflow memory-first, la distinction `rag`/`brew`, la contrainte de plan valide et la procedure Ragbrew.
- Aligne les consignes locales sur les etapes de consultation memoire, lecture directe de `MEMORY.md` en fallback, et rebuild obligatoire de l index apres documentation.

**Contexte :**
Le diff staged ajoute une regle locale pour que l agent applique directement dans l IDE le meme cadre d execution que celui defini au niveau du depot, en particulier pour les sequences `rag` et `brew`.

**Decisions techniques :**
1. Dupliquer la regle sous `.agent/rules/agent-context.md` plutot que de s appuyer uniquement sur `AGENTS.md`, afin que le contexte local de l agent reste coherent avec le workflow du projet.
2. Garder la regle en mode `always_on` pour imposer memory-first et la discipline de documentation sans dependre d un rappel manuel.

### 2026-03-25 - Opacite NDE synchronisee sur endviews et vue 3D
**Tags :** `#branch:annotation`, `#controllers/endview_controller.py`, `#controllers/master_controller.py`, `#models/view_state_model.py`, `#views/endview_view.py`, `#views/tools_panel.py`, `#views/volume_view.py`, `#nde`, `#opacity`, `#pyqt6`, `#vispy`, `#mvc`

**Actions effectuees :**
- Ajoute `nde_alpha` dans `models/view_state_model.py` avec une valeur par defaut a `1.0` pour conserver un NDE full opaque tant qu aucun reglage n est applique.
- Complete `views/tools_panel.py` avec un vrai signal `nde_opacity_changed`, un setter de synchro UI et une activation conditionnelle du controle NDE.
- Ajoute `set_nde_opacity()` dans `views/endview_view.py` pour piloter l opacite de l image de base sur les endviews sans toucher a l overlay.
- Ajoute `set_nde_opacity()` dans `views/volume_view.py` et reapplique la valeur sur le `VolumeVisual` VisPy lors des rebuilds de scene.
- Etend `controllers/endview_controller.py` et `controllers/master_controller.py` pour propager l opacite NDE vers toutes les endviews, la vue 3D, le chargement NDE et les switches de session.

**Contexte :**
Le besoin etait de disposer d une opacite NDE distincte de l overlay, avec le meme mode de pilotage UI, une valeur par defaut a 100 pour cent et une application coherente sur toutes les vues qui affichent le signal NDE brut.

**Decisions techniques :**
1. Stocker l opacite NDE dans `ViewStateModel` comme etat de vue persistant, afin qu elle survive aux switches de session et reste dans la couche modele.
2. Appliquer l opacite du NDE directement sur les renderers de base (`QGraphicsPixmapItem` pour les endviews, `VolumeVisual.opacity` pour la 3D) plutot que de melanger cette logique avec l overlay.
3. Reprendre le meme schema de propagation que pour l opacite overlay, avec `ToolsPanel` comme source UI, `MasterController` comme orchestrateur et les vues limitees au rendu.

### 2026-03-26 - Navigation endview relocalisee dans les vues U V
**Tags :** `#branch:annotation`, `#controllers/endview_controller.py`, `#controllers/master_controller.py`, `#toolspanel.ui`, `#ui_toolspanel.py`, `#views/endview_view.py`, `#views/tools_panel.py`, `#endview`, `#navigation`, `#titlebar`, `#pyqt6`, `#mvc`

**Actions effectuees :**
- Retire du `ToolsPanel` les labels NDE Endview position et les controles de navigation primaire secondaire, avec mise a jour de `toolspanel.ui`, `ui_toolspanel.py` et `views/tools_panel.py` pour ne garder que les outils, opacites et colormap.
- Etend `views/endview_view.py` avec une UI locale de navigation composee d un titre d axe, d un slider, d un spinbox et d une ligne de statut affichant le nom d endview et, pour la vue d annotation, la position courante.
- Etend `controllers/endview_controller.py` avec des helpers pour pousser bornes de navigation, noms d axes, noms d endview distincts primaire secondaire et texte de position sur la vue principale.
- Rebranche `controllers/master_controller.py` pour piloter directement les controles des vues U et V, mettre le chemin NDE complet dans le titlebar principal, et synchroniser un nom d endview propre a chaque vue a partir de `current_slice` et `secondary_slice`.

**Contexte :**
Le workflow precedent affichait encore la navigation des endviews dans le dock outils alors que les vues U et V etaient deja distinctes. Le besoin etait de rapprocher les controles de leur vue respective, de montrer le chemin NDE dans la fenetre principale, et de differencier explicitement le nom de l endview principale de celui de l endview secondaire.

**Decisions techniques :**
1. Conserver `EndviewView` comme composant de base et lui ajouter une petite UI locale, plutot que creer un nouveau widget composite, afin de limiter l impact sur `AnnotationView`, la vue corrosion et le layout ADS existant.
2. Garder `MasterController` comme orchestrateur des indices `current_slice` et `secondary_slice`, mais deplacer l affichage de navigation dans les vues pour respecter une separation MVC ou la vue possede ses widgets et le controleur pousse seulement l etat.
3. Gerer deux setters distincts pour les noms d endview primaire et secondaire dans `EndviewController`, afin d eviter que la vue orthogonale reutilise par erreur le nom calcule pour la vue principale.

### 2026-03-27 - Action Draw Erase et box finalisee au relachement
**Tags :** `#branch:annotation`, `#controllers/annotation_controller.py`, `#controllers/master_controller.py`, `#models/view_state_model.py`, `#toolspanel.ui`, `#ui_toolspanel.py`, `#views/annotation_view.py`, `#views/tools_panel.py`, `#annotation`, `#draw`, `#erase`, `#threshold`, `#shortcuts`, `#box`, `#pyqt6`, `#mvc`

**Actions effectuees :**
- Ajoute dans `toolspanel.ui` et `ui_toolspanel.py` un combo `Action` avec `Draw` et `Erase`, place a cote du choix d outil dans le dock d annotation.
- Etend `views/tools_panel.py` avec un signal `annotation_action_changed`, des helpers de lecture selection et de synchro UI pour piloter le nouveau combo d action.
- Etend `models/view_state_model.py` avec un etat `annotation_action`, initialise le label actif sur un label de dessin `> 0`, et expose un label effectif ainsi qu un threshold effectif forces a `0` quand l action `Erase` est active.
- Rebranche `controllers/master_controller.py` pour connecter le combo d action, ajouter les raccourcis clavier `R` pour `Draw` et `E` pour `Erase`, puis retirer le label `0` de la liste des labels actifs tout en conservant sa logique d effacement dans les modeles et les settings NDE.
- Rebranche `controllers/annotation_controller.py` pour que les outils `grow`, `line`, `free hand`, `peak`, `box` et `paint` utilisent le label effectif et le threshold effectif, afin que `Erase` supprime 100 pour cent de la zone sans modifier le threshold memorise pour `Draw`.
- Modifie `views/annotation_view.py` pour finaliser une ROI `box` au relachement du premier clic gauche au lieu d attendre un second clic, avec preview conservee pendant le drag.

**Contexte :**
Le besoin etait de separer clairement l intention utilisateur entre dessin et effacement, de ne plus exposer `label 0` comme label actif dans l interface, et de garantir qu en mode `Erase` la zone marquee soit entierement effacee sans etre filtree par le threshold. En parallele, le mode `box` devait etre fluidifie en supprimant le deuxieme clic de validation.

**Decisions techniques :**
1. Decoupler l action `Draw` ou `Erase` du `active_label` dans `ViewStateModel`, plutot que surcharger directement la selection de label, afin de preserver les workflows existants de modification de masque, corrosion et persistance de session.
2. Conserver le slider de threshold comme etat de `Draw`, mais appliquer un threshold effectif a `0` uniquement dans `AnnotationController` quand l action `Erase` est active, pour eviter toute perte du reglage utilisateur.
3. Garder `label 0` dans la palette et dans les mecanismes d effacement deja existants, tout en le filtrant de la liste des labels actifs du `ToolsPanel`, afin de reutiliser l infrastructure d erase sans exposer un faux label de dessin.

### 2026-03-27 - Autosave asynchrone robuste et sauvegarde session allegee
**Tags :** `#branch:annotation`, `#controllers/master_controller.py`, `#controllers/session_workspace_controller.py`, `#services/annotation_session_manager.py`, `#services/project_persistence.py`, `#session`, `#autosave`, `#persistence`, `#performance`, `#corrosion`, `#pyqt6`, `#mvc`

**Actions effectuees :**
- Rebranche `controllers/session_workspace_controller.py` sur un vrai worker asynchrone `SessionSaveWorker` via `QThreadPool`, avec conservation explicite des workers actifs pour eviter la destruction prematuree de `SessionSaveSignals`.
- Ajoute dans `controllers/session_workspace_controller.py` un suivi des autosaves en vol, un nettoyage differe des fichiers temporaires verrouilles et un retry court pour eviter les `PermissionError` Windows pendant `unlink()`.
- Etend `services/annotation_session_manager.py` avec une copie legere du `ViewStateModel` qui ne duplique plus les `np.ndarray` derives, reutilisee dans les snapshots de session, l application de session, le dump persistant et la creation de la session corrosion.
- Fait retourner `build_session_dump()` sous `services/annotation_session_manager.py` un etat persistable sans `overlay_cache`, afin d eviter de serialiser un cache derive reconstructible.
- Abaisse le niveau de compression gzip dans `services/project_persistence.py` a `COMPRESS_LEVEL = 1` pour privilegier la vitesse de sauvegarde des `.session`.
- Remplace dans `controllers/master_controller.py` le `copy.deepcopy(self.view_state_model)` du flux corrosion par la copie legere centralisee du gestionnaire de sessions.

**Contexte :**
Le flux de sauvegarde de session restait lent meme apres passage en worker, car une partie du travail couteux etait faite avant l ecriture disque: copies profondes des etats corrosion dans le `view_state`, payload persistant trop lourd, et compression gzip maximale. En parallele, l autosave temporaire provoquait des erreurs `PermissionError` et `RuntimeError` quand un fichier temp etait supprime pendant qu un worker l ecrivait encore ou quand les signaux du worker etaient detruits trop tot.

**Decisions techniques :**
1. Stabiliser d abord le cycle de vie des autosaves en conservant explicitement les workers et en deferant le nettoyage des fichiers temporaires, plutot que masquer les exceptions ou desactiver l autosave.
2. Traiter les tableaux corrosion comme des donnees derivees quasi immuables dans les copies de `view_state`, afin d eliminer les `deepcopy` couteux sans casser le switch de session ni la restauration.
3. Distinguer l etat en memoire utile au runtime du payload persistant disque, en excluant `overlay_cache` du dump `.session` et en baissant la compression pour privilegier la rapidite sur le stockage local.

### 2026-03-27 - Etat ToolsPanel restaure par session et threshold erase optionnel
**Tags :** `#branch:annotation`, `#controllers/annotation_controller.py`, `#controllers/master_controller.py`, `#models/view_state_model.py`, `#toolspanel.ui`, `#ui_toolspanel.py`, `#views/tools_panel.py`, `#session`, `#tools_panel`, `#state-management`, `#threshold`, `#erase`, `#pyqt6`, `#mvc`

**Actions effectuees :**
- Rebranche `controllers/master_controller.py` pour reappliquer au demarrage et apres `_after_session_switch()` les etats `threshold`, `Box percentiles`, `Apply volume`, `ROI persistence`, `paint size` et la synchro du range volume, afin que le `ToolsPanel` refl ete bien la session active.
- Etend `views/tools_panel.py` avec des setters silencieux pour les checkboxes de threshold auto, apply volume, ROI persistence et, dans le diff stage courant, pour le nouveau checkbox `Force threshold (erase)`.
- Ajoute dans `toolspanel.ui` et `ui_toolspanel.py` le checkbox `Force threshold (erase)` et remappe dans `MasterController` le checkbox du crosshair sur `checkBox_6` apres l insertion du nouveau controle.
- Etend `models/view_state_model.py` avec `force_threshold_erase`, expose son setter, et ne force plus `effective_annotation_threshold()` a `0` en mode `Erase` quand l opt-in est coche.
- Rebranche `controllers/annotation_controller.py` et `controllers/master_controller.py` pour propager le toggle `force_threshold_erase` depuis le `ToolsPanel` jusqu au modele de vue.

**Contexte :**
Le commit `924b706e962564b110ac54dfdd89673fdb48930b` corrigeait un decalage entre l etat reel des sessions et l etat affiche par le dock outils, visible au chargement initial et apres un switch de session. Le diff stage courant ajoute ensuite un besoin plus fin en annotation : conserver `Erase` comme effacement total par defaut, tout en permettant explicitement d appliquer le threshold courant quand l utilisateur veut un erase seuille.

**Decisions techniques :**
1. Restaurer l etat du `ToolsPanel` depuis `ViewStateModel` dans `MasterController` plutot que laisser la vue deduire seule ses valeurs, afin de garder une source de verite unique par session.
2. Garder `Erase` avec un threshold effectif a `0` par defaut pour ne pas casser le comportement introduit precedemment, puis ajouter un opt-in explicite `Force threshold (erase)` dans le modele de vue pour les cas de seuillage volontaire.
3. Ajouter le nouveau controle dans le dock existant et utiliser des setters silencieux dans `ToolsPanel` pour eviter les emissions de signaux parasites lors des restaurations de session et de l initialisation UI.

### 2026-03-27 - Apply auto sur tous les tools ROI et nettoyage overlay cross
**Tags :** `#branch:annotation`, `#controllers/annotation_controller.py`, `#controllers/master_controller.py`, `#models/view_state_model.py`, `#toolspanel.ui`, `#ui_toolspanel.py`, `#views/tools_panel.py`, `#apply-auto`, `#temp-mask`, `#roi`, `#tools_panel`, `#overlay`, `#cross`, `#session`, `#pyqt6`, `#mvc`

**Actions effectuees :**
- Reorganise `toolspanel.ui` et `ui_toolspanel.py` pour regrouper `Apply auto`, `Force threshold (erase)`, `Apply volume`, `Box percentiles` et `ROI persistence` dans le meme bloc options du `ToolsPanel`.
- Etend `views/tools_panel.py` avec le nouveau toggle `apply_auto_toggled`, un setter silencieux `set_apply_auto_checked()`, et retire les anciens reliquats `overlay/cross` qui ne doivent plus etre portes par le panneau.
- Etend `models/view_state_model.py` avec l etat persistant `apply_auto`, puis rebranche `controllers/master_controller.py` pour restaurer ce toggle au demarrage et apres `_after_session_switch()`.
- Ajoute dans `controllers/annotation_controller.py` le setter `on_apply_auto_toggled()` et fait retourner un booleen aux handlers ROI/temp mask (`grow`, `paint`, `line`, `free_hand`, `peak`, `box`) pour indiquer si une preview exploitable a ete creee.
- Centralise dans `controllers/master_controller.py` des wrappers d interaction qui declenchent le pipeline standard `_apply_roi_non_corrosion()` quand `Apply auto` est actif, y compris sur la fin de drag du tool `mod`.

**Contexte :**
Le nouveau checkbox `Apply auto` avait ete ajoute dans le `ToolsPanel` pour appliquer automatiquement le temp mask a la fin d une action utilisateur, mais il fallait d une part le persister par session, et d autre part etendre le comportement au dela du seul free hand. En parallele, le panneau gardait encore un vieux contrat optionnel pour `overlay` et `cross` alors que ces toggles vivent desormais dans le menu `Affichage`.

**Decisions techniques :**
1. Garder `ViewStateModel` comme source de verite du toggle `Apply auto`, afin que l etat suive naturellement les snapshots et restaurations de session sans logique speciale cote persistence.
2. Lancer l auto-apply depuis `MasterController` apres creation reussie de la preview par chaque tool, pour reutiliser le pipeline standard d apply, l historique undo/redo et le marquage dirty de session.
3. Supprimer le contrat `overlay/cross` du `ToolsPanel` plutot que le maintenir optionnel par inertie, afin d eliminer les collisions de mapping avec le nouveau checkbox `Apply auto` et d aligner clairement l UI sur la source d etat du menu `Affichage`.

### 2026-03-27 - Push overlay VolumeView rendu optionnel
**Tags :** `#branch:annotation`, `#controllers/annotation_controller.py`, `#controllers/master_controller.py`, `#models/view_state_model.py`, `#toolspanel.ui`, `#ui_toolspanel.py`, `#views/tools_panel.py`, `#overlay`, `#volume_view`, `#tools_panel`, `#3d-visualization`, `#session`, `#pyqt6`, `#mvc`

**Actions effectuees :**
- Ajoute dans `toolspanel.ui` et `ui_toolspanel.py` une case `Volume view` dediee au controle du push overlay vers la vue 3D.
- Etend `views/tools_panel.py` avec le signal `volume_view_overlay_toggled`, le stockage du nouveau checkbox Designer et un setter silencieux `set_volume_view_overlay_checked()`.
- Etend `models/view_state_model.py` avec l etat persistant `show_volume_view_overlay`, initialise a `True`, pour separer le toggle global overlay du push specifique a `VolumeView`.
- Rebranche `controllers/master_controller.py` pour injecter `checkBox_6`, connecter son signal vers `AnnotationController`, et restaurer son etat au demarrage ainsi qu apres `_after_session_switch()`.
- Modifie `controllers/annotation_controller.py` pour que `refresh_overlay()` continue de pousser l overlay 2D quand `show_overlay` est actif, mais n envoie l overlay a `volume_view` que si `show_volume_view_overlay` est coche; sinon la 3D est nettoyee seule.

**Contexte :**
Le besoin etait de rendre facultatif le push de l overlay d annotation dans `VolumeView` sans changer le comportement des overlays 2D. L utilisateur devait pouvoir activer ou desactiver cette projection 3D depuis le `ToolsPanel`, tout en conservant le toggle global overlay comme commande de visibilite generale.

**Decisions techniques :**
1. Introduire un etat dedie dans `ViewStateModel` plutot que reutiliser `show_overlay`, afin de dissocier clairement la visibilite overlay globale de l envoi specifique a la vue 3D.
2. Garder `AnnotationController.refresh_overlay()` comme point unique de push overlay vers les vues 2D et 3D, pour preserver l orchestration MVC et eviter des branches de logique dispersees.
3. Restaurer explicitement la checkbox `Volume view` depuis `MasterController` pendant l initialisation et les switches de session, afin que le choix utilisateur soit coherent avec l etat de la session active.

### 2026-03-27 - Export nnU-Net autonome depuis le split dataset
**Tags :** `#branch:annotation`, `#controllers/master_controller.py`, `#services/split_service.py`, `#nnunet`, `#split`, `#export`, `#dataset`, `#ucoord`, `#vcoord`, `#prefix-suffix`, `#mvc`

**Actions effectuees :**
- Etend `services/split_service.py` avec un contexte d export partage et un nouveau flux `export_nnunet_dataset()` qui genere un dataset local autonome dans `imagesTr/` et `labelsTr/`.
- Fait reutiliser par `services/split_service.py` le volume source ou traite deja charge dans l application, calcule la normalisation grayscale uint8, et exporte les deux orientations primaire et secondaire en suffixant automatiquement les noms avec `_ucoord` ou `_vcoord`.
- Conserve dans `services/split_service.py` le split `flaw/noflaw` existant, mais le refactorise autour des memes helpers de contexte, de min/max et d ecriture PNG pour eviter la duplication de logique.
- Rebranche `controllers/master_controller.py` pour que l action `Split flaw/noflaw` ouvre d abord un choix de type d export (`flaw/noflaw` ou `nnU-Net`), puis reutilise les memes prompts dossier, prefixe et suffixe avant d appeler le bon flux de service.
- Ajoute dans `controllers/master_controller.py` un helper centralise `_current_signal_processing_options()` pour propager les options Hilbert/lissage du modele actif vers les deux exports.

**Contexte :**
Le besoin etait d ajouter un export de dataset compatible nnU-Net directement dans le repo courant, sans dependance runtime vers `nnunet-pipeline`, tout en gardant le split `flaw/noflaw` existant. L utilisateur voulait choisir librement le dossier de sortie, conserver les options de prefixe/suffixe deja en place, et produire aussi bien les endviews `UCoordinate` que `VCoordinate` avec un suffixe d axe ajoute automatiquement.

**Decisions techniques :**
1. Garder toute la logique de generation du dataset dans `services/split_service.py` plutot que pointer vers un autre repo, afin que le projet reste autonome et conforme a l architecture MVC existante.
2. Reutiliser l action menu existante et ajouter un prompt de mode dans `controllers/master_controller.py`, afin d eviter de toucher les fichiers UI deja modifies localement et de limiter le changement a l orchestration controller.
3. Produire directement des paires de noms identiques entre `imagesTr/` et `labelsTr`, avec suffixe automatique d axe ajoute apres le suffixe utilisateur, afin de rester compatible avec un renommage ulterieur sans imposer de convention externe supplementaire.

### 2026-03-30 - Remap definitif des labels persistants sans classe 100
**Tags :** `#branch:annotation`, `#MEMORY.md`, `#config/constants.py`, `#controllers/master_controller.py`, `#models/view_state_model.py`, `#services/corrosion_label_service.py`, `#labels`, `#palette`, `#corrosion`, `#mvc`

**Actions effectuees :**
- Remplace dans `config/constants.py` le schema persistant `0,1,2,3,100` par `0,1,2,3,4`, introduit des constantes semantiques de labels, et fait demarrer les labels libres a `5` pour obtenir `BW echo 1 (5)`, `BW echo 2 (6)`, etc.
- Rebranche dans `config/constants.py` les noms UI definitifs sur `Background`, `Reflector`, `Paint`, `FW` et `BW`, puis realigne les palettes BGR/RGB/BGRA pour que les couleurs suivent le sens metier plutot que l ancien id numerique `100`.
- Modifie `models/view_state_model.py` et `controllers/master_controller.py` pour que le label actif par defaut reste `Paint (2)` meme si `Reflector (1)` devient le premier label persistant non nul.
- Met a jour `services/corrosion_label_service.py` pour que la corrosion prefere desormais `FW (3)` et `BW (4)` avec le nouveau remap des classes.

**Contexte :**
Le design precedent exposait `Reflector` sur une classe persistante `100`, alors que l utilisateur voulait repartir d un schema simple et definitif ou toutes les classes persistantes utiles vivent entre `1` et `4`. La retrocompatibilite avec les anciens `.npz` et anciennes sessions n etait pas necessaire, l utilisateur prevoyant de supprimer ces anciens artefacts et de recommencer sur la nouvelle nomenclature.

**Decisions techniques :**
1. Centraliser le remap dans `config/constants.py` avec des ids semantiques nommes, afin d eviter de repropager des nombres magiques dans les controllers, services et vues deja branches sur `format_label_text()`.
2. Faire commencer les labels libres a `5` plutot qu a `4`, afin que `4` devienne `BW` persistant et que la suite des labels utilisateur conserve la nomenclature `BW echo N` sans trou.
3. Introduire des constantes explicites pour le label actif par defaut et la paire corrosion par defaut, afin de garder un comportement fonctionnel stable malgre le changement d ordre des ids persistants.

### 2026-03-30 - Remap non destructif des classes pour overlays NPZ importes
**Tags :** `#branch:annotation`, `#MEMORY.md`, `#controllers/master_controller.py`, `#models/annotation_model.py`, `#models/imported_overlay_model.py`, `#services/overlay_class_remap_service.py`, `#views/overlay_class_remap_dialog.py`, `#ui_mainwindow.py`, `#untitled.ui`, `#overlay`, `#npz`, `#labels`, `#mvc`, `#pyqt6`

**Actions effectuees :**
- Ajoute dans `ui_mainwindow.py` et `untitled.ui` une action menu `Overlay > Remap classes`, puis la branche dans `controllers/master_controller.py`.
- Introduit `models/imported_overlay_model.py` pour conserver en memoire le NPZ importe original, ses classes detectees, le mapping courant et le dernier masque applique, sans jamais modifier le fichier source.
- Etend `models/annotation_model.py` avec `detected_label_ids` et `get_detected_label_ids()` afin de reutiliser la detection deja faite au chargement via `set_mask_volume()` plutot que dupliquer une inspection des classes ailleurs.
- Cree `services/overlay_class_remap_service.py` comme service pur de validation et de remap source->target sur un volume `uint8` 3D en memoire.
- Cree `views/overlay_class_remap_dialog.py` pour afficher les classes detectees, editer les classes cibles, previsualiser le libelle cible via `format_label_text()` et proposer un reset identite.
- Refactorise `controllers/master_controller.py` pour centraliser la re-injection d un masque overlay via `_apply_overlay_mask_volume()`, memoriser la source lors de `_on_load_npz()`, invalider cette source quand un NDE ou un resultat nnUNet remplace l overlay, et appliquer le remap comme un nouveau chargement overlay afin de rafraichir labels, vues, corrosion et etat dirty.

**Contexte :**
Le besoin etait d ajouter un remap de classes depuis le menu principal pour les overlays NPZ importes, avec une fenetre d edition et un comportement non destructif. L utilisateur voulait s inspirer du script externe `npz_remap_classes.py` sans l integrer tel quel, et surtout reutiliser la logique deja existante dans l application pour la detection des classes et la reconstruction des labels au chargement d un NPZ.

**Decisions techniques :**
1. Reutiliser la detection de classes deja realisee par `AnnotationModel.set_mask_volume()` plutot que recopier la phase d inspection du script externe, afin de garder une seule source de verite pour les classes presentes dans le masque courant.
2. Stocker un snapshot du NPZ importe original dans un modele runtime dedie, afin que chaque remap reparte de la source importee et reste strictement non destructif vis-a-vis du fichier charge et de l historique de remaps.
3. Rejouer le pipeline overlay existant apres remap via un helper controller centralise, afin que la mise a jour des labels, de l overlay 2D/3D, des vues corrosion et de l export NPZ reste coherente comme si le fichier d origine avait eu ces classes des le depart.

### 2026-03-30 - Contraste NDE implemente separement de l opacite
**Tags :** `#branch:annotation`, `#MEMORY.md`, `#controllers/endview_controller.py`, `#controllers/master_controller.py`, `#models/view_state_model.py`, `#views/endview_view.py`, `#views/tools_panel.py`, `#views/volume_view.py`, `#nde`, `#contrast`, `#opacity`, `#endview`, `#3d`, `#mvc`

**Actions effectuees :**
- Ajoute `nde_contrast` dans `models/view_state_model.py` comme etat de vue persistant, avec setter borne entre `0.0` et `2.0` et valeur neutre a `1.0`.
- Branche dans `views/tools_panel.py` le slider/spinbox Designer existants `horizontalSlider_7` et `spinBox_5` sous le signal `nde_contrast_changed`, avec synchro UI en pourcentage `0-200` et activation/desactivation liees a la disponibilite de l affichage NDE.
- Etend `controllers/master_controller.py` pour injecter les widgets de contraste, synchroniser le contraste au chargement, aux refreshs et apres switch de session, et propager les changements vers les vues via `_on_nde_contrast_changed`.
- Etend `controllers/endview_controller.py` pour diffuser `set_nde_contrast()` vers les vues principales, secondaires et corrosion.
- Modifie `views/endview_view.py` pour appliquer le contraste au rendu NDE par remapping d intensite autour du milieu de gamme apres normalisation de slice, sans toucher a `set_nde_opacity()` ni a l overlay.
- Modifie `views/volume_view.py` pour appliquer le meme contraste au volume NDE 3D via `VolumeVisual.clim`, tout en conservant l opacite NDE comme controle d alpha distinct.

**Contexte :**
Un slider `NDE contraste` existait deja dans le `ToolsPanel`, mais n etait relie a aucun etat ni rendu. La demande etait de l implementer en faisant explicitement attention a ne pas confondre contraste et opacite: le contraste doit agir sur les intensites du signal NDE, alors que l opacite doit continuer a ne piloter que la transparence du rendu.

**Decisions techniques :**
1. Stocker le contraste dans `ViewStateModel` au meme niveau que `nde_alpha`, afin qu il survive aux refreshs de vues et aux switches de session sans introduire de logique d etat dans les vues.
2. Definir `100%` comme valeur neutre et exposer une plage `0-200%`, afin de reutiliser le contrat UI des autres sliders tout en gardant une lecture simple pour l utilisateur.
3. Appliquer le contraste par remapping d intensite autour de `0.5` dans `EndviewView` et par ajustement de `clim` dans `VolumeView`, afin de modifier la dynamique visuelle du signal NDE sans melanger cette logique avec l alpha du NDE ni avec l overlay.

### 2026-03-30 - Drag continu pour l outil paint
**Tags :** `#branch:annotation`, `#MEMORY.md`, `#controllers/annotation_controller.py`, `#controllers/master_controller.py`, `#views/annotation_view.py`, `#paint`, `#drag`, `#brush`, `#apply-auto`, `#mvc`, `#pyqt6`

**Actions effectuees :**
- Ajoute dans `views/annotation_view.py` des signaux dedies `paint_stroke_started`, `paint_stroke_moved` et `paint_stroke_finished`, avec un etat local de stroke pour capter le clic-glisse du pinceau sans perturber les autres outils.
- Rebranche `controllers/master_controller.py` pour router ces nouveaux signaux vers `AnnotationController`, puis declencher `Apply auto` une seule fois a la fin du trait au lieu du premier press.
- Etend `controllers/annotation_controller.py` avec un etat de stroke paint, une interpolation des points entre deux positions souris, et un helper `_handle_paint_points()` qui applique un segment complet dans le `TempMaskModel`.
- Conserve dans `controllers/annotation_controller.py` la logique existante de restriction globale, blocked masks, preview temp mask et erase via label `0`, tout en la reappliquant a chaque segment du drag.

**Contexte :**
L outil `paint` ne gerait jusque-la qu un clic simple. Le besoin etait d obtenir un vrai pinceau en clic-glisse, avec un trait continu, sans casser l architecture MVC ni la logique deja en place pour le preview temporaire, l effacement par label `0` et le mode `Apply auto`.

**Decisions techniques :**
1. Gerer la capture press/move/release du pinceau dans `AnnotationView` plutot que detourner `drag_update`, afin de garder ce dernier dedie au suivi de position/crosshair et de limiter l impact sur les autres interactions Endview.
2. Interpoler les points du trait dans `AnnotationController` entre deux evenements souris, afin d eviter les trous visuels quand la souris se deplace plus vite que la frequence des events.
3. Reporter l auto-application du `paint` a la fin du stroke dans `MasterController`, afin d eviter qu un drag en mode `Apply auto` ne committe le masque des le premier press puis efface le preview avant la fin du geste.

### 2026-03-30 - Closing mask ROI configurable avec comblement de trous et fusion proche
**Tags :** `#branch:annotation`, `#MEMORY.md`, `#controllers/annotation_controller.py`, `#controllers/master_controller.py`, `#models/view_state_model.py`, `#services/annotation_service.py`, `#services/annotation_session_manager.py`, `#toolspanel.ui`, `#ui_toolspanel.py`, `#views/nde_settings_view.py`, `#views/tools_panel.py`, `#closing-mask`, `#roi`, `#morphology`, `#mvc`, `#ui`

**Actions effectuees :**
- Branche le checkbox `Closing mask` deja present dans `toolspanel.ui` et `ui_toolspanel.py` vers `views/tools_panel.py`, `controllers/master_controller.py` et l etat de session.
- Ajoute dans `views/nde_settings_view.py` deux parametres `Closing mask`: aire max de trou (`px2`) et distance de fusion (`px`).
- Etend `models/view_state_model.py` avec `closing_mask_enabled`, `closing_mask_tolerance` et `closing_mask_merge_distance`.
- Durcit `services/annotation_session_manager.py` pour reappliquer les valeurs par defaut du `ViewStateModel` quand une session plus ancienne ne contient pas encore ces nouveaux champs.
- Centralise dans `services/annotation_service.py` un pipeline ROI de closing borne par la zone autorisee: fermeture morphologique pour relier les composants proches, puis remplissage des trous internes par aire maximale.
- Propage ces parametres dans `controllers/annotation_controller.py` pour les flux ROI `box`, `free hand`, `grow`, `line`, `peak`, ainsi que `Recalculer ROI` et les variantes range/volume, sans modifier `paint` ni `mod`.

**Contexte :**
Le besoin etait d activer le checkbox `Closing mask` deja ajoute dans l interface, d abord pour combler automatiquement les petits trous dans les masks ROI, puis pour relier entre eux des morceaux tres proches afin d obtenir des regions plus compactes. Les reglages devaient rester modifiables depuis `Fichier > Parametres` tout en conservant la separation MVC entre vue, modele d etat et service d annotation.

**Decisions techniques :**
1. Garder un seul toggle `Closing mask` et separer ses reglages en deux parametres orthogonaux (`aire max trou` et `distance fusion`) afin de distinguer le remplissage des cavites internes du rapprochement de composants voisins.
2. Centraliser l algorithme dans `AnnotationService` plutot que dans `AnnotationController` ou les vues, afin de reutiliser exactement le meme post-traitement pour le preview, le recompute ROI et les applies volume.
3. Contraindre le closing par un `allowed_mask` derive des restrictions et zones bloquees, afin d eviter de fusionner ou remplir a travers des zones interdites.
4. Limiter explicitement le scope aux outils ROI (`box`, `free hand`, `grow`, `line`, `peak`) et exclure `paint` / `mod`, conformement au comportement demande.

### 2026-03-31 - Mask cleanup ROI avec ilots, excroissances, entailles et lissage de contour
**Tags :** `#branch:annotation`, `#MEMORY.md`, `#controllers/annotation_controller.py`, `#controllers/master_controller.py`, `#models/view_state_model.py`, `#services/annotation_service.py`, `#toolspanel.ui`, `#ui_toolspanel.py`, `#views/nde_settings_view.py`, `#views/tools_panel.py`, `#mask-cleanup`, `#clean-outliers`, `#contour-smoothing`, `#roi`, `#mvc`, `#morphology`

**Actions effectuees :**
- Branche le checkbox `Mask cleanup` dans `toolspanel.ui`, `ui_toolspanel.py`, `views/tools_panel.py` et `controllers/master_controller.py`, avec persistance via `ViewStateModel`.
- Etend `models/view_state_model.py` et `views/nde_settings_view.py` avec les reglages `aire max ilot`, `largeur max excroissance`, `largeur max entaille` et `lissage contour`.
- Refactorise `controllers/annotation_controller.py` pour centraliser tous les parametres de post-traitement ROI dans `_mask_post_process_kwargs()` puis les propager sans duplication vers preview, recompute ROI et apply volume.
- Etend `services/annotation_service.py` avec un pipeline `Mask cleanup` dedie aux ROI: trim des excroissances du masque `1`, suppression des petits ilots detaches, comblement optionnel des entailles fines du `0`, puis lissage optionnel du contour.
- Reordonne le pipeline pour appliquer le trim des excroissances avant la suppression des petites zones, afin que les fragments crees par le trim soient ensuite supprimes par la tolerance d aire.

**Contexte :**
Le besoin etait de faire evoluer le simple `Clean outliers` vers un vrai nettoyage de masque ROI plus visuel et plus controllable. L utilisateur voulait retirer les petits ilots, lisser les contours, supprimer les fines excroissances du masque lui-meme, tout en gardant separement un reglage optionnel pour combler des entailles etroits du fond lorsque souhaite.

**Decisions techniques :**
1. Garder le libelle visible `Mask cleanup` tout en conservant les noms internes `clean_outliers_*`, afin d eviter un refactor transversal inutile dans le code deja branche.
2. Separer explicitement le nettoyage du foreground et du background avec deux parametres distincts (`largeur max excroissance` pour le masque `1`, `largeur max entaille` pour le fond `0`), afin d eviter l ambiguite du precedent reglage unique.
3. Centraliser tout le post-traitement ROI dans `AnnotationService`, afin que les memes regles s appliquent partout sans divergence entre preview, recalcul ROI et propagation volume.
4. Placer la suppression des ilots apres le trim des excroissances, afin que les petits fragments detaches par ce trim puissent etre supprimes automatiquement par la tolerance d aire deja exposee.

### 2026-04-07 - Ecrasement des labels generalise par source avec destination configurable
**Tags :** `#branch:annotation`, `#MEMORY.md`, `#controllers/annotation_controller.py`, `#controllers/master_controller.py`, `#models/view_state_model.py`, `#views/nde_settings_view.py`, `#labels`, `#overwrite`, `#annotation`, `#ui`, `#mvc`

**Actions effectuees :**
- Generalise dans `models/view_state_model.py` la regle d ecrasement depuis un cas special `label 0` vers une table `label source -> destination autorisee`, tout en gardant un alias de compatibilite pour `label0_erase_target`.
- Remplace dans `views/nde_settings_view.py` le seul combo `Effacement label 0` par un editeur a deux listes `label source` et `autorise sur`, avec les modes `Aucun`, `Tous` et un label cible explicite.
- Refactorise `controllers/master_controller.py` pour synchroniser les sources et destinations disponibles, persister la regle du label selectionne et inclure `Background (0)` parmi les destinations autorisees.
- Refactorise `controllers/annotation_controller.py` pour resoudre partout un blocked mask generique par label source (paint, grow, line, free hand, box, recompute ROI, apply-volume) au lieu de brancher sur `if label == 0`.
- Corrige dans `controllers/annotation_controller.py` le cas `destination = 0` en n autorisant le `temp_mask` de fond qu aux pixels reellement couverts, afin d eviter que tout le fond implicite soit considere comme destination valide.

**Contexte :**
Le besoin etait d obtenir pour les labels non nuls le meme controle d ecrasement que celui deja present pour le `label 0`, puis d etendre ce controle pour autoriser aussi explicitement `Background (0)` comme destination. Le comportement devait rester coherent avec les previews ROI, le paint, les propagations volume et l architecture MVC existante.

**Decisions techniques :**
1. Stocker la politique d ecrasement dans `ViewStateModel` sous forme de mapping par label source, afin de supprimer les cas speciaux disperses et de rendre `label 0` equivalent aux autres labels.
2. Garder la resolution du blocked mask dans `AnnotationController` plutot que de pousser cette logique dans les vues ou dans le service, afin de rester dans le role d orchestration du controller sans etendre inutilement l API publique.
3. Inclure `Background (0)` dans les destinations mais pas dans les sources de dessin actives du `ToolsPanel`, afin de conserver la distinction existante entre action `Erase` et selection de label.
4. Intersecter le `temp_mask` de destination `0` avec `coverage`, afin que le fond implicite hors preview ne soit pas interprete a tort comme une zone explicitement autorisee.

### 2026-04-07 - Mask cleanup inverse pour ROI thresholdes en mode erase
**Tags :** `#branch:annotation`, `#MEMORY.md`, `#controllers/annotation_controller.py`, `#models/roi_model.py`, `#services/annotation_service.py`, `#erase`, `#mask-cleanup`, `#roi`, `#threshold`, `#mvc`

**Actions effectuees :**
- Etend `controllers/annotation_controller.py` pour activer un mode special seulement quand `action == erase`, `Force threshold (erase)` est coche et `Mask cleanup` est actif, puis propager ce mode sur les flux ROI `box`, `free hand`, `grow`, `line`, `peak`, ainsi que sur `apply volume` et `Recalculer ROI`.
- Ajoute dans `controllers/annotation_controller.py` des helpers pour reconstruire le masque effectif d une slice et en extraire le masque source a nettoyer avant conversion en `label 0`, en respectant les regles d overwrite deja generalisees.
- Etend `models/roi_model.py` avec un flag `erase_cleanup` persiste sur chaque ROI afin que les previews reconstruits et les recalculs ROI conservent le comportement inverse apres creation.
- Refactorise `services/annotation_service.py` pour accepter un `erase_cleanup_source_mask`, appliquer `Mask cleanup` sur le masque source existant selectionne par la ROI, puis generer seulement le delta `original - cleaned` comme masque d effacement.
- Verifie le flux par compilation `py_compile` et par sanity checks Python en memoire sur le chemin `ROI -> rebuild preview`, sans ajouter de script temporaire au depot.

**Contexte :**
Le besoin etait de faire fonctionner `Mask cleanup` comme un nettoyage inverse en mode `erase` pour les ROI basees sur le threshold. Au lieu d effacer toute la zone seuillee en `label 0`, il fallait n effacer que les petites composantes et excroissances que le pipeline de cleanup aurait supprimees, tout en laissant `draw` et le pinceau `erase` classiques inchanges.

**Decisions techniques :**
1. Activer ce comportement uniquement sous la combinaison `erase + force threshold (erase) + mask cleanup`, afin de limiter le scope aux ROI thresholdes et de ne pas changer le sens du mode `erase` standard.
2. Stocker un flag `erase_cleanup` dans `RoiModel`, afin que `Recalculer ROI` et les reconstructions volume rejouent exactement le meme comportement sans dependre seulement de l etat transitoire du controller.
3. Calculer le masque a effacer comme `selected_target - cleaned_target` dans `AnnotationService`, afin de reutiliser le pipeline `Mask cleanup` existant sans jamais rajouter de pixels en mode `erase`.
4. Reutiliser le masque effectif de slice resolu par le controller comme source de nettoyage, afin de respecter les masques deja appliques et les regles d overwrite ciblees lors de l effacement.

### 2026-04-08 - Rechargement du NPZ nnUNet via le pipeline overlay standard
**Tags :** `#branch:annotation`, `#MEMORY.md`, `#controllers/master_controller.py`, `#services/overlay_loader.py`, `#nnunet`, `#npz`, `#overlay`, `#pyqt6`, `#mvc`

**Actions effectuees :**
- Refactorise `controllers/master_controller.py` pour centraliser le chargement d un fichier overlay dans `_load_overlay_from_file()`, puis reutiliser ce helper a la fois pour `Charger un overlay` et pour la fin d inference nnUNet.
- Ajoute dans `controllers/master_controller.py` un pont `NnUnetUiSignals` base sur `QObject` et `pyqtSignal` afin de rerouter les callbacks nnUNet asynchrones vers le thread UI avant de pousser le NPZ sauvegarde dans `annotation_model` et les vues.
- Met a jour `controllers/master_controller.py` pour pre-remplir la boite de sauvegarde nnUNet avec un nom `NDE_MODELE.npz` derive du `.nde` courant et du modele choisi.
- Etend `services/overlay_loader.py` pour preferer la cle `mask` puis `arr_0` lors du chargement d un NPZ, afin de relire correctement les sorties nnUNet qui contiennent aussi `labels_mapping`.
- Verifie les changements par compilation ciblee `python -m py_compile controllers/master_controller.py services/overlay_loader.py services/nnunet_service.py`.

**Contexte :**
Le besoin etait que le resultat d inference nnUNet se comporte exactement comme un `Charger NPZ` manuel une fois le fichier enregistre, au lieu d injecter directement le masque en memoire avec un chemin special. Il fallait aussi eviter que le post-traitement UI dependa d un callback potentiellement hors thread principal.

**Decisions techniques :**
1. Passer par un helper unique `_load_overlay_from_file()` dans `MasterController`, afin que le chargement manuel et le rechargement post-nnUNet appliquent la meme sequence `overlay_loader -> annotation_model -> refresh_overlay`.
2. Utiliser un `QObject` avec `pyqtSignal` pour rapatrier la fin nnUNet sur le thread UI, plutot que de continuer a faire du push direct depuis le callback du plugin.
3. Lire en priorite la cle `mask` dans `OverlayLoader`, afin de rendre les NPZ nnUNet compatibles avec le chargeur overlay existant sans dependre de l ordre interne des cles du fichier.
4. Conserver la boite de sauvegarde nnUNet mais la pre-remplir a partir du NDE et du modele, afin d automatiser le nommage sans supprimer le controle utilisateur sur l emplacement final.

### 2026-04-10 - Tools panel labels dynamiques en deux colonnes
**Tags :** `#branch:annotation`, `#MEMORY.md`, `#controllers/master_controller.py`, `#views/tools_panel.py`, `#toolspanel.ui`, `#ui_toolspanel.py`, `#labels`, `#overlay`, `#ui`, `#pyqt6`, `#mvc`

**Actions effectuees :**
- Recompose `toolspanel.ui` pour reserver dans le panneau d outils une zone labels basee sur un `QScrollArea` avec deux colonnes dediees `frame_5` et `frame_6`, tout en reajustant la presentation visuelle des sections opacite/colormap et en regenerant `ui_toolspanel.py`.
- Etend `views/tools_panel.py` pour generer dynamiquement les labels a partir de la palette courante sous forme de radios dans `frame_5` et de boutons couleur dans `frame_6`, avec alignement des lignes, selection du label actif et edition de couleur via `QColorDialog`.
- Ajoute dans `views/tools_panel.py` le signal `label_color_changed` et la mise a jour locale des boutons couleur afin que le panneau principal puisse modifier la palette sans reconstruire toute l interface.
- Refactorise `controllers/master_controller.py` pour passer les deux conteneurs Designer au `ToolsPanel`, convertir la palette BGRA du modele en `QColor`, et centraliser la synchro des changements de couleur entre le tools panel, `OverlaySettingsView` et le modele d annotation.

**Contexte :**
Le besoin etait d obtenir dans le tools panel principal un rendu visuel en deux colonnes distinctes, avec les labels a gauche et les couleurs a droite, sans perdre la generation dynamique deja en place. L utilisateur voulait aussi pouvoir modifier la couleur depuis ce panneau comme dans les parametres d overlay, tout en s appuyant sur le nouveau layout Designer plutot que sur une ligne de widgets generes dans un conteneur unique.

**Decisions techniques :**
1. Conserver la generation dynamique dans `ToolsPanel`, mais l injecter dans les conteneurs Designer `frame_5` et `frame_6`, afin de garder un nombre de labels arbitraire sans figer l interface a quatre widgets statiques.
2. Faire transiter tous les changements de couleur par `MasterController` via `_on_label_color_changed`, afin de synchroniser en un seul point le tools panel, `OverlaySettingsView`, la palette du modele et les previews ROI.
3. Convertir explicitement les couleurs BGRA en `QColor` au moment de la synchro des labels, afin de laisser `ToolsPanel` purement focalise sur l UI PyQt et d eviter de dupliquer cette conversion dans la vue.
4. Fixer une largeur et une hauteur homogenes aux widgets generes d une meme passe de rendu, afin que les deux colonnes restent visuellement alignees malgre des libelles de labels de longueurs variables.

### 2026-04-11 - Affichage outline only des overlays 2D
**Tags :** `#branch:annotation`, `#MEMORY.md`, `#controllers/annotation_controller.py`, `#controllers/master_controller.py`, `#models/view_state_model.py`, `#ui_mainwindow.py`, `#untitled.ui`, `#views/annotation_view.py`, `#views/endview_view.py`, `#overlay`, `#outline`, `#ui`, `#pyqt6`, `#mvc`

**Actions effectuees :**
- Ajoute dans `ui_mainwindow.py` et `untitled.ui` l action menu `Affichage > Toggle outline only`.
- Etend `models/view_state_model.py` avec un etat UI `show_outline_only` pour piloter ce mode d affichage.
- Refactorise `controllers/master_controller.py` pour rendre l action cochable, synchroniser son etat avec le modele et propager le toggle vers le pipeline d annotation.
- Etend `controllers/annotation_controller.py` avec `set_outline_only()` afin de pousser le mode `outline only` vers les vues 2D principales et secondaires sans toucher au pipeline metier de post-traitement.
- Etend `views/endview_view.py` avec un rendu non destructif de contour 4-connexe qui conserve les ids de labels, puis l applique au rendu des overlays 2D.
- Etend `views/annotation_view.py` pour reutiliser ce rendu contour sur les previews ROI temporaires et maintenir la coherence visuelle entre overlay importe et preview de masque.

**Contexte :**
L utilisateur avait ajoute l option `Affichage > Toggle outline only` et voulait afficher uniquement le contour des masques quand ce toggle est actif. Le code de post-traitement contenait deja une logique de contour pour le lissage, mais pas de sortie d affichage directement reutilisable. Il fallait donc introduire un mode de rendu dedie, strictement cote UI, sans melanger logique metier et vue.

**Decisions techniques :**
1. Ne pas reutiliser directement le contour du post-traitement ROI, car il sert a lisser puis rerasterizer un masque plein et non a produire une representation d affichage stable.
2. Stocker le toggle dans `ViewStateModel`, afin que l etat reste centralise et synchronisable entre menu, controllers et restauration de session.
3. Implementer l extraction de contour dans `EndviewView`, afin de traiter `outline only` comme une option de rendu 2D non destructive, independante des donnees source et du pipeline d annotation.
4. Conserver les ids de labels sur les pixels de bord, afin de reutiliser la palette existante sans introduire de mapping special pour le mode contour.

### 2026-04-13 - Overlay des masques sur l A-scan avec toggle menu
**Tags :** `#branch:annotation`, `#MEMORY.md`, `#controllers/annotation_controller.py`, `#controllers/ascan_controller.py`, `#controllers/master_controller.py`, `#models/view_state_model.py`, `#services/ascan_service.py`, `#ui_mainwindow.py`, `#untitled.ui`, `#views/ascan_view.py`, `#ascan`, `#overlay`, `#pyqtgraph`, `#ui`, `#mvc`

**Actions effectuees :**
- Etend `models/view_state_model.py` avec un etat UI `show_overlay_ascan` afin de piloter independamment l affichage des overlays dans l A-scan.
- Branche dans `controllers/master_controller.py` l action menu `Affichage > Toggle overlay ascan`, synchronise son etat avec le modele et propage aussi l opacite globale d overlay vers les vues A-scan.
- Refactorise `services/ascan_service.py` pour projeter le `mask_volume` courant sur le profil A-scan actif et produire des intervalles continus `(start, end, label)` limites aux labels visibles.
- Etend `controllers/ascan_controller.py` pour injecter `mask_volume`, palette et visibilite des labels dans le calcul du profil, puis pousser les segments projetes vers les vues standard et corrosion.
- Etend `views/ascan_view.py` avec un rendu PyQtGraph base sur `BarGraphItem` afin d afficher des zones rectangulaires semi-transparentes derriere la courbe A-scan pour chaque section segmentee.
- Ajoute dans `controllers/annotation_controller.py` un callback de rafraichissement afin que l A-scan soit recalcule automatiquement apres import overlay, changement de visibilite/couleur ou toggle d overlay.
- Verifie l implementation par compilation ciblee `python -m compileall` et par smoke tests Python sur la generation des spans et l instanciation `offscreen` de `AScanView`.

**Contexte :**
L utilisateur voulait voir directement sur l A-scan quelles portions du profil courant correspondent a des zones segmentees, sous forme de rectangles colores. Il avait aussi ajoute manuellement l entree menu `Toggle overlay ascan`, qu il fallait rendre reellement fonctionnelle sans introduire un calcul global couteux sur tous les A-scans du volume.

**Decisions techniques :**
1. Calculer l overlay uniquement pour le profil A-scan courant, afin de garder un cout lineaire sur la longueur du signal affiche plutot que sur tout le volume.
2. Produire des spans continus par label dans `AScanService`, afin de fusionner les pixels contigus et d eviter un rendu ou une allocation par echantillon.
3. Garder le rendu des rectangles strictement dans `AScanView`, afin de preserver la separation MVC entre calcul metier, orchestration controller et dessin UI.
4. Faire dependre l overlay A-scan de `show_overlay`, `show_overlay_ascan`, de la visibilite des labels et de l opacite globale, afin de conserver un comportement coherent avec le pipeline d overlay deja existant.

### 2026-04-15 - Pipeline corrosion decouple : donnees brutes puis interpolation a la demande
**Tags :** `#branch:annotation`, `#services/cscan_corrosion_service.py`, `#controllers/master_controller.py`, `#views/tools_panel.py`, `#models/view_state_model.py`, `#ui_toolspanel.py`, `#toolspanel.ui`, `#corrosion`, `#interpolation`, `#pipeline`, `#mvc`

**Actions effectuees :**
- Modifie `run_analysis()` dans `CScanCorrosionService` pour ne plus interpoler automatiquement les peak maps. Les donnees brutes (distance_map, peak maps, overlay) sont retournees directement.
- Ajout de `raw_peak_index_map_a` / `raw_peak_index_map_b` dans `CorrosionAnalysisResult` et `CorrosionWorkflowResult` pour conserver les peak maps originaux.
- Ajout du champ `mask_height` dans `CorrosionWorkflowResult` pour permettre la re-interpolation sans recalcul du volume masque.
- Nouvelle methode `apply_interpolation(raw_result, algo, mask_height, ...)` dans `CScanCorrosionService` : accepte `"brut"` ou `"1d_dual_axis"`, reconstruit peak maps + distance map + overlay + piece 3D.
- Nouvelle methode `run_interpolation(raw_result, algo, nde_model)` dans `CorrosionWorkflowService` : orchestre l interpolation a la demande sur un resultat brut existant.
- Ajout de `corrosion_raw_peak_index_map_a/b` dans `ViewStateModel` avec reset dans `deactivate_corrosion()`.
- Ajout du signal `corrosion_interpolation_requested(str)` dans `ToolsPanel`, avec mapping `_INTERP_ALGO_BY_TEXT` (`"brut"` → `"brut"`, `"1d dual-axis"` → `"1d_dual_axis"`).
- Wiring de `comboBox_4` (algorithme) et `pushButton` (Calculer) dans `attach_designer_widgets()` du ToolsPanel.
- Reecrit `_on_corrosion_completed()` dans `MasterController` pour afficher les donnees brutes sans interpolation et stocker le resultat dans `_raw_corrosion_workflow_result`.
- Nouveau handler `_on_corrosion_interpolation_requested(algo)` dans `MasterController` : appelle `run_interpolation()` puis rafraichit toutes les vues (C-scan, Endview, overlay, 3D piece).

**Contexte :**
L utilisateur souhaitait pouvoir comparer facilement differents algorithmes d interpolation sur les profils de corrosion. Auparavant, l interpolation 1D dual-axis etait appliquee automatiquement lors de l analyse, rendant impossible la visualisation des donnees brutes ou la comparaison entre algorithmes. Le nouveau workflow affiche d abord les donnees brutes, puis permet de lancer un algorithme choisi via le combobox + bouton Calculer du ToolsPanel. A terme, d autres algorithmes (scipy.interpolate) seront ajoutes au combobox.

**Decisions techniques :**
1. Separer le calcul brut de l interpolation dans `CScanCorrosionService` via une methode dediee `apply_interpolation()`, pour ne pas dupliquer le pipeline complet a chaque nouvel algorithme.
2. Stocker les peak maps bruts dans le resultat (`raw_peak_index_map_a/b`) plutot que de re-executer `run_analysis()` a chaque changement d algorithme.
3. Garder le signal `corrosion_interpolation_requested(str)` dans la View (`ToolsPanel`) et le handler dans le Controller (`MasterController`) pour respecter MVC.
4. Le combobox et le bouton dans `toolspanel.ui` (frame_12) sont connectes via `attach_designer_widgets` avec des parametres optionnels pour rester retrocompatible.

### 2026-04-15 - Masques overlay absents de la vue A-scan corrosion
**Tags :** `#branch:annotation`, `#controllers/ascan_controller.py`, `#ascan`, `#corrosion`, `#overlay`, `#mvc`

**Actions effectuees :**
- Modifie `controllers/ascan_controller.py` dans la boucle `update_trace` pour ne passer que `[]` a `corrosion_view.set_overlay_segments()` quand `corrosion_active` est vrai, au lieu de `profile.overlay_spans`.

**Contexte :**
Apres l activation du pipeline corrosion avec interpolation, des rectangles de spans de masque restaient visibles sur la vue A-scan corrosion. Le profil corrosion est una simple ligne FW/BW — il ne peut avoir qu une seule valeur y par x — donc les spans de masque de segmentation n y ont pas de sens.

**Decisions techniques :**
1. Conserver le calcul de `overlay_spans` dans `AScanService` tel quel et filtrer uniquement a l injection dans la vue corrosion, afin de ne pas changer le comportement de la vue standard ni le pipeline de calcul existant.

### 2026-04-17 - Piece 3D integree au dock Volume et commit corrosion aligne sur l algo courant
**Tags :** `#branch:annotation`, `#controllers/corrosion_profile_controller.py`, `#controllers/dock_layout_controller.py`, `#controllers/master_controller.py`, `#models/view_state_model.py`, `#services/corrosion_profile_edit_service.py`, `#services/cscan_corrosion_service.py`, `#views/piece3d_view.py`, `#views/tools_panel.py`, `#corrosion`, `#piece3d`, `#volume`, `#interpolation`, `#mvc`, `#stack`

**Actions effectuees :**
- Ajoute `corrosion_interpolation_algo` dans `ViewStateModel` et le wiring associe dans `ToolsPanel` et `MasterController` pour conserver l algo corrosion actuellement selectionne dans l UI.
- Remplace le `1d_dual_axis` hardcode du commit du profil corrosion par un passage via `interpolate_peak_map_with_algo(...)` dans `CScanCorrosionService`, appele depuis `CorrosionProfileEditService` et `CorrosionProfileController`.
- Refactorise `DockLayoutController` pour construire un `volume_stack` contenant `VolumeView` standard et une page corrosion avec `Piece3DView` + bouton de bascule brut/interpole.
- Supprime dans `MasterController` l ouverture de la fenetre flottante `QDialog` pour la piece 3D, et fait basculer l action `Afficher solide 3d` entre la vue Volume standard et la vue corrosion embarquee.
- Ajoute le reset explicite de l etat 3D corrosion lors des changements de dataset, de session et d overlay global pour revenir a la vue Volume standard et eviter l affichage de donnees stale.
- Ajuste `Piece3DView` pour masquer le slider secondaire et vider proprement la vue quand aucun volume corrosion n est disponible.

**Contexte :**
L utilisateur voulait conserver le comportement actuel du commit du profil corrosion, mais selon l algorithme choisi dans le combo plutot qu avec `1d_dual_axis` par defaut. En parallele, la piece 3D corrosion ne devait plus s ouvrir automatiquement dans une fenetre libre, et devait suivre le meme principe de remplacement de vue que les stacks deja utilises pour les vues A-scan, C-scan et Endview corrosion.

**Decisions techniques :**
1. Centraliser le choix de l algorithme de commit dans `CScanCorrosionService` avec `interpolate_peak_map_with_algo(...)`, afin que le commit du profil et le recalcul d interpolation partagent le meme dispatch.
2. Stocker l algo courant dans `ViewStateModel` et le synchroniser depuis `ToolsPanel`, afin de rester dans un flux MVC sans lecture directe de widgets depuis les services.
3. Integrer `Piece3DView` dans le dock `Volume` via un `QStackedLayout`, plutot que de remplacer l instance `volume_view` ou de maintenir un `QDialog` flottant, pour rester coherent avec l architecture corrosion existante.
4. Conserver l affichage de la piece 3D comme action manuelle via `Afficher solide 3d`, tout en mettant a jour les donnees en cache apres les calculs corrosion sans changement visuel implicite.

### 2026-04-17 - Nouveaux algorithmes d interpolation corrosion et garde-fous 2D
**Tags :** `#branch:annotation`, `#controllers/corrosion_profile_controller.py`, `#services/cscan_corrosion_service.py`, `#toolspanel.ui`, `#ui_toolspanel.py`, `#views/tools_panel.py`, `#corrosion`, `#interpolation`, `#scipy`, `#ui`, `#mvc`

**Actions effectuees :**
- Etend `CScanCorrosionService` avec les dispatchs `1d_pchip_dual_axis`, `1d_makima_dual_axis`, `2d_linear_nd`, `2d_clough_tocher`, `2d_rbf_thin_plate` et `2d_gaussian_fill`, plus des helpers communs pour le clipping, l interpolation 1D et la preparation des donnees 2D.
- Ajoute les constantes `RBF_MAX_POINTS`, `RBF_NEIGHBORS` et `GAUSSIAN_FILL_SIGMA` pour borner le cout du thin-plate spline et parametrer le remplissage nearest + lissage gaussien.
- Met a jour `views/tools_panel.py`, `toolspanel.ui` et `ui_toolspanel.py` pour exposer les nouvelles options du combo corrosion et leurs mappings textuels.
- Protege `CorrosionProfileController.commit_pending_edits()` pour convertir une erreur d interpolation 2D en message utilisateur plutot qu en exception non geree.

**Contexte :**
L utilisateur voulait comparer plusieurs algorithmes d interpolation supplementaires dans le flux corrosion au-dela du `1d_dual_axis` existant, avec une priorite sur les variantes 1D shape-preserving puis des options 2D plus lisses ou diagnostiques. Les nouveaux algos devaient rester compatibles avec le pipeline d interpolation a la demande et avec le commit de profil corrosion deja aligne sur l algo selectionne dans l UI.

**Decisions techniques :**
1. Reutiliser `interpolate_peak_map_with_algo(...)` comme point de dispatch unique pour le recalcul a la demande et le commit de profil, afin de garder un seul contrat d interpolation dans le service corrosion.
2. Conserver la regle actuelle "ne remplir que les trous `-1` autorises par le `support_map`" et ne jamais modifier les points deja mesures, pour comparer les algos sans changer la semantique metier du pipeline.
3. Encadrer `RBFInterpolator(kernel="thin_plate_spline")` par un sous-echantillonnage (`RBF_MAX_POINTS`) et un voisinage limite (`RBF_NEIGHBORS`) pour eviter un cout memoire et CPU excessif sur des peak maps denses.
4. Implementer `2d_gaussian_fill` comme `griddata(..., method="nearest")` suivi de `gaussian_filter`, en le positionnant comme algo de comparaison visuelle plutot que comme reconstruction physique stricte.
5. Intercepter les `ValueError` dans `CorrosionProfileController` pour remonter une erreur utilisateur propre quand une interpolation 2D est impossible sur une geometrie de points degeneree.

### 2026-04-20 - Workflow corrosion par sessions, modes par label et restauration du solide 3D
**Tags :** `#branch:annotation`, `#config/constants.py`, `#controllers/corrosion_profile_controller.py`, `#controllers/cscan_controller.py`, `#controllers/master_controller.py`, `#models/view_state_model.py`, `#services/annotation_session_manager.py`, `#services/corrosion_profile_edit_service.py`, `#services/cscan_corrosion_service.py`, `#services/distance_measurement.py`, `#services/peak_plateau.py`, `#toolspanel.ui`, `#ui_mainwindow.py`, `#ui_toolspanel.py`, `#untitled.ui`, `#views/corrosion_settings_view.py`, `#views/tools_panel.py`, `#corrosion`, `#interpolation`, `#session`, `#piece3d`, `#menu`, `#ui`, `#mvc`

**Actions effectuees :**
- Ajoute dans `config/constants.py` les normaliseurs communs pour les modes de pic corrosion et les algos d interpolation, ainsi que les etats `base`, `raw` et `interpolated` du workflow corrosion.
- Etend `services/peak_plateau.py`, `services/distance_measurement.py`, `services/cscan_corrosion_service.py` et `controllers/cscan_controller.py` pour supporter un choix de mode de pic distinct par label (`optimistic`, `pessimistic`, `max_peak`) au lieu d un seul mode global.
- Refactorise `models/view_state_model.py`, `services/annotation_session_manager.py` et `controllers/master_controller.py` pour gerer un workflow corrosion en trois etats de session : session de base, session brute issue de `Analyze`, puis session interpolee issue de `Interpolate`.
- Deplace le pilotage corrosion du `ToolsPanel` vers le menu `Analyse` : `Analyze` remplace l ancienne action de menu, `Interpolate` devient une action dediee, et les branchements / widgets corrosion sont retires de `views/tools_panel.py`, `toolspanel.ui` et `ui_toolspanel.py`.
- Etend `views/corrosion_settings_view.py` en fenetre de parametres a 5 combobox : `Label A`, `Mode label A`, `Label B`, `Mode label B`, `Interpolation`, avec verrouillage des controles selon l etat de session.
- Fait creer par `MasterController` une nouvelle session brute apres `Analyze`, puis une nouvelle session interpolee apres `Interpolate`, tout en interdisant les transitions invalides directement depuis les actions du menu.
- Retire `brut` de la liste des interpolations UI et aligne `services/corrosion_profile_edit_service.py` et `controllers/corrosion_profile_controller.py` pour que le commit du profil corrosion utilise le flux de session courant plutot qu un pseudo type d interpolation brut.
- Rend persistants dans le `ViewStateModel` les volumes du solide 3D corrosion, leur ancre et le choix brut/interpole, puis les restaure sur changement de session dans `MasterController` afin de reparer l affichage du dock 3D.
- Recalcule aussi les volumes du solide 3D lors d un `Apply ROI` sur le profil corrosion, pour que les edits de profil restent coherents avec la session brute ou interpolee active.

**Contexte :**
Le flux corrosion avait accumule plusieurs ambiguities : le calcul brut et l interpolation etaient confondus dans l UI, l action historique `Corrosion analyse` du menu doublonnait avec les boutons du `ToolsPanel`, et la selection du pic A-scan ne permettait pas de differencier le comportement selon chaque label BW/FW. En parallele, le solide 3D corrosion etait maintenu uniquement dans un cache runtime du controller, ce qui cassait son affichage lors des changements de session. Le lot staged vise a clarifier le pipeline complet : `Analyze` extrait et projette les donnees brutes avec des modes de pic configurables par label, `Interpolate` cree une nouvelle session derivee a partir de cette base brute, et la piece 3D suit maintenant correctement la vie des sessions corrosion.

**Decisions techniques :**
1. Modeliser explicitement les etats `base`, `raw` et `interpolated` dans le `ViewStateModel`, puis piloter l activation des actions `Analyze` / `Interpolate` depuis `MasterController`, afin d eviter les transitions invalides et de rendre le workflow lisible.
2. Associer les strategies `optimistic`, `pessimistic` et `max_peak` directement aux labels choisis (`Label A` / `Label B`) plutot qu a une notion implicite de haut/bas, pour garder un contrat stable entre UI, controller et pipeline de mesure.
3. Sortir toute la configuration corrosion du `ToolsPanel` et la centraliser dans le menu `Analyse` et la fenetre `CorrosionSettingsView`, afin de supprimer le doublon menu/panneau et de nettoyer le code mort genere par l ancienne UI.
4. Conserver l interpolation comme une transformation derivee d une session brute existante, en creant une nouvelle session plutot qu en ecrasant la session source, pour permettre la comparaison et preserver un historique exploitable.
5. Sauvegarder les sources du solide 3D corrosion et leur ancre dans l etat de session plutot que dans un cache volatil du controller, car la restauration a la volee depuis la seule projection 2D ne couvrait pas correctement tous les cas de bascule et d edition.

### 2026-04-21 - Outil Prune d annotation et parametres dedies
**Tags :** `#branch:annotation`, `#controllers/annotation_controller.py`, `#controllers/master_controller.py`, `#models/view_state_model.py`, `#services/annotation_service.py`, `#views/annotation_view.py`, `#views/nde_settings_view.py`, `#views/tools_panel.py`, `#toolspanel.ui`, `#ui_toolspanel.py`, `#prune`, `#annotation`, `#ui`, `#session`, `#mvc`

**Actions effectuees :**
- Ajoute le mode outil `Prune` dans le `ToolsPanel` Designer/codegen et le mappe dans `views/tools_panel.py` pour qu il soit selectionnable comme outil d annotation.
- Etend `views/annotation_view.py` et `controllers/annotation_controller.py` pour reutiliser le rectangle existant, construire une preview temp sur la slice ou sur une plage `Apply volume`, puis appliquer le prune sur le label actif.
- Ajoute dans `services/annotation_service.py` la logique `prune_disconnected_label_bands()` qui detecte, colonne par colonne dans la box, les bandes disconnectees d un label et supprime les runs non desires.
- Reutilise la logique de selection `max_peak` / `optimistic` / `pessimistic` avec reference eventuelle a un label compagnon, tout en calculant la reference sur la slice effective complete plutot que de la tronquer a la box.
- Cree dans `views/nde_settings_view.py` et `models/view_state_model.py` des parametres d annotation dedies a `Prune` (`label A`, `label B`, `mode`) separes des reglages d analyse corrosion.
- Synchronise ces nouveaux parametres dans `controllers/master_controller.py`, y compris la normalisation du couple de labels, la restauration apres switch de session et la resynchronisation lors d ajout/suppression de labels.

**Contexte :**
Le besoin etait d ajouter un outil d annotation permettant d effacer selectivement une bande parasite quand un meme label apparait en plusieurs bandes disconnectees dans une box. En parallele, les choix de labels et de mode utilises par `Prune` ne devaient plus dependre des parametres d analyse corrosion, afin de separer clairement le workflow d annotation du workflow corrosion.

**Decisions techniques :**
1. Garder `Prune` comme un outil de rectangle autonome dans le pipeline d annotation existant, afin de reutiliser la preview temp, `Apply` et `Apply volume` sans introduire un nouveau flux UI.
2. Centraliser la logique de suppression selective dans `AnnotationService` plutot que dans la vue ou le controller, pour rester conforme a l architecture MVC et rendre l algorithme testable.
3. Introduire `prune_label_a`, `prune_label_b` et `prune_peak_selection_mode` dans `ViewStateModel` au lieu de reutiliser `corrosion_label_a/b`, afin d isoler l annotation des reglages metier corrosion.
4. Normaliser le couple de labels prune avec `CorrosionLabelService.normalize_pair(...)`, pour partager les memes regles de validation de couple sans dupliquer la logique dans le controller.
5. Conserver un fallback FW/BW seulement quand aucun couple prune explicite n est configure, pour preserver le comportement historique tout en donnant priorite aux nouveaux parametres d annotation.

### 2026-04-21 - Toggle plans volume et persistance du solide 3D corrosion
**Tags :** `#branch:annotation`, `#controllers/master_controller.py`, `#models/view_state_model.py`, `#services/annotation_session_manager.py`, `#ui_mainwindow.py`, `#untitled.ui`, `#views/volume_view.py`, `#piece3d`, `#volume`, `#colormap`, `#session`, `#ui`, `#mvc`

**Actions effectuees :**
- Ajoute l action menu `Toggle plans volume` dans `ui_mainwindow.py` et `untitled.ui`, puis la connecte dans `MasterController` comme un toggle d affichage standard.
- Introduit `show_volume_planes` dans `ViewStateModel` et applique cet etat a `VolumeView` pour afficher ou masquer uniquement les plans mobiles et leurs contours, sans cacher les sliders UI.
- Ajoute dans `VolumeView` la synchronisation `_apply_volume_plane_visibility()` afin de reappliquer la visibilite des plans apres chaque reconstruction de scene 3D.
- Centralise dans `MasterController` la reapplication des colormaps sauvegardees avec `_apply_saved_colormaps()`, en resolvant la LUT au lieu de repasser `None`, pour eviter le fallback gris apres changement de session.
- Fait afficher automatiquement la vue du solide 3D corrosion apres creation des sessions `raw` et `interpolated`, quand des volumes piece sont disponibles.
- Introduit `corrosion_piece_view_enabled` dans `ViewStateModel` et `AnnotationSessionManager`, puis l utilise dans `MasterController` pour memoriser par session si la vue solide 3D etait laissee ouverte.

**Contexte :**
Le workflow corrosion gardait deux incoherences visibles dans l UI. D une part, le changement de session reappliquait les colormaps sans LUT resolue, ce qui faisait retomber Endview, Volume et C-scan en gris. D autre part, revenir sur une session de base sans solide 3D decochait implicitement l action `Afficher solide 3d`, si bien qu un retour sur une session `raw` ou `interpolated` ne restaurait plus l ouverture de la vue 3D dans l etat laisse par l utilisateur. En parallele, l utilisateur voulait pouvoir masquer les plans mobiles de la vue volume depuis le menu `Affichage`, sans toucher aux sliders de navigation.

**Decisions techniques :**
1. Stocker `show_volume_planes` dans `ViewStateModel` plutot que dans `VolumeView`, afin que le toggle du menu `Affichage` suive les changements de session comme le reste des etats UI.
2. Limiter `Toggle plans volume` aux plans mobiles et a leurs contours dans `VolumeView`, sans masquer les sliders, pour respecter le comportement demande et garder les controles de navigation toujours visibles.
3. Centraliser la restauration des colormaps dans `_apply_saved_colormaps()` au niveau de `MasterController`, afin de recalculer explicitement les LUT OmniScan/Gris avant de pousser l etat dans les vues et d eliminer le fallback implicite sur `None`.
4. Memoriser l ouverture de la vue solide 3D avec `corrosion_piece_view_enabled` dans l etat de session, afin de ne plus confondre "session sans donnees 3D" et "utilisateur a volontairement ferme la vue".
5. Continuer d ouvrir automatiquement la vue solide 3D apres `Analyze` et `Interpolate`, mais persister cette ouverture dans la session nouvellement creee pour que les allers-retours `base -> raw/interpolated` restaurent le meme etat visuel.

### 2026-04-30 - Export overlay multi-format et orientation explicite U/V
**Tags :** `#branch:annotation`, `#MEMORY.md`, `#controllers/annotation_controller.py`, `#controllers/master_controller.py`, `#services/overlay_export.py`, `#services/overlay_loader.py`, `#views/overlay_export_dialog.py`, `#overlay`, `#import`, `#export`, `#npz`, `#sentinel`, `#orientation`, `#ui`, `#mvc`

**Actions effectuees :**
- Etend `views/overlay_export_dialog.py` avec un choix de cible `Normal` ou `Sentinel`, puis expose les parametres de transformation Sentinel (rotation, axes, transpose, suffixe, miroirs, mode strict) dans une section UI dediee activee uniquement pour ce mode.
- Refactorise `services/overlay_export.py` pour separer l export applicatif standard et l export de compatibilite Sentinel, avec validation commune du volume, normalisation du chemin de sortie et pipeline de transformations `transpose -> rotate -> mirrors`.
- Fait ecrire par l export standard un NPZ versionne contenant `mask_ucoord` et `mask_vcoord`, afin de persister explicitement les deux orientations d annotation au lieu d un unique `arr_0`.
- Etend `services/overlay_loader.py` pour detecter ce nouveau format NPZ, choisir automatiquement `mask_ucoord` ou `mask_vcoord` selon l axe primaire courant, puis conserver un fallback sur les anciens fichiers `mask` ou `arr_0`.
- Propage dans `controllers/master_controller.py` et `controllers/annotation_controller.py` le `primary_axis_name` derive des metadonnees NDE, aussi bien lors de l import que lors de la sauvegarde, afin d aligner le format exporte et la relecture sur l orientation active.

**Contexte :**
Le besoin etait de supporter deux usages distincts pour les overlays. D un cote, l application devait sauvegarder un format plus robuste pour ses propres reimports, sans ambiguite sur l orientation U/V. De l autre, l utilisateur avait besoin d un export Sentinel configurable avec une chaine de transformations explicite pour s adapter a des conventions externes de rotation, transpose et miroir.

**Decisions techniques :**
1. Stocker explicitement `mask_ucoord` et `mask_vcoord` dans le format applicatif, afin d eliminer la dependance a des heuristiques de shape ou a des transposes implicites lors des reimports.
2. Garder la logique de transformation Sentinel dans `OverlayExport`, plutot que dans la vue ou le controller, pour respecter MVC et centraliser la validation des permutations, axes de rotation et miroirs.
3. Faire transiter l axe primaire courant via `MasterController` et `AnnotationController`, afin que l import/export reste coherent avec les metadonnees `axis_order` du dataset actif sans acces direct des services a l UI.
4. Conserver un fallback de lecture sur les anciens NPZ `mask` et `arr_0`, afin de ne pas casser la compatibilite avec les overlays historiques deja produits.
