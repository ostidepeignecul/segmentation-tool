# 🤖 AGENTS.md — Règles d'Exécution

> **Objectif :** Définir les règles que l'agent doit suivre avant chaque exécution pour maintenir la cohérence du projet.

---

## 🎯 Principes Fondamentaux

### 1. **Memory-First Development**
L'agent doit **toujours** consulter la mémoire avant toute action, selon une **stratégie hybride ByteRover + MEMORY.md** :

#### 📖 Stratégie de Lecture Hybride (ByteRover → MEMORY.md)

**🔹 PRIORITÉ 1 : ByteRover (Toujours en premier)**

L'agent doit **TOUJOURS** commencer par interroger ByteRover avec `byterover-retrieve-knowledge` :

```python
# Exemple de requête ByteRover
byterover-retrieve-knowledge(
    query="Comment fonctionne l'annotation des polygones dans mask_editor?",
    limit=5
)
```

**Niveau 1 - Recherche Ciblée ByteRover (par défaut)** :
- Formuler une **question précise** liée à la tâche (ex: "Où est géré le threshold dans mask_editor?")
- Utiliser **tags spécifiques** si connus (`#annotation_controller.py`, `#threshold`, `#mvc`)
- Limiter à **3-5 résultats** les plus pertinents
- Identifier rapidement les patterns et décisions récentes

**Niveau 2 - Recherche Contextuelle ByteRover** :
- Élargir la requête aux **concepts architecturaux** (`#architecture`, `#mvc`, `#3d-visualization`)
- Chercher les **entrées des 30 derniers jours** avec queries temporelles
- Combiner avec `git log` pour voir l'historique des fichiers
- Augmenter limite à **5-10 résultats** si nécessaire

**Niveau 3 - Recherche Exhaustive ByteRover** :
- Requêtes multiples sur différents aspects du projet
- Recherche par **tous les tags majeurs** (`#performance`, `#overlay`, `#nde_loader`)
- Limite étendue à **10+ résultats**
- Utilisé pour modifications architecturales majeures

**🔹 FALLBACK : MEMORY.md (Si ByteRover indisponible)**

**SI et SEULEMENT SI** ByteRover ne répond pas ou n'est pas connecté, utiliser `MEMORY.md` :

**Niveau 1 - Lecture Ciblée** :
- Lire les **50-100 premières lignes** de `MEMORY.md` (en-tête + entrées récentes)
- Utiliser `grep` pour rechercher les **tags pertinents** (`#fichier.py`, `#concept`)
- Identifier rapidement le contexte des fichiers concernés

**Niveau 2 - Lecture Contextuelle** :
- Lire sections liées aux **tags architecturaux** (`#architecture`, `#mvc`)
- Consulter entrées des **30 derniers jours**
- Vérifier `git log` pour historique fichiers
- Lire sections spécifiques trouvées au niveau 1

**Niveau 3 - Lecture Complète** :
- Lire `MEMORY.md` **en entier** si :
  - Modifications architecturales majeures
  - Aucun contexte pertinent trouvé aux niveaux 1-2
  - Demande explicite de l'utilisateur

**Objectifs communs** :
- Comprendre le contexte historique du projet
- Identifier les patterns et décisions passées
- Éviter de répéter des erreurs
- Maintenir cohérence entre ByteRover et MEMORY.md

### 2. **Plan Before Action**
**OBLIGATOIRE** : Avant toute modification significative :
1. **Créer un plan détaillé** avec des étapes bien définies
2. **Présenter le plan à l'utilisateur**
3. **Attendre l'approbation explicite** avant de commencer
4. **Ne jamais modifier sans confirmation**

Si la tâche comporte plusieurs points :
- Découper en étapes claires et numérotées
- Expliquer l'impact de chaque étape
- Demander validation avant de procéder

### 3. **Context-Aware Execution**
Chaque action doit être informée par :
- L'architecture établie (MVC)
- Les conventions de code du projet
- Les politiques de documentation
- Les dépendances autorisées

#### 🔍 Consultation Git pour Contexte Supplémentaire

Avant toute modification de code, **vérifier Git** pour comprendre l'historique :

**Commandes essentielles :**
```bash
# Voir les 10 derniers commits du projet
git log --oneline -n 10

# Voir l'historique d'un fichier spécifique
git log --oneline -- path/to/file.py

# Voir les modifications récentes (dernier commit)
git show --stat

# Voir les fichiers modifiés récemment
git log --name-only --pretty=format: -n 5 | sort -u
```

**Workflow Git + ByteRover + MEMORY.md :**
1. Identifier les fichiers à modifier dans la tâche
2. Vérifier `git log` pour ces fichiers
3. Interroger **ByteRover** avec query ciblée sur ces fichiers
4. Si ByteRover indisponible → Comparer avec entrées `MEMORY.md` correspondantes
5. Si divergence → lire sections complètes (ByteRover ou MEMORY.md)

**Pourquoi Git + ByteRover + MEMORY.md ?**
- Git : Historique factuel des commits
- ByteRover : Recherche sémantique intelligente, contexte récent accessible rapidement
- MEMORY.md : Backup local, lecture séquentielle complète si besoin
- Ensemble : Vision complète et rapide de l'évolution du code

### 4. **Controlled Documentation (Double Storage)**
**RÈGLE CRITIQUE** : **JAMAIS** documenter dans ByteRover ou MEMORY.md sans l'ordre **EXPLICITE** de l'utilisateur.

**Quand documenter :**
- **UNIQUEMENT** : Lorsque l'utilisateur le demande explicitement (ex: "store ceci", "documente ça").
- **INTERDIT** : De documenter automatiquement à la fin d'une tâche.
- **INTERDIT** : De documenter sans ordre direct.

**Exceptions (ne PAS documenter) :**
- Tâche en cours, non terminée
- Modifications temporaires ou expérimentales
- Tests en cours de développement

**🔹 STRATÉGIE DE DOUBLE STORAGE (ByteRover + MEMORY.md)**

**TOUJOURS** documenter dans les **DEUX** systèmes en **ordre chronologique** :

**1. ByteRover (En premier)** :
```python
byterover-store-knowledge(
    messages="""
    # YYYY-MM-DD — Titre de la modification
    
    **Tags:** `#fichier.py`, `#concept`, `#technologie`
    
    **Actions effectuées:**
    - Action 1 : Description précise
    - Action 2 : Description précise
    
    **Contexte:**
    Explication technique détaillée...
    
    **Décisions techniques:**
    1. Décision 1 et justification
    2. Décision 2 et justification
    """
)
```

**2. MEMORY.md (Immédiatement après)** :
```markdown
### **YYYY-MM-DD** — [Titre identique]
**Tags :** `#fichier.py`, `#concept`, `#technologie`

**Actions effectuées :**
- Action 1 : Description précise [IDENTIQUE à ByteRover]
- Action 2 : Description précise [IDENTIQUE à ByteRover]

**Contexte :**
[MÊME contenu que ByteRover]

**Décisions techniques :**
[MÊME contenu que ByteRover]

---
```

**🔹 ORDRE CHRONOLOGIQUE DANS MEMORY.MD :**
- **IMPORTANT** : Les entrées sont organisées du **plus vieux au plus récent** (ordre chronologique croissant)
- **🚨 CRITIQUE** : **TOUJOURS ajouter la nouvelle entrée À LA FIN du fichier MEMORY.md**
- **🚨 CRITIQUE** : **JAMAIS ajouter une entrée au début, au milieu, ou après l'en-tête**
- **🚨 CRITIQUE** : **JAMAIS insérer une entrée entre deux entrées existantes**
- **Méthode d'ajout** :
  1. Lire la dernière ligne du fichier MEMORY.md
  2. Identifier la dernière entrée (dernière section `### **YYYY-MM-DD**`)
  3. Ajouter la nouvelle entrée **immédiatement après** cette dernière entrée
  4. Vérifier que la nouvelle entrée est bien à la fin du fichier
- **Format** : Date au format ISO (YYYY-MM-DD) pour faciliter le tri
- **Raison** : Simplifie l'ajout de nouvelles entrées (pas besoin de chercher où les insérer, on ajoute toujours à la fin)
- **ByteRover** : Gère automatiquement l'ordre chronologique, pas besoin d'intervention

**Format de documentation automatique :**
```markdown
# UNIQUEMENT si l'utilisateur demande explicitement de documenter :

1. Préparer l'entrée avec :
   - Date du jour au format YYYY-MM-DD
   - Titre concis de la modification
   - Tags pertinents (#fichier.py, #concept, #technologie)
   - Actions effectuées détaillées
   - Contexte et décisions techniques

2. Stocker dans ByteRover (ordre chronologique automatique)
3. **🚨 CRITIQUE** : Ajouter à la fin de MEMORY.md (après la dernière entrée existante)
   - **JAMAIS** ajouter au début ou au milieu du fichier
   - **TOUJOURS** lire la fin du fichier pour trouver la dernière entrée
   - **TOUJOURS** ajouter après la dernière ligne du fichier
```

**🚨 RÈGLES STRICTES :**
- **TOUJOURS** documenter dans les **DEUX** systèmes (ByteRover ET MEMORY.md)
- **TOUJOURS** utiliser la **date du jour** au format ISO (YYYY-MM-DD)
- **TOUJOURS** maintenir contenu **identique** entre les deux systèmes
- **TOUJOURS** respecter ordre chronologique :
  - **ByteRover** : Ordre chronologique géré automatiquement par le système
  - **MEMORY.md** : Ordre chronologique **croissant** (plus vieux → plus récent)
  - **🚨 CRITIQUE MEMORY.md** : **TOUJOURS ajouter à la fin du fichier**
  - **🚨 CRITIQUE MEMORY.md** : **JAMAIS ajouter au début, au milieu, ou après l'en-tête**
  - **🚨 CRITIQUE MEMORY.md** : Lire la fin du fichier pour trouver la dernière entrée, puis ajouter après
- **TOUJOURS** documenter immédiatement après validation de la tâche complète

---

## 📋 Checklist Pré-Exécution

Avant de commencer **toute tâche**, l'agent doit vérifier :

- [ ] **ByteRover interrogé en premier (PRIORITÉ 1) :**
  - [ ] Niveau 1 (ciblé) : Query précise avec 3-5 résultats
  - [ ] Niveau 2 (contextuel) : Query élargie + tags architecturaux
  - [ ] Niveau 3 (exhaustif) : Queries multiples avec 10+ résultats
- [ ] **Si ByteRover indisponible → Fallback MEMORY.md :**
  - [ ] Niveau 1 (ciblé) : En-tête + tags pertinents via `grep`
  - [ ] Niveau 2 (contextuel) : + sections architecturales + `git log`
  - [ ] Niveau 3 (complet) : Lecture intégrale si nécessaire
- [ ] `AGENTS.md` a été consulté (ce fichier)
- [ ] `git log` vérifié pour les fichiers concernés (si applicable)
- [ ] L'architecture MVC du projet est comprise
- [ ] Les fichiers concernés par la tâche sont identifiés
- [ ] **Un plan détaillé a été créé et présenté à l'utilisateur**
- [ ] **L'approbation de l'utilisateur a été reçue**
- [ ] La politique de double storage (ByteRover + MEMORY.md) est respectée
  - [ ] **Pas de documentation automatique (Attendre demande explicite)**
  - [ ] Date du jour (YYYY-MM-DD) prête
  - [ ] Contenu identique pour ByteRover et MEMORY.md
  - [ ] **🚨 CRITIQUE** : Plan d'ajout à la fin de MEMORY.md (après la dernière entrée)
  - [ ] **🚨 CRITIQUE** : Vérifier que l'entrée sera ajoutée à la fin, JAMAIS au début/milieu
- [ ] Aucun fichier temporaire ne subsiste de sessions précédentes
- [ ] Aucun script de test ne reste dans le dépôt

---

## 🏗️ Règles Architecturales du Projet

### Architecture MVC Stricte

**🚨 OBLIGATION ULTIME : TOUJOURS respecter MVC**

Chaque type de logique doit être placé dans la bonne couche. **Aucune exception n'est tolérée.**

| Type de logique | Où ça va ? | Exemple |
|----------------|------------|---------|
| Logique métier | ✔ **MODELS** (ou services qui manipulent des models) | calcul mask, interpolation, C-scan |
| Logique applicative | ✔ **CONTROLLERS** | "user a cliqué → lancer pipeline → mettre à jour vue" |
| Logique UI | ✔ **VIEWS** | afficher slice, overlay, 3D, A-scan |

**Models** (gestion des données, état de l'application) :
- `models/annotation_model.py` : Gestion des données d'annotation, images, polygones, masques
- `models/ascan_model.py` : Gestion des données A-scan
- `models/cscan_model.py` : Gestion des données C-scan
- `models/defect_navigator_panel_model.py` : État du panneau de navigation des défauts
- `models/endviews_model.py` : Gestion des données des vues d'extrémité
- `models/settings_window_model.py` : État de la fenêtre de paramètres
- `models/volume_model.py` : Gestion des données de volume 3D

**Règles Models :**
- Gestion des données : images, polygones, masques
- État de l'application
- Mode mémoire vs fichier
- **Ne jamais** inclure de logique UI
- **Ne jamais** inclure de logique applicative (orchestration)

**Views** (interface utilisateur, rendu graphique) :
- `views/annotation_view.py` : Vue principale d'annotation
- `views/ascan_view.py` : Vue A-scan
- `views/cscan_view.py` : Vue C-scan
- `views/defect_analysis_dialog.py` : Dialogue d'analyse de défauts
- `views/defect_navigator_panel_view.py` : Vue du panneau de navigation des défauts
- `views/defect_navigator_panel.py` : Panneau de navigation des défauts
- `views/endview_export_dialog.py` : Dialogue d'export des vues d'extrémité
- `views/endview_view.py` : Vue des extrémités
- `views/parameters_3d_ascan_window.py` : Fenêtre des paramètres 3D A-scan
- `views/settings_window_view.py` : Vue de la fenêtre de paramètres
- `views/settings_window.py` : Fenêtre de paramètres
- `views/volume_view.py` : Vue de volume 3D

**Règles Views :**
- Interface utilisateur PyQt6
- Rendu graphique (VisPy, PyQtGraph)
- **Ne jamais** accéder directement au Model
- **Ne jamais** inclure de logique métier ou applicative

**Controllers** (coordination Model ↔ View, orchestration) :
- `controllers/annotation_controller.py` : Contrôleur principal d'annotation
- `controllers/ascan_controller.py` : Contrôleur A-scan
- `controllers/cscan_controller.py` : Contrôleur C-scan
- `controllers/endviews_controller.py` : Contrôleur des vues d'extrémité
- `controllers/volume_controller.py` : Contrôleur de volume 3D

**Règles Controllers :**
- Coordination Model ↔ View
- Gestion des événements utilisateur
- Logique applicative (orchestration)
- Seul point de contact entre Model et View
- **Ne jamais** inclure de logique métier complexe (déléguer aux services/models)
- **Ne jamais** inclure de logique UI (déléguer aux views)

### Gestion des Dépendances

**Dépendances autorisées :**
```
PyQt6          # Framework UI
numpy          # Calculs numériques
opencv-python  # Traitement d'image
vispy          # Visualisation 3D
pyqtgraph      # Graphiques 2D
h5py           # Fichiers HDF5
```

**Interdictions :**
- Pas de nouvelles dépendances sans justification explicite
- Pas de bibliothèques redondantes
- Pas de frameworks alternatifs (ex: Tkinter, wxPython)

### Politique de Documentation

**Autorisé :**
- `README.md` : Documentation principale (seul fichier officiel)
- `MEMORY.md` : Historique technique (ce fichier)
- `AGENTS.md` : Règles d'exécution (ce fichier)
- `DOCUMENTATION_COMPLETE.md` : Artefact historique (lecture seule)

**Interdit :**
- Fichiers de test temporaires (ex: `test_*.py` sans `pytest`)
- Documentation redondante
- Fichiers générés non listés dans `.gitignore`
- **Scripts de test qui restent dans le repo**

**RÈGLE CRITIQUE - Nettoyage des Scripts de Test :**
- **JAMAIS** laisser de scripts de test dans le dépôt
- **TOUJOURS** nettoyer immédiatement après utilisation
- Si un script de test est créé pour validation, le supprimer dès que le test est terminé
- Exemples de scripts à nettoyer : `test_*.py`, `debug_*.py`, `temp_*.py`, `_test.py`

**Workflow de test :**
1. Créer le script de test si nécessaire
2. Exécuter le test
3. **Supprimer immédiatement le script**
4. Documenter uniquement le résultat (pas le script lui-même)

**Après tests :**
- Toujours nettoyer les fichiers temporaires
- Supprimer tous les scripts de test créés
- Mettre à jour `.gitignore` si nécessaire
- Consigner les tests réussis dans `MEMORY.md` (uniquement sur approbation)

---

## 🏷️ Système de Tagging

### Format des Tags

Utiliser `#` suivi du nom du fichier ou concept :

**Fichiers :**
- `#annotation_controller.py`
- `#qt_mask_model.py`
- `#MEMORY.md`

**Concepts techniques :**
- `#mvc`
- `#global-state`
- `#3d-visualization`
- `#knowledge-management`

**Technologies :**
- `#pyqt6`
- `#vispy`
- `#numpy`
- `#hdf5`

**Catégories générales :**
- `#architecture`
- `#refactoring`
- `#documentation`
- `#testing`
- `#optimization`

**Branches Git :**
- Toujours ajouter un tag de branche de la forme `#branch:<nom_de_branche_git>`
- Exemple : `#branch:main`, `#branch:feature/segmentation-refactor`

### Règles de Tagging

1. **Obligatoire** : Taguer tous les fichiers modifiés
2. **Pertinent** : Ajouter les concepts/technologies impliqués
3. **Consistant** : Utiliser les mêmes tags pour les mêmes concepts
4. **Recherchable** : Faciliter la recherche dans `MEMORY.md`
5. **Obligatoire** : Pour chaque nouvelle entrée de mémoire (ByteRover + `MEMORY.md`), ajouter un tag de branche `#branch:<nom_de_branche_git>` dans la liste des tags

---

## 📊 Workflow de Développement

### 1. Initialisation
```
┌─────────────────────────────────────────────┐
│ 1. Lire AGENTS.md (ce fichier)              │
│ 2. Identifier la tâche                      │
│ 3. 🔹 PRIORITÉ : Interroger ByteRover       │
│    - Query ciblée (niveau 1/2/3)            │
│ 4. Si ByteRover KO → Fallback MEMORY.md     │
│    - Lecture selon niveau (1/2/3)           │
│ 5. Vérifier git log si pertinent            │
│ 6. Comprendre l'architecture                │
│ 7. Vérifier s'il reste des scripts          │
└─────────────────────────────────────────────┘
```

### 2. Planification (OBLIGATOIRE)
```
┌─────────────────────────────────────┐
│ 1. Créer un plan détaillé           │
│ 2. Découper en étapes numérotées    │
│ 3. Identifier fichiers à modifier   │
│ 4. Anticiper impacts et risques     │
│ 5. PRÉSENTER à l'utilisateur        │
│ 6. ATTENDRE approbation explicite   │
└─────────────────────────────────────┘
```

### 3. Exécution (Après Approbation)
```
┌─────────────────────────────────────┐
│ 1. Respecter MVC                    │
│ 2. Modifier les fichiers            │
│ 3. Tester localement                │
│ 4. Vérifier linter                  │
│ 5. Supprimer scripts de test        │
└─────────────────────────────────────┘
```

### 4. Validation
```
┌─────────────────────────────────────┐
│ 1. Présenter résultats              │
│ 2. ATTENDRE validation utilisateur  │
│ 3. Vérifier que la tâche est        │
│    complètement terminée            │
└─────────────────────────────────────┘
```

### 5. Documentation (SUR DEMANDE EXPLICITE UNIQUEMENT)
```
┌──────────────────────────────────────────────┐
│ 1. 🔹 ATTENTE : Attendre demande "Store"    │
│ 2. 🔹 ÉTAPE 1 : Store dans ByteRover        │
│    - byterover-store-knowledge(...)         │
│    - Utiliser date du jour (YYYY-MM-DD)     │
│    - Ordre chronologique automatique         │
│ 3. 🔹 ÉTAPE 2 : Ajouter dans MEMORY.md      │
│    - Contenu IDENTIQUE à ByteRover          │
│    - Utiliser date du jour (YYYY-MM-DD)     │
│    - 🚨 CRITIQUE : Ajouter À LA FIN du fichier│
│    - 🚨 CRITIQUE : Lire la fin du fichier    │
│    - 🚨 CRITIQUE : Après la dernière entrée │
│    - 🚨 CRITIQUE : JAMAIS au début/milieu    │
│    - Ordre : plus vieux → plus récent       │
│ 4. Vérifier date + tags identiques          │
│ 5. Confirmer ordre chronologique croissant  │
└──────────────────────────────────────────────┘
```

### 6. Nettoyage (CRITIQUE)
```
┌─────────────────────────────────────┐
│ 1. Supprimer TOUS scripts de test   │
│ 2. Supprimer fichiers temporaires   │
│ 3. Vérifier .gitignore              │
│ 4. Confirmer aucun déchet reste     │
└─────────────────────────────────────┘
```

---

## 🚨 Règles d'Exception et Cas Spéciaux

### Modifications Structurelles

Si l'architecture MVC doit être modifiée :

1. **Justification explicite** : Documenter pourquoi c'est nécessaire
2. **Discussion préalable** : Consulter l'historique pour éviter régressions
3. **Mise à jour complète** : Modifier Model, View ET Controller
4. **Documentation étendue** : Créer une entrée détaillée dans `MEMORY.md`

### Nouvelles Dépendances

Avant d'ajouter une nouvelle dépendance :

1. **Vérifier redondance** : Existe-t-il une solution avec les dépendances actuelles ?
2. **Évaluer l'impact** : Taille, compatibilité, maintenance
3. **Documenter** : Justifier l'ajout dans `MEMORY.md`
4. **Mettre à jour** : `requirements.txt` ET `README.md`

---

## 🧪 Testing et Validation

### Avant de Modifier

- Lire les tests existants (si présents)
- Comprendre les cas limites
- Identifier les dépendances du code

### Pendant les Modifications

- Tester au fur et à mesure
- Vérifier que l'architecture MVC est respectée
- S'assurer que les vues restent découplées du modèle

### Après les Modifications

- Exécuter l'application complète
- Tester les fonctionnalités impactées
- Vérifier les logs (app.log, nde_debug_log.txt)
- **Nettoyer les fichiers de test temporaires**

---

## 📝 Template d'Entrée (ByteRover + MEMORY.md)

**🔹 TEMPLATE BYTEROVER :**

```python
byterover-store-knowledge(
    messages="""
# YYYY-MM-DD — Titre concis de la modification

**Tags:** `#fichier1.py`, `#concept`, `#technologie`

**Actions effectuées:**
- Action 1 : Description précise avec détails techniques
- Action 2 : Description précise avec détails techniques

**Contexte:**
Explication du pourquoi de ces modifications, lien avec l'architecture,
décisions techniques prises. Inclure code snippets si pertinent.

**Décisions techniques:**
1. Décision 1 et justification détaillée
2. Décision 2 et justification détaillée

**Implémentation (si applicable):**
```python
# Exemple de code
def example():
    pass
```
"""
)
```

**🔹 TEMPLATE MEMORY.MD (IDENTIQUE) :**

```markdown
### **YYYY-MM-DD** — Titre concis de la modification

**Tags :** `#fichier1.py`, `#concept`, `#technologie`

**Actions effectuées :**
- Action 1 : Description précise avec détails techniques
- Action 2 : Description précise avec détails techniques

**Contexte :**
Explication du pourquoi de ces modifications, lien avec l'architecture,
décisions techniques prises. Inclure code snippets si pertinent.

**Décisions techniques :**
1. Décision 1 et justification détaillée
2. Décision 2 et justification détaillée

**Implémentation (si applicable) :**
```python
# Exemple de code
def example():
    pass
```

---
```

**🚨 RÈGLE : Contenu DOIT être identique entre ByteRover et MEMORY.md**

---

## 🔍 Support et Debugging

### En Cas de Problème

1. **Lire les logs** :
   - `app.log`
   - `nde_debug_log.txt`
   - `npz_debug_log.txt`

2. **Vérifier la cohérence** :
   - `MEMORY.md` est-il à jour ?
   - Les fichiers du projet sont-ils corrompus ?

3. **Consulter l'historique** :
   - Rechercher dans `MEMORY.md` pour des problèmes similaires
   - Identifier les changements récents qui pourraient être la cause

---

## 🎯 Résumé — Règles Essentielles

### Règles CRITIQUES (Non Négociables)

1. 🚨 **JAMAIS modifier sans plan approuvé**
   - Créer un plan détaillé avec étapes
   - Présenter à l'utilisateur
   - Attendre approbation explicite

2. 🚨 **JAMAIS laisser de scripts de test dans le dépôt**
   - Supprimer immédiatement après utilisation
   - Aucun `test_*.py`, `debug_*.py`, `temp_*.py`

3. 🚨 **JAMAIS documenter sans demande explicite**
   - **INTERDIT** : Documenter automatiquement
   - **OBLIGATOIRE** : Attendre l'ordre explicite de l'utilisateur
   - **TOUJOURS** stocker dans ByteRover ET MEMORY.md (les deux systèmes)
   - **TOUJOURS** utiliser la date du jour (YYYY-MM-DD)
   - **TOUJOURS** ajouter à la fin de MEMORY.md (après la dernière entrée, ordre chronologique croissant)
   - **TOUJOURS** maintenir contenu identique entre les deux systèmes

### Règles Importantes

4. ✅ **Toujours consulter ByteRover EN PREMIER (puis MEMORY.md si KO)**
5. ✅ **Vérifier `git log` pour le contexte des fichiers modifiés**
6. ✅ **Respecter l'architecture MVC strictement**
7. ✅ **Taguer tous les fichiers modifiés**
8. ✅ **Nettoyer tous les fichiers temporaires**
9. ✅ **Suivre la politique de double storage (ByteRover + MEMORY.md)**
10. ✅ **Tester avant de valider**

---

*Ce fichier est un guide vivant. Il doit être mis à jour si de nouvelles règles ou patterns émergent.*

*Dernière mise à jour : 2025-01-27 (Restructuration MEMORY.md : ordre chronologique croissant, ajout à la fin du fichier)*
