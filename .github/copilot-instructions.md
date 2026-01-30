# AGENTS.md — Règles d'Exécution

> **Objectif :** Cohérence, Stabilité et Mémoire.

## 1. Memory-First Development (Priorité Absolue)
Avant toute action, consulter la mémoire pour contexte et historique.

### Contexte Obligatoire (Avant tout)
1.  **Date du jour** : Connaître la date actuelle.
2.  **Git Recent** : Analyser les diffs des 3 derniers commits (`git show HEAD~2..HEAD`).
2.  **Git Staged Changes** : Analyser les diffs des commits staged (`git diff --cached`).

### Stratégie Hybride
1.  **Priorité 1 : ByteRover (`brv query`)**
    *   **Attente** : Patienter jusqu'à **200 secondes** pour la réponse.
    *   **Format Requis** : `"Question ou Sujets #tag1 #tag2"`
    *   *Exemple* : `brv query "Logique de redimensionnement #endview #resize"`
2.  **Fallback : MEMORY.md** (Si timeout > 200s ou KO)

## 2. Plan & Validation
**RÈGLE D'OR :** Pas de code sans plan validé.
1.  **Analyser** : Comprendre le problème et l'architecture.
2.  **Planifier** : Étapes claires, chiffrées, présentées à l'utilisateur.
3.  **Valider** : **Attendre l'approbation explicite** (ne jamais assumer).

## 3. Architecture MVC Stricte
**Aucune exception.** Chaque couche à sa place.

| Couche | Rôle | Interdictions |
| :--- | :--- | :--- |
| **Model** | Données, Logique métier, État | JAMAIS de logique UI ou App. |
| **View** | Interface, Rendu (PyQt, VisPy) | JAMAIS de logique métier. Pas d'accès direct au Model. |
| **Controller** | Orchestration, Events, Lien Model-View | Pas de métier lourd ni de dessin UI. |

*Exemples:*
*   *Models*: `models/annotation_model.py`, `models/volume_model.py`
*   *Views*: `views/annotation_view.py`, `views/volume_view.py`
*   *Controllers*: `controllers/annotation_controller.py`, `controllers/volume_controller.py`

## 4. Documentation & Double Storage
**RÈGLE :** Documenter **UNIQUEMENT sur demande explicite** de l'utilisateur.

Si demandé ("Store this") :
1.  **Récupérer Contexte** : Date du jour + Git Diff 3 derniers commits + Git Staged Changes (Ce qui est fait actuellement).
2.  **ByteRover** : `brv curate "CONTENU_COMPLET_DE_LA_MEMOIRE (Titre + Tags + Actions + Contexte)"`
    *   **Règle** : Tout le bloc texte doit être entre guillemets `" "`.
    *   **Attention** : Échapper les guillemets internes (`\"`).
    *   *Exemple* : `brv curate "### Titre ... avec \"citation\" interne ..."`
    *   **Fallback Shell** : Si erreur d'arguments (PowerShell), retirer les guillemets internes problématiques.
3.  **MEMORY.md** : Copie **IDENTIQUE** du contenu.
    *   **CRITIQUE** : Toujours ajouter **À LA FIN** du fichier (ordre chronologique croissant).
    *   **FORMAT STRICT OBLIGATOIRE** :

    ```markdown
    ### **YYYY-MM-DD** — Titre de la modification
    **Tags :** `#branch:<nom>`, `#fichier.py`, `#concept` 

    **Actions effectuées :**
    - Action 1 (détail technique)
    - Action 2

    **Contexte :**
    Pourquoi ce changement ? Reference aux tickets ou discussions.

    **Décisions techniques :**
    1. Décision A (Justification)
    2. Décision B
    ```

## 5. Propreté & Rigueur
*   **Scripts de Test** : Créer, Tester, **SUPPRIMER**. Ne jamais commiter `test_*.py`.
*   **Nettoyage** : Aucun fichier temporaire ne doit survivre à la tâche.
*   **Git Manuel** : JAMAIS de commit, push ou pull. L'utilisateur gère Git manuellement.
*   **Tagging** : `#branch:<nom>`, `#fichier.py`, `#concept`, `#technologie`. Tous les fichiers modifiés doivent être tagués.

## 6. Checklist de Démarrage
Avant de coder :
- [ ] Contexte acquis (Date + Git Diff 3 derniers commits + Git Staged Changes).
- [ ] Mémoire consultée (ByteRover ou MEMORY.md).
- [ ] Architecture comprise.
- [ ] Plan validé par l'utilisateur.
- [ ] Environnement propre (pas de vieux scripts).

## 7. Reference ByteRover CLI

### Available Commands

- `brv curate` - Curate context to the context tree
- `brv query` - Query and retrieve information from the context tree
- `brv status` - Show CLI status and project information

Run `brv query --help` for query instruction and `brv curate --help` for curation instruction.
