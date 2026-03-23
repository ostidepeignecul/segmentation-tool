# AGENTS.md — Règles d'Exécution

> **Objectif :** Cohérence, Stabilité et Mémoire.

## 1. Memory-First Development (Priorité Absolue)
Avant toute action, consulter la mémoire pour contexte et historique.

### Contexte Variable
1.  **Si prompt contient `rag`** :
    *   **Date du jour**.
    *   **Git Recent** : Analyser les diffs des 3 derniers commits (`git show HEAD~2..HEAD`).
2.  **Si prompt contient `brew`** :
    *   **Git Staged Changes** : Analyser les diffs des commits staged (`git diff --cached`).

### Stratégie Hybride
1.  **Priorité 1 : Ragbrew MCP (`search_memory`)**
    *   **Fréquence** : Exécuter **UNIQUEMENT si le mot `rag` est explicitement présent dans le prompt**.
    *   **Outil MCP** : `search_memory(query="Question ou Sujets", top_k=5)`
    *   **Détail chunk** : Si un résultat est pertinent, utiliser `get_memory_chunk(chunk_id="...")` pour le texte complet.
    *   *Exemple* : `search_memory(query="Logique de redimensionnement endview resize")`
2.  **Fallback : MEMORY.md en lecture directe** (Si MCP indisponible)

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
**RÈGLE :** Documenter **UNIQUEMENT si le mot `brew` est explicitement présent dans le prompt**.

Si le prompt contient le mot `brew` :
1.  **Récupérer Contexte** : Date du jour + **Git Staged Changes** (`git diff --cached`).
2.  **MEMORY.md** (Storage principal) : Ajouter l'entrée **À LA FIN** du fichier (ordre chronologique croissant).
    *   **FORMAT STRICT OBLIGATOIRE** :

    ```markdown

    ### YYYY-MM-DD - Titre de la modification
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

3.  **Ragbrew MCP** (`rebuild_index`) : Après modification de MEMORY.md, reconstruire l'index vectoriel.
    *   **Outil MCP** : `rebuild_index()` — ré-indexe automatiquement MEMORY.md.
    *   **Vérification** : `get_memory_status()` pour confirmer le nombre d'entrées/chunks.
    *   **CRITIQUE** : Toujours rebuilder après un `brew` pour que `search_memory` retourne les nouvelles entrées.

## 5. Propreté & Rigueur
*   **Scripts de Test** : Créer, Tester, **SUPPRIMER**. Ne jamais commiter `test_*.py`.
*   **Nettoyage** : Aucun fichier temporaire ne doit survivre à la tâche.
*   **Git Manuel** : JAMAIS de commit, push ou pull. L'utilisateur gère Git manuellement.
*   **Tagging** : `#branch:<nom>`, `#fichier.py`, `#concept`, `#technologie`. Tous les fichiers modifiés doivent être tagués.

## 6. Checklist de Démarrage
Avant de coder :
- [ ] Contexte acquis (Date + Git adapté à l'action).
- [ ] Mémoire consultée (Ragbrew `search_memory` ou MEMORY.md).
- [ ] Architecture comprise.
- [ ] Plan validé par l'utilisateur.
- [ ] Environnement propre (pas de vieux scripts).

## 7. Référence Ragbrew MCP

### Outils MCP Disponibles

| Outil | Usage | Description |
| :--- | :--- | :--- |
| `search_memory` | `rag` | Recherche sémantique dans l'index vectoriel de MEMORY.md |
| `get_memory_chunk` | `rag` | Récupère le texte complet d'un chunk par son ID |
| `get_memory_status` | `rag` / `brew` | Affiche le statut de l'index (nb entrées, chunks, projet) |
| `rebuild_index` | `brew` | Reconstruit l'index après modification de MEMORY.md |

### Workflow type

1. **Consulter** : `search_memory(query="sujet", top_k=5)` → résultats avec scores
2. **Approfondir** : `get_memory_chunk(chunk_id="...")` → texte complet
3. **Documenter** : Écrire dans MEMORY.md → `rebuild_index()` → `get_memory_status()`
