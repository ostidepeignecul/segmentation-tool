## Installation
### 0. Créer fichier requirements.txt

```bash
# à faire seulement si on modifie le requirements.in pour recréer le requirements.txt
pip install pip-tools
python -m piptools compile requirements.in
```

### 1. Environnement conda 

```bash
# Créer un environnement conda
conda create -n segmentation-tool python=3.12 -y
conda activate segmentation-tool
```


### 2. Installation des dépendances du pipeline

```bash
# Installer toutes les dépendances depuis le fichier requirements.txt
pip install -r requirements.txt
```
