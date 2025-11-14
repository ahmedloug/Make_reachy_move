# Reachy Mini — Contrôle par pose et voix

Projet: contrôler un Reachy Mini (simulation ou réel) avec
- la pose détectée par la webcam (hanches pour translation de la tête, poignets pour antennes)
- quelques commandes vocales simples

## Ce qu’il faut avoir

- Python 3.10 ou 3.11
- MuJoCo installé (pour la simulation)
- Webcam + micro
- Modèle Vosk hors-ligne: dossier `vosk-model-small-en-us-0.15` à la racine (téléchargé depuis Vosk)
- Le modèle de pose `yolov8n-pose.pt` est téléchargé automatiquement

## Installation minimale

```bash
pip install requirements.txt

```

## Lancer la simulation (optionnel)

```bash
mjpython -m reachy_mini.daemon.app.main --sim
```

## Lancer le script

```bash
python reachy-le-combattant.py
```

## Commandes vocales prises en charge

```
hit
respect
yes
neck crack
```

## Résumé

Webcam pour la pose → mouvement tête + antennes.
Micro pour déclencher des animations.
MuJoCo requis si vous n’avez pas le robot réel.