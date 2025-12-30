# MemeMotion

**Application de reconnaissance d'expressions faciales en temps réel**

MemeMotion compare vos expressions faciales avec une base de memes iconiques et vous donne un score de similarité en direct. Prenez des screenshots automatiques quand vous réussissez à reproduire parfaitement une expression de meme !

---

## Fonctionnalités

- **Capture webcam en temps réel** (30 FPS minimum)
- **Détection faciale ultra-rapide** avec MediaPipe Face Mesh (468 landmarks)
- **Analyse d'expressions** : ouverture bouche, plissement des yeux, sourcils, inclinaison tête
- **Matching intelligent** avec cosine similarity pour comparer vos expressions
- **Score de similarité live** (0-100%) mis à jour toutes les 100ms
- **Screenshots automatiques** quand votre score dépasse 85%
- **Interface PyQt5 moderne** avec affichage split-screen
- **Extensible** : ajoutez vos propres memes facilement !

---

## Architecture du Projet

```
meme-motion/
├── main.py                    # Point d'entrée de l'application
├── requirements.txt           # Dépendances Python
├── README.md                  # Ce fichier
├── src/
│   ├── __init__.py
│   ├── camera_handler.py      # Gestion webcam OpenCV
│   ├── face_detector.py       # MediaPipe Face Mesh (468 landmarks)
│   ├── expression_analyzer.py # Extraction features (mouth_ratio, eyebrow_angle, etc.)
│   ├── meme_matcher.py        # Algorithme de matching (cosine similarity)
│   ├── ui_manager.py          # Interface PyQt5
│   └── screenshot_handler.py  # Sauvegarde captures
├── assets/
│   └── memes/                 # Images de memes de référence (.jpg, .png)
├── data/
│   └── meme_metadata.json     # Métadonnées des memes (auto-généré)
└── screenshots/               # Screenshots sauvegardés (auto-créé)
```

---

## Installation

### Prérequis

- **Python 3.8+**
- **Webcam fonctionnelle**
- **Système d'exploitation** : Windows, macOS, ou Linux

### Étapes d'installation

1. **Cloner ou télécharger le projet**

```bash
cd /path/to/MemeMotion
```

2. **Créer un environnement virtuel** (recommandé)

```bash
python -m venv venv

# Activer l'environnement
# Sur macOS/Linux:
source venv/bin/activate

# Sur Windows:
venv\Scripts\activate
```

3. **Installer les dépendances**

```bash
pip install -r requirements.txt
```

4. **Ajouter vos images de memes**

Placez vos images de memes (format `.jpg` ou `.png`) dans le dossier `assets/memes/`.

Les images doivent contenir **un visage clairement visible** pour que l'analyse fonctionne.

Exemples de memes :
- `awkward_monkey.jpg` - Le singe qui détourne le regard
- `shocked_pikachu.png` - Pikachu surpris
- `drake_yes.jpg` - Drake qui approuve

**Important** : Au moins une image est nécessaire pour que l'application fonctionne.

---

## Utilisation

### Lancer l'application

```bash
python main.py
```

Au premier lancement, l'application va :
1. Analyser toutes les images dans `assets/memes/`
2. Détecter les visages et extraire les features d'expression
3. Sauvegarder les métadonnées dans `data/meme_metadata.json`

Les lancements suivants seront plus rapides car les métadonnées sont déjà calculées.

### Interface

```
┌────────────────────────────────────────────────────┐
│  [Votre Expression]     [Meme Matching]            │
│   (Webcam en direct)    (Image du meme le + proche)│
│                                                     │
│          Score: 87%        Awkward Monkey          │
│                                                     │
│  [📸 Prendre un Screenshot]  [❌ Quitter]          │
└────────────────────────────────────────────────────┘
```

### Fonctionnalités de l'interface

- **Webcam** : Affiche votre visage en temps réel avec les landmarks facials
- **Meme Matching** : Affiche le meme qui correspond le mieux à votre expression
- **Score** : Pourcentage de similarité (couleur verte = bon match, rouge = mauvais)
- **Screenshot Manuel** : Cliquez pour capturer l'instant
- **Auto-Screenshot** : Capture automatique quand score > 85%

### Fichiers de sortie

Les screenshots sont sauvegardés dans `screenshots/` avec le format :

```
auto_awkward_monkey_87_20250115_143022.jpg
└─┬─┘ └──────┬───────┘ ├┘ └──────┬───────┘
  │          │         │          │
  │          │         │          └── Timestamp
  │          │         └──────────── Score
  │          └────────────────────── Nom du meme
  └───────────────────────────────── Type (auto/manual)
```

Comparaisons côte-à-côte disponibles aussi :

```
comparison_auto_shocked_face_92_20250115_143045.jpg
```

---

## Ajouter vos propres memes

### Méthode simple (recommandée)

1. **Téléchargez une image de meme** avec un visage visible
2. **Placez-la dans** `assets/memes/`
3. **Supprimez le fichier** `data/meme_metadata.json`
4. **Relancez l'application** - elle va analyser tous les memes automatiquement

### Méthode manuelle (avancée)

Si vous voulez éditer manuellement les features d'un meme :

1. Ouvrez `data/meme_metadata.json`
2. Ajoutez une nouvelle entrée :

```json
{
  "mon_meme_custom": {
    "name": "Mon Meme Custom",
    "image": "mon_meme.jpg",
    "key_features": {
      "mouth_open_ratio": 0.5,
      "mouth_aspect_ratio": 0.5,
      "eye_squint": 0.4,
      "eyebrow_raise": 0.6,
      "head_rotation": 0.1
    }
  }
}
```

**Features expliquées** :
- `mouth_open_ratio` : Ouverture de la bouche (0=fermée, 1=grande ouverte)
- `mouth_aspect_ratio` : Ratio hauteur/largeur de la bouche
- `eye_squint` : Plissement des yeux (0=ouverts, 1=fermés)
- `eyebrow_raise` : Levée des sourcils (0=bas, 1=haut)
- `head_rotation` : Rotation de la tête

---

## Troubleshooting

### Erreur : "Impossible d'ouvrir la caméra"

**Causes possibles** :
- Webcam déjà utilisée par une autre application
- Permissions caméra refusées (macOS/Windows)
- Pilotes webcam manquants ou obsolètes

**Solutions** :
1. Fermez toutes les applications utilisant la webcam (Zoom, Skype, etc.)
2. Sur **macOS** : Système > Confidentialité > Caméra > Autoriser Terminal/Python
3. Sur **Windows** : Paramètres > Confidentialité > Caméra > Autoriser applications
4. Redémarrez votre ordinateur

### Erreur : "Aucun visage détecté"

**Causes possibles** :
- Éclairage insuffisant
- Visage trop loin de la caméra
- Angle de la caméra inadapté

**Solutions** :
1. Améliorez l'éclairage de votre pièce
2. Rapprochez-vous de la webcam
3. Regardez directement la caméra
4. Vérifiez que votre visage est bien dans le cadre

### Erreur : "Aucune image de meme trouvée"

**Causes possibles** :
- Dossier `assets/memes/` vide
- Format d'image non supporté

**Solutions** :
1. Ajoutez des images `.jpg` ou `.png` dans `assets/memes/`
2. Vérifiez que les images contiennent un visage visible
3. Assurez-vous que les fichiers ne sont pas corrompus

### Performance lente (< 30 FPS)

**Causes possibles** :
- Machine trop lente
- Trop de memes dans la base

**Solutions** :
1. Réduisez le nombre de memes dans `assets/memes/`
2. Fermez les applications lourdes en arrière-plan
3. Utilisez une webcam de résolution inférieure (640x480 au lieu de 1080p)

### Imports manquants (ModuleNotFoundError)

**Cause** : Dépendances non installées

**Solution** :
```bash
pip install -r requirements.txt --upgrade
```

---

## Technologies utilisées

- **[MediaPipe](https://google.github.io/mediapipe/)** - Détection faciale ultra-rapide (468 landmarks)
- **[OpenCV](https://opencv.org/)** - Capture webcam et traitement d'image
- **[PyQt5](https://www.riverbankcomputing.com/software/pyqt/)** - Interface graphique moderne
- **[NumPy](https://numpy.org/)** - Calculs vectoriels pour le matching
- **[Pillow](https://python-pillow.org/)** - Manipulation d'images

---

## Performances

- **FPS** : 30+ sur machine moderne (Intel i5/AMD Ryzen 5)
- **Latence matching** : < 100ms
- **Résolution webcam** : 640x480 (optimisé pour performance)
- **Landmarks détectés** : 468 points faciaux

---

## Limitations connues

- Ne fonctionne qu'avec **un seul visage** à la fois
- Nécessite un **bon éclairage** pour la détection
- Performance réduite sur machines anciennes (< 2015)
- Masques faciaux empêchent la détection

---

## Améliorations futures

- [ ] Support multi-visages
- [ ] Mode entraînement pour créer ses propres expressions
- [ ] Historique des meilleurs scores
- [ ] Partage social des screenshots
- [ ] Support GIF animés
- [ ] Mode challenge avec timer

---

## Licence

Ce projet est fourni à des fins éducatives et personnelles.

Les images de memes utilisées peuvent être soumises à des droits d'auteur.
Assurez-vous d'avoir les droits nécessaires avant de distribuer vos screenshots.

---

## Crédits

Développé avec Python et les bibliothèques open-source :
- Google MediaPipe Team
- OpenCV Contributors
- PyQt5 / Riverbank Computing
- NumPy Community

---

## Support

Pour toute question ou problème :
1. Consultez la section **Troubleshooting**
2. Vérifiez que vos dépendances sont à jour
3. Testez avec une webcam différente si possible

Bon amusement avec MemeMotion ! 🎭🚀
