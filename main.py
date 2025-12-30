"""
MemeMotion - Application de Reconnaissance d'Expressions Faciales
Point d'entrée principal de l'application

Compare vos expressions faciales en temps réel avec des memes iconiques
"""

import sys
import os
from pathlib import Path

# IMPORTANT: Rediriger stderr AVANT tout import pour supprimer les warnings MediaPipe
_original_stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

# Suppression des warnings de TensorFlow et MediaPipe
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore')

from PyQt5.QtWidgets import QApplication
from src.ui_manager import MemeMotionUI


# Configuration des chemins
BASE_DIR = Path(__file__).parent
MEME_FOLDER = BASE_DIR / "assets" / "memes"
METADATA_FILE = BASE_DIR / "data" / "meme_metadata.json"


def check_folders():
    """Vérifie et crée les dossiers nécessaires."""
    folders = [MEME_FOLDER, BASE_DIR / "data"]

    for folder in folders:
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)
            print(f"Dossier créé: {folder}")


def main():
    """Point d'entrée principal de l'application."""
    print("\n" + "=" * 50)
    print("  MemeMotion")
    print("=" * 50 + "\n")

    # Vérification des dossiers
    check_folders()

    # Vérification de la présence de memes
    meme_count = (len(list(MEME_FOLDER.glob('*.jpg'))) +
                  len(list(MEME_FOLDER.glob('*.jpeg'))) +
                  len(list(MEME_FOLDER.glob('*.png'))))
    if meme_count == 0:
        print("ATTENTION: Aucune image trouvee")
        print(f"Ajoutez des images dans: {MEME_FOLDER}\n")

    # Création de l'application Qt
    app = QApplication(sys.argv)
    window = MemeMotionUI()

    # Initialisation
    print("Initialisation...", end=" ")
    success = window.initialize_components(
        str(MEME_FOLDER),
        str(METADATA_FILE)
    )

    if success:
        print("OK\n")
        print(f"{meme_count} memes charges")
        print("Pret.\n")

        window.show()
        sys.exit(app.exec_())
    else:
        print("ERREUR")
        print("Verifiez votre webcam\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
