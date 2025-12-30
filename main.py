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
    print("=" * 70)
    print("  MemeMotion - Reconnaissance Faciale, Corporelle et des Mains")
    print("=" * 70)
    print()

    # Vérification des dossiers
    check_folders()

    # Vérification de la présence de memes (incluant .jpeg)
    meme_count = (len(list(MEME_FOLDER.glob('*.jpg'))) +
                  len(list(MEME_FOLDER.glob('*.jpeg'))) +
                  len(list(MEME_FOLDER.glob('*.png'))))
    if meme_count == 0:
        print("⚠️  ATTENTION: Aucune image de meme trouvée!")
        print(f"   Ajoutez des images (.jpg, .jpeg ou .png) dans: {MEME_FOLDER}")
        print("   L'application va démarrer mais ne pourra pas faire de matching.")
        print()

    # Création de l'application Qt
    app = QApplication(sys.argv)

    # Création de la fenêtre principale
    window = MemeMotionUI()

    # Initialisation des composants
    print("Initialisation des composants...")

    success = window.initialize_components(
        str(MEME_FOLDER),
        str(METADATA_FILE)
    )

    if success:
        print("✓ Composants initialisés avec succès")
        print()
        print("🎭 Application prête!")
        print("   👤 Détection faciale : 468 points du visage")
        print("   🤸 Détection corporelle : 33 points du corps")
        print("   ✋ Détection des mains : 21 points par main avec TOUS les doigts")
        print("   🎯 Le meme correspondant s'affiche automatiquement")
        print()
        print("Faites des grimaces, des poses ou des gestes avec vos mains ! 🚀")
        print("L'application détecte maintenant chaque doigt individuellement !")
        print()

        # Affichage de la fenêtre
        window.show()

        # Lancement de la boucle événementielle
        sys.exit(app.exec_())
    else:
        print("❌ Erreur lors de l'initialisation")
        print("   Vérifiez que votre webcam est connectée et accessible")
        sys.exit(1)


if __name__ == "__main__":
    main()
