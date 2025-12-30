"""
MemeMotion - Application de Reconnaissance d'Expressions Faciales
Point d'entrée principal de l'application

Compare vos expressions faciales en temps réel avec des memes iconiques
"""

import sys
import os
from pathlib import Path
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
    print("=" * 65)
    print("  MemeMotion - Reconnaissance d'Expressions et Poses Corporelles")
    print("=" * 65)
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
    if window.initialize_components(
        str(MEME_FOLDER),
        str(METADATA_FILE)
    ):
        print("✓ Composants initialisés avec succès")
        print()
        print("🎭 Application prête!")
        print("   - La webcam capture vos expressions et poses en temps réel")
        print("   - Détection faciale ET corporelle simultanées")
        print("   - Le meme correspondant s'affiche automatiquement (score > 70%)")
        print("   - Les memes changent dynamiquement selon vos mouvements")
        print()
        print("Faites des grimaces ou des poses pour voir les memes correspondants! 🚀")
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
