"""
Screenshot Handler Module
Gestion de la sauvegarde des captures d'écran
"""

import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional


class ScreenshotHandler:
    """Handles screenshot capture and saving."""

    def __init__(self, output_folder: str = "screenshots"):
        """
        Initialize screenshot handler.

        Args:
            output_folder: Dossier de sauvegarde des screenshots
        """
        self.output_folder = Path(output_folder)

        # Créer le dossier s'il n'existe pas
        self.output_folder.mkdir(parents=True, exist_ok=True)

        # Cooldown pour éviter trop de screenshots auto
        self.last_auto_screenshot_time = None
        self.auto_screenshot_cooldown = 2.0  # secondes

    def save_screenshot(self, frame: np.ndarray, meme_name: str,
                       score: float, is_auto: bool = False) -> Optional[str]:
        """
        Save screenshot with metadata in filename.

        Args:
            frame: Image BGR à sauvegarder
            meme_name: Nom du meme matché
            score: Score de matching (0-100)
            is_auto: True si screenshot automatique, False si manuel

        Returns:
            Chemin du fichier sauvegardé, ou None si erreur
        """
        try:
            # Vérification du cooldown pour auto-screenshot
            if is_auto:
                current_time = datetime.now()
                if self.last_auto_screenshot_time is not None:
                    elapsed = (current_time - self.last_auto_screenshot_time).total_seconds()
                    if elapsed < self.auto_screenshot_cooldown:
                        return None
                self.last_auto_screenshot_time = current_time

            # Génération du nom de fichier
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            meme_name_clean = meme_name.replace(' ', '_').lower()
            score_str = f"{int(score)}"

            prefix = "auto" if is_auto else "manual"
            filename = f"{prefix}_{meme_name_clean}_{score_str}_{timestamp}.jpg"

            filepath = self.output_folder / filename

            # Sauvegarde de l'image
            success = cv2.imwrite(str(filepath), frame)

            if success:
                print(f"Screenshot sauvegardé: {filename}")
                return str(filepath)
            else:
                print(f"Erreur lors de la sauvegarde du screenshot")
                return None

        except Exception as e:
            print(f"Erreur lors de la sauvegarde du screenshot: {e}")
            return None

    def save_comparison_screenshot(self, user_frame: np.ndarray,
                                   meme_frame: np.ndarray,
                                   meme_name: str,
                                   score: float,
                                   is_auto: bool = False) -> Optional[str]:
        """
        Save side-by-side comparison screenshot.

        Args:
            user_frame: Frame de l'utilisateur
            meme_frame: Frame du meme de référence
            meme_name: Nom du meme matché
            score: Score de matching
            is_auto: True si auto-screenshot

        Returns:
            Chemin du fichier sauvegardé, ou None si erreur
        """
        try:
            # Vérification du cooldown pour auto-screenshot
            if is_auto:
                current_time = datetime.now()
                if self.last_auto_screenshot_time is not None:
                    elapsed = (current_time - self.last_auto_screenshot_time).total_seconds()
                    if elapsed < self.auto_screenshot_cooldown:
                        return None
                self.last_auto_screenshot_time = current_time

            # Redimensionner les images pour qu'elles aient la même hauteur
            height = max(user_frame.shape[0], meme_frame.shape[0])

            # Redimensionner user_frame
            user_aspect = user_frame.shape[1] / user_frame.shape[0]
            user_width = int(height * user_aspect)
            user_resized = cv2.resize(user_frame, (user_width, height))

            # Redimensionner meme_frame
            meme_aspect = meme_frame.shape[1] / meme_frame.shape[0]
            meme_width = int(height * meme_aspect)
            meme_resized = cv2.resize(meme_frame, (meme_width, height))

            # Créer une image côte à côte
            comparison = np.hstack([user_resized, meme_resized])

            # Ajouter le texte du score
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"Match: {int(score)}% - {meme_name}"
            text_size = cv2.getTextSize(text, font, 1, 2)[0]

            # Position du texte (centré en haut)
            text_x = (comparison.shape[1] - text_size[0]) // 2
            text_y = 40

            # Fond noir pour le texte
            cv2.rectangle(comparison,
                         (text_x - 10, text_y - 30),
                         (text_x + text_size[0] + 10, text_y + 10),
                         (0, 0, 0),
                         -1)

            # Texte blanc
            cv2.putText(comparison, text, (text_x, text_y),
                       font, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Génération du nom de fichier
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            meme_name_clean = meme_name.replace(' ', '_').lower()
            score_str = f"{int(score)}"

            prefix = "comparison_auto" if is_auto else "comparison_manual"
            filename = f"{prefix}_{meme_name_clean}_{score_str}_{timestamp}.jpg"

            filepath = self.output_folder / filename

            # Sauvegarde de l'image
            success = cv2.imwrite(str(filepath), comparison)

            if success:
                print(f"Screenshot comparaison sauvegardé: {filename}")
                return str(filepath)
            else:
                print(f"Erreur lors de la sauvegarde du screenshot")
                return None

        except Exception as e:
            print(f"Erreur lors de la sauvegarde du screenshot comparaison: {e}")
            return None

    def get_screenshot_count(self) -> int:
        """
        Get total number of screenshots saved.

        Returns:
            Nombre de screenshots dans le dossier
        """
        try:
            return len(list(self.output_folder.glob('*.jpg')))
        except Exception as e:
            print(f"Erreur lors du comptage des screenshots: {e}")
            return 0

    def clear_old_screenshots(self, keep_last: int = 50) -> int:
        """
        Clear old screenshots, keeping only the most recent ones.

        Args:
            keep_last: Nombre de screenshots à conserver

        Returns:
            Nombre de fichiers supprimés
        """
        try:
            # Liste tous les screenshots triés par date de modification
            screenshots = sorted(self.output_folder.glob('*.jpg'),
                               key=lambda x: x.stat().st_mtime,
                               reverse=True)

            # Supprime les anciens
            deleted_count = 0
            for screenshot in screenshots[keep_last:]:
                screenshot.unlink()
                deleted_count += 1

            if deleted_count > 0:
                print(f"{deleted_count} anciens screenshots supprimés")

            return deleted_count

        except Exception as e:
            print(f"Erreur lors du nettoyage des screenshots: {e}")
            return 0
