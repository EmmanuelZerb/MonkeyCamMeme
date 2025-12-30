"""
Meme Matcher Module
Algorithme de matching entre expressions faciales et memes de référence
"""

import json
import os
import cv2
import numpy as np
from typing import Dict, Optional, List, Tuple
from pathlib import Path

from .face_detector import FaceDetector
from .expression_analyzer import ExpressionAnalyzer
from .pose_detector import PoseDetector
from .pose_analyzer import PoseAnalyzer


class MemeMatcher:
    """Matches current facial expression with meme database using cosine similarity."""

    def __init__(self, meme_folder: str, metadata_file: str):
        """
        Initialize meme matcher.

        Args:
            meme_folder: Chemin vers le dossier contenant les images de memes
            metadata_file: Chemin vers le fichier JSON de métadonnées
        """
        self.meme_folder = Path(meme_folder)
        self.metadata_file = Path(metadata_file)
        self.meme_database: Dict = {}
        self.face_detector = FaceDetector()
        self.expression_analyzer = ExpressionAnalyzer()
        self.pose_detector = PoseDetector()
        self.pose_analyzer = PoseAnalyzer()

        # Seuil pour affichage automatique du meme
        self.AUTO_DISPLAY_THRESHOLD = 70.0  # Réduit pour afficher plus facilement

    def load_or_generate_metadata(self) -> bool:
        """
        Load existing metadata or generate from meme images.

        Returns:
            True si chargement/génération réussi, False sinon
        """
        # Si le fichier existe, on le charge
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self.meme_database = json.load(f)
                print(f"Métadonnées chargées: {len(self.meme_database)} memes")
                return True
            except Exception as e:
                print(f"Erreur lors du chargement des métadonnées: {e}")
                print("Génération des métadonnées...")

        # Sinon on génère depuis les images
        return self.generate_metadata_from_images()

    def generate_metadata_from_images(self) -> bool:
        """
        Generate metadata by analyzing all meme images in folder.

        Returns:
            True si génération réussie, False sinon
        """
        if not self.meme_folder.exists():
            print(f"Erreur: Dossier memes introuvable: {self.meme_folder}")
            return False

        # Extensions d'images supportées (incluant .jpeg)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

        meme_files = []
        for ext in image_extensions:
            meme_files.extend(self.meme_folder.glob(f'*{ext}'))
            meme_files.extend(self.meme_folder.glob(f'*{ext.upper()}'))

        if not meme_files:
            print(f"Aucune image trouvée dans {self.meme_folder}")
            return False

        print(f"Analyse de {len(meme_files)} images de memes...")

        self.meme_database = {}

        for meme_path in meme_files:
            try:
                # Chargement de l'image
                image = cv2.imread(str(meme_path))
                if image is None:
                    print(f"Erreur: Impossible de charger {meme_path.name}")
                    continue

                # Détection du visage et de la pose
                face_detection = self.face_detector.detect_face(image)
                pose_detection = self.pose_detector.detect_pose(image)

                # Au moins un des deux doit être détecté
                if (not face_detection or not face_detection['face_found']) and \
                   (not pose_detection or not pose_detection['pose_found']):
                    print(f"Aucun visage ou pose détecté dans {meme_path.name}")
                    continue

                # Analyse de l'expression faciale
                face_features = None
                if face_detection and face_detection['face_found']:
                    face_features = self.expression_analyzer.analyze_expression(face_detection)

                # Analyse de la pose corporelle
                pose_features = None
                if pose_detection and pose_detection['pose_found']:
                    pose_features = self.pose_analyzer.analyze_pose(pose_detection)

                if face_features is None and pose_features is None:
                    print(f"Erreur d'analyse pour {meme_path.name}")
                    continue

                # Création de l'ID du meme (nom sans extension)
                meme_id = meme_path.stem
                meme_name = meme_id.replace('_', ' ').title()

                # Ajout à la base de données avec les deux types de features
                self.meme_database[meme_id] = {
                    'name': meme_name,
                    'image': meme_path.name,
                    'face_features': face_features,
                    'pose_features': pose_features
                }

                detection_types = []
                if face_features:
                    detection_types.append("visage")
                if pose_features:
                    detection_types.append("pose")

                print(f"✓ {meme_name} analysé ({', '.join(detection_types)})")

            except Exception as e:
                print(f"Erreur lors de l'analyse de {meme_path.name}: {e}")
                continue

        # Sauvegarde des métadonnées
        if self.meme_database:
            try:
                with open(self.metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(self.meme_database, f, indent=2, ensure_ascii=False)
                print(f"Métadonnées sauvegardées: {len(self.meme_database)} memes")
                return True
            except Exception as e:
                print(f"Erreur lors de la sauvegarde des métadonnées: {e}")
                return False
        else:
            print("Aucun meme valide trouvé")
            return False

    def calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two feature vectors.

        Args:
            vec1: Premier vecteur de features
            vec2: Deuxième vecteur de features

        Returns:
            Score de similarité (0-1)
        """
        # Éviter division par zéro
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Cosine similarity
        similarity = np.dot(vec1, vec2) / (norm1 * norm2)

        # Normalisation entre 0 et 1
        similarity = (similarity + 1) / 2

        return float(similarity)

    def find_best_match(self, current_face_features: Optional[Dict[str, float]],
                       current_pose_features: Optional[Dict[str, float]]) -> Optional[Tuple[str, str, float, str]]:
        """
        Find best matching meme for current expression and pose.

        Args:
            current_face_features: Features de l'expression faciale actuelle
            current_pose_features: Features de la pose corporelle actuelle

        Returns:
            Tuple (meme_id, meme_name, score, image_path) du meilleur match,
            ou None si pas de match
        """
        if not self.meme_database:
            return None

        # Conversion des features actuelles en vecteurs
        current_face_vector = None
        if current_face_features:
            current_face_vector = self.expression_analyzer.features_to_vector(current_face_features)

        current_pose_vector = None
        if current_pose_features:
            current_pose_vector = self.pose_analyzer.features_to_vector(current_pose_features)

        if current_face_vector is None and current_pose_vector is None:
            return None

        best_match = None
        best_score = 0.0

        # Comparaison avec tous les memes
        for meme_id, meme_data in self.meme_database.items():
            # Calcul du score combiné (visage + pose)
            total_score = 0.0
            weight_sum = 0.0

            # Score du visage (poids 0.6)
            if current_face_vector is not None and meme_data.get('face_features'):
                meme_face_vector = self.expression_analyzer.features_to_vector(meme_data['face_features'])
                face_similarity = self.calculate_cosine_similarity(current_face_vector, meme_face_vector)
                total_score += face_similarity * 0.6
                weight_sum += 0.6

            # Score de la pose (poids 0.4)
            if current_pose_vector is not None and meme_data.get('pose_features'):
                meme_pose_vector = self.pose_analyzer.features_to_vector(meme_data['pose_features'])
                pose_similarity = self.calculate_cosine_similarity(current_pose_vector, meme_pose_vector)
                total_score += pose_similarity * 0.4
                weight_sum += 0.4

            # Si aucun match possible, passer au suivant
            if weight_sum == 0:
                continue

            # Normalisation du score
            score = (total_score / weight_sum) * 100

            if score > best_score:
                best_score = score
                best_match = (
                    meme_id,
                    meme_data['name'],
                    score,
                    str(self.meme_folder / meme_data['image'])
                )

        return best_match

    def should_auto_display(self, score: float) -> bool:
        """
        Determine if score is high enough for auto-display.

        Args:
            score: Score de matching (0-100)

        Returns:
            True si score > seuil
        """
        return score >= self.AUTO_DISPLAY_THRESHOLD

    def get_meme_image(self, meme_id: str) -> Optional[np.ndarray]:
        """
        Load meme image by ID.

        Args:
            meme_id: ID du meme

        Returns:
            Image BGR ou None si erreur
        """
        if meme_id not in self.meme_database:
            return None

        try:
            image_path = self.meme_folder / self.meme_database[meme_id]['image']
            image = cv2.imread(str(image_path))
            return image
        except Exception as e:
            print(f"Erreur lors du chargement de l'image {meme_id}: {e}")
            return None

    def get_all_meme_names(self) -> List[str]:
        """
        Get list of all meme names in database.

        Returns:
            Liste des noms de memes
        """
        return [data['name'] for data in self.meme_database.values()]

    def close(self):
        """Release resources."""
        self.face_detector.close()
        self.pose_detector.close()

    def __del__(self):
        """Destructor."""
        self.close()
