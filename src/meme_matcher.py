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
from .hand_detector import HandDetector


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
        self.hand_detector = HandDetector()

        # Seuil pour affichage automatique du meme
        self.AUTO_DISPLAY_THRESHOLD = 65.0  # Seuil ajusté pour meilleur matching

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

                # Détection du visage, de la pose ET des mains
                face_detection = self.face_detector.detect_face(image)
                pose_detection = self.pose_detector.detect_pose(image)
                hands_detection = self.hand_detector.detect_hands(image)

                # Au moins un des trois doit être détecté
                if (not face_detection or not face_detection['face_found']) and \
                   (not pose_detection or not pose_detection['pose_found']) and \
                   (not hands_detection or not hands_detection['hands_found']):
                    print(f"Aucun visage, pose ou mains détectés dans {meme_path.name}")
                    continue

                # Analyse de l'expression faciale
                face_features = None
                if face_detection and face_detection['face_found']:
                    face_features = self.expression_analyzer.analyze_expression(face_detection)

                # Analyse de la pose corporelle
                pose_features = None
                if pose_detection and pose_detection['pose_found']:
                    pose_features = self.pose_analyzer.analyze_pose(pose_detection)

                # Analyse des mains (nouvellement ajouté)
                hands_features = None
                if hands_detection and hands_detection['hands_found']:
                    hands_features = self._analyze_hands(hands_detection)

                if face_features is None and pose_features is None and hands_features is None:
                    print(f"Erreur d'analyse pour {meme_path.name}")
                    continue

                # Création de l'ID du meme (nom sans extension)
                meme_id = meme_path.stem
                meme_name = meme_id.replace('_', ' ').title()

                # Ajout à la base de données avec les trois types de features
                self.meme_database[meme_id] = {
                    'name': meme_name,
                    'image': meme_path.name,
                    'face_features': face_features,
                    'pose_features': pose_features,
                    'hands_features': hands_features
                }

                detection_types = []
                if face_features:
                    detection_types.append("visage")
                if pose_features:
                    detection_types.append("pose")
                if hands_features:
                    detection_types.append("mains")

                print(f"{meme_name} analyse ({', '.join(detection_types)})")

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

    def _analyze_hands(self, hands_detection: Dict) -> Optional[Dict[str, float]]:
        """
        Analyze hands detection and extract features.

        Args:
            hands_detection: Résultat de HandDetector.detect_hands()

        Returns:
            Dict contenant les features des mains
        """
        if not hands_detection or not hands_detection['hands_found']:
            return None

        features = {}

        # Initialiser toutes les features à 0
        features['left_hand_present'] = 0.0
        features['right_hand_present'] = 0.0
        features['left_fingers_extended'] = 0.0
        features['right_fingers_extended'] = 0.0
        features['both_hands_present'] = 0.0
        features['hands_distance'] = 0.0

        left_hand = None
        right_hand = None

        # Séparer les mains gauche et droite
        for hand_data in hands_detection['hands']:
            if hand_data['handedness'] == 'Left':
                left_hand = hand_data
            elif hand_data['handedness'] == 'Right':
                right_hand = hand_data

        # Analyser la main gauche
        if left_hand:
            features['left_hand_present'] = 1.0
            features['left_fingers_extended'] = self.hand_detector.count_extended_fingers(left_hand['landmarks']) / 5.0

        # Analyser la main droite
        if right_hand:
            features['right_hand_present'] = 1.0
            features['right_fingers_extended'] = self.hand_detector.count_extended_fingers(right_hand['landmarks']) / 5.0

        # Les deux mains présentes
        if left_hand and right_hand:
            features['both_hands_present'] = 1.0
            # Distance entre les mains
            left_wrist = left_hand['landmarks'][0]
            right_wrist = right_hand['landmarks'][0]
            distance = np.sqrt((right_wrist[0] - left_wrist[0])**2 + (right_wrist[1] - left_wrist[1])**2)
            features['hands_distance'] = float(distance)

        return features

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
                       current_pose_features: Optional[Dict[str, float]],
                       current_hands_features: Optional[Dict[str, float]] = None) -> Optional[Tuple[str, str, float, str]]:
        """
        Find best matching meme for current expression, pose and hands.

        Args:
            current_face_features: Features de l'expression faciale actuelle
            current_pose_features: Features de la pose corporelle actuelle
            current_hands_features: Features des mains actuelles

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

        current_hands_vector = None
        if current_hands_features:
            current_hands_vector = self._hands_features_to_vector(current_hands_features)

        if current_face_vector is None and current_pose_vector is None and current_hands_vector is None:
            return None

        best_match = None
        best_score = 0.0

        # Comparaison avec tous les memes
        for meme_id, meme_data in self.meme_database.items():
            # Calcul du score combiné (visage + mains prioritaires, pose secondaire)
            total_score = 0.0
            weight_sum = 0.0

            # Score du visage (poids 0.4)
            if current_face_vector is not None and meme_data.get('face_features'):
                meme_face_vector = self.expression_analyzer.features_to_vector(meme_data['face_features'])
                face_similarity = self.calculate_cosine_similarity(current_face_vector, meme_face_vector)
                total_score += face_similarity * 0.4
                weight_sum += 0.4

            # Score de la pose (poids 0.2 - pour supporter les images sans visage/mains)
            if current_pose_vector is not None and meme_data.get('pose_features'):
                meme_pose_vector = self.pose_analyzer.features_to_vector(meme_data['pose_features'])
                pose_similarity = self.calculate_cosine_similarity(current_pose_vector, meme_pose_vector)
                total_score += pose_similarity * 0.2
                weight_sum += 0.2

            # Score des mains (poids 0.4)
            if current_hands_vector is not None and meme_data.get('hands_features'):
                meme_hands_vector = self._hands_features_to_vector(meme_data['hands_features'])
                hands_similarity = self.calculate_cosine_similarity(current_hands_vector, meme_hands_vector)
                total_score += hands_similarity * 0.4
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

    def _hands_features_to_vector(self, features: Dict[str, float]) -> np.ndarray:
        """Convert hands features dict to numpy vector."""
        feature_names = [
            'left_hand_present',
            'right_hand_present',
            'left_fingers_extended',
            'right_fingers_extended',
            'both_hands_present',
            'hands_distance'
        ]

        vector = []
        for name in feature_names:
            vector.append(features.get(name, 0.0))

        return np.array(vector, dtype=np.float32)

    def close(self):
        """Release resources."""
        self.face_detector.close()
        self.pose_detector.close()
        self.hand_detector.close()

    def __del__(self):
        """Destructor."""
        self.close()
