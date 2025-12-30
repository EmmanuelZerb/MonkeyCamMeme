"""
Expression Analyzer Module
Extraction des features d'expressions faciales depuis les landmarks
"""

import numpy as np
from typing import Dict, Optional, List, Tuple
import math


class ExpressionAnalyzer:
    """Analyzes facial expressions from landmarks and extracts key features."""

    # Indices des landmarks clés (basés sur MediaPipe Face Mesh)
    # Bouche
    UPPER_LIP_TOP = 13
    LOWER_LIP_BOTTOM = 14
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291
    UPPER_LIP_CENTER = 0
    LOWER_LIP_CENTER = 17

    # Yeux
    LEFT_EYE_TOP = 159
    LEFT_EYE_BOTTOM = 145
    RIGHT_EYE_TOP = 386
    RIGHT_EYE_BOTTOM = 374

    # Sourcils
    LEFT_EYEBROW_INNER = 70
    LEFT_EYEBROW_OUTER = 107
    RIGHT_EYEBROW_INNER = 300
    RIGHT_EYEBROW_OUTER = 336
    LEFT_EYE_CENTER = 33
    RIGHT_EYE_CENTER = 263

    # Tête
    NOSE_TIP = 1
    CHIN = 152
    FOREHEAD = 10

    def __init__(self):
        """Initialize expression analyzer."""
        pass

    def calculate_distance(self, point1: Tuple[float, float, float],
                          point2: Tuple[float, float, float]) -> float:
        """
        Calculate Euclidean distance between two 3D points.

        Args:
            point1: Premier point (x, y, z) normalisé
            point2: Deuxième point (x, y, z) normalisé

        Returns:
            Distance euclidienne
        """
        return math.sqrt(
            (point1[0] - point2[0]) ** 2 +
            (point1[1] - point2[1]) ** 2 +
            (point1[2] - point2[2]) ** 2
        )

    def calculate_angle(self, point1: Tuple[float, float],
                       point2: Tuple[float, float],
                       point3: Tuple[float, float]) -> float:
        """
        Calculate angle formed by three points (in degrees).

        Args:
            point1: Premier point (x, y)
            point2: Point central (vertex de l'angle)
            point3: Troisième point (x, y)

        Returns:
            Angle en degrés
        """
        # Vecteurs
        v1 = (point1[0] - point2[0], point1[1] - point2[1])
        v2 = (point3[0] - point2[0], point3[1] - point2[1])

        # Produit scalaire et normes
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        norm_v1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        norm_v2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

        # Éviter division par zéro
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0

        # Calcul de l'angle
        cos_angle = dot_product / (norm_v1 * norm_v2)
        cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp pour éviter erreurs
        angle = math.acos(cos_angle)

        return math.degrees(angle)

    def extract_mouth_features(self, landmarks: List[Tuple[float, float, float]]) -> Dict[str, float]:
        """
        Extract mouth-related features.

        Args:
            landmarks: Liste des 468 landmarks

        Returns:
            Dict contenant:
            - mouth_open_ratio: Ratio d'ouverture de la bouche (0=fermée, 1=très ouverte)
            - mouth_width_ratio: Largeur de la bouche
            - mouth_aspect_ratio: Ratio hauteur/largeur
        """
        try:
            # Calcul de l'ouverture verticale de la bouche
            upper_lip = landmarks[self.UPPER_LIP_TOP]
            lower_lip = landmarks[self.LOWER_LIP_BOTTOM]
            mouth_height = self.calculate_distance(upper_lip, lower_lip)

            # Calcul de la largeur de la bouche
            left_corner = landmarks[self.MOUTH_LEFT]
            right_corner = landmarks[self.MOUTH_RIGHT]
            mouth_width = self.calculate_distance(left_corner, right_corner)

            # Normalisation par rapport à la largeur (ratio)
            mouth_open_ratio = mouth_height / mouth_width if mouth_width > 0 else 0.0

            # Aspect ratio de la bouche
            mouth_aspect_ratio = mouth_height / mouth_width if mouth_width > 0 else 0.0

            return {
                'mouth_open_ratio': min(mouth_open_ratio, 1.0),  # Cap à 1.0
                'mouth_width_ratio': mouth_width,
                'mouth_aspect_ratio': mouth_aspect_ratio
            }

        except Exception as e:
            print(f"Erreur extraction features bouche: {e}")
            return {'mouth_open_ratio': 0.0, 'mouth_width_ratio': 0.0, 'mouth_aspect_ratio': 0.0}

    def extract_eye_features(self, landmarks: List[Tuple[float, float, float]]) -> Dict[str, float]:
        """
        Extract eye-related features.

        Args:
            landmarks: Liste des 468 landmarks

        Returns:
            Dict contenant:
            - left_eye_openness: Ouverture oeil gauche (0=fermé, 1=ouvert)
            - right_eye_openness: Ouverture oeil droit
            - eye_squint: Niveau de plissement des yeux
        """
        try:
            # Oeil gauche
            left_top = landmarks[self.LEFT_EYE_TOP]
            left_bottom = landmarks[self.LEFT_EYE_BOTTOM]
            left_eye_height = self.calculate_distance(left_top, left_bottom)

            # Oeil droit
            right_top = landmarks[self.RIGHT_EYE_TOP]
            right_bottom = landmarks[self.RIGHT_EYE_BOTTOM]
            right_eye_height = self.calculate_distance(right_top, right_bottom)

            # Normalisation approximative (basée sur des valeurs moyennes)
            left_eye_openness = min(left_eye_height * 10, 1.0)
            right_eye_openness = min(right_eye_height * 10, 1.0)

            # Squint = yeux moins ouverts que la normale
            avg_openness = (left_eye_openness + right_eye_openness) / 2
            eye_squint = max(0.0, 1.0 - avg_openness)

            return {
                'left_eye_openness': left_eye_openness,
                'right_eye_openness': right_eye_openness,
                'eye_squint': eye_squint
            }

        except Exception as e:
            print(f"Erreur extraction features yeux: {e}")
            return {'left_eye_openness': 0.5, 'right_eye_openness': 0.5, 'eye_squint': 0.0}

    def extract_eyebrow_features(self, landmarks: List[Tuple[float, float, float]]) -> Dict[str, float]:
        """
        Extract eyebrow-related features.

        Args:
            landmarks: Liste des 468 landmarks

        Returns:
            Dict contenant:
            - left_eyebrow_raise: Niveau de levée sourcil gauche
            - right_eyebrow_raise: Niveau de levée sourcil droit
            - eyebrow_angle: Angle des sourcils (positif=levés, négatif=froncés)
        """
        try:
            # Sourcil gauche
            left_brow_inner = landmarks[self.LEFT_EYEBROW_INNER]
            left_eye = landmarks[self.LEFT_EYE_CENTER]
            left_brow_distance = self.calculate_distance(left_brow_inner, left_eye)

            # Sourcil droit
            right_brow_inner = landmarks[self.RIGHT_EYEBROW_INNER]
            right_eye = landmarks[self.RIGHT_EYE_CENTER]
            right_brow_distance = self.calculate_distance(right_brow_inner, right_eye)

            # Normalisation approximative
            left_eyebrow_raise = min(left_brow_distance * 5, 1.0)
            right_eyebrow_raise = min(right_brow_distance * 5, 1.0)

            # Angle moyen des sourcils
            eyebrow_angle = (left_eyebrow_raise + right_eyebrow_raise) / 2

            return {
                'left_eyebrow_raise': left_eyebrow_raise,
                'right_eyebrow_raise': right_eyebrow_raise,
                'eyebrow_raise': eyebrow_angle
            }

        except Exception as e:
            print(f"Erreur extraction features sourcils: {e}")
            return {'left_eyebrow_raise': 0.5, 'right_eyebrow_raise': 0.5, 'eyebrow_raise': 0.5}

    def extract_head_pose(self, landmarks: List[Tuple[float, float, float]]) -> Dict[str, float]:
        """
        Extract head pose features.

        Args:
            landmarks: Liste des 468 landmarks

        Returns:
            Dict contenant:
            - head_tilt: Inclinaison de la tête (-1 à 1)
            - head_rotation: Rotation de la tête
        """
        try:
            # Utilisation de points clés pour estimer l'inclinaison
            nose = landmarks[self.NOSE_TIP]
            chin = landmarks[self.CHIN]

            # Calcul basique de l'inclinaison (x offset)
            head_tilt = nose[0] - chin[0]

            # Normalisation
            head_tilt = max(-1.0, min(1.0, head_tilt * 2))

            return {
                'head_tilt': head_tilt,
                'head_rotation': abs(head_tilt)  # Valeur absolue pour rotation
            }

        except Exception as e:
            print(f"Erreur extraction head pose: {e}")
            return {'head_tilt': 0.0, 'head_rotation': 0.0}

    def analyze_expression(self, detection_result: Dict) -> Optional[Dict[str, float]]:
        """
        Analyze facial expression and extract all features.

        Args:
            detection_result: Résultat de FaceDetector.detect_face()

        Returns:
            Dict contenant toutes les features extraites, ou None si pas de visage
        """
        if not detection_result or not detection_result['face_found']:
            return None

        landmarks = detection_result['landmarks']

        if len(landmarks) < 468:
            print(f"Erreur: Nombre insuffisant de landmarks ({len(landmarks)})")
            return None

        try:
            # Extraction de toutes les features
            mouth_features = self.extract_mouth_features(landmarks)
            eye_features = self.extract_eye_features(landmarks)
            eyebrow_features = self.extract_eyebrow_features(landmarks)
            head_features = self.extract_head_pose(landmarks)

            # Combinaison de toutes les features
            all_features = {
                **mouth_features,
                **eye_features,
                **eyebrow_features,
                **head_features
            }

            return all_features

        except Exception as e:
            print(f"Erreur lors de l'analyse d'expression: {e}")
            return None

    def features_to_vector(self, features: Dict[str, float]) -> np.ndarray:
        """
        Convert features dict to numpy vector for comparison.

        Args:
            features: Dict des features

        Returns:
            Vecteur numpy des features
        """
        # Ordre fixe des features pour cohérence
        feature_order = [
            'mouth_open_ratio',
            'mouth_aspect_ratio',
            'eye_squint',
            'eyebrow_raise',
            'head_rotation'
        ]

        vector = []
        for key in feature_order:
            vector.append(features.get(key, 0.0))

        return np.array(vector)
