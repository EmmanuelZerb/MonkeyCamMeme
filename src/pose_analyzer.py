"""
Pose Analyzer Module
Analyse des poses corporelles et extraction de features caractéristiques
"""

import numpy as np
from typing import Dict, Optional, Tuple
import math


class PoseAnalyzer:
    """Analyzes body pose and extracts characteristic features."""

    def __init__(self):
        """Initialize pose analyzer."""
        # Liste ordonnée des features pour la vectorisation
        self.feature_names = [
            # Angles des bras
            'left_arm_angle',
            'right_arm_angle',
            'left_elbow_angle',
            'right_elbow_angle',

            # Angles des jambes
            'left_leg_angle',
            'right_leg_angle',
            'left_knee_angle',
            'right_knee_angle',

            # Position des mains
            'left_hand_height',
            'right_hand_height',
            'left_hand_lateral',
            'right_hand_lateral',

            # Position globale du corps
            'body_lean_angle',
            'shoulder_tilt',
            'hip_width',
            'shoulder_width',

            # Distances relatives
            'arms_spread',
            'legs_spread',
            'torso_length',
            'body_symmetry'
        ]

    def analyze_pose(self, detection_result: Dict) -> Optional[Dict[str, float]]:
        """
        Analyze body pose and extract key features.

        Args:
            detection_result: Résultat de PoseDetector.detect_pose()

        Returns:
            Dict contenant les features caractéristiques, ou None si erreur
        """
        if not detection_result or not detection_result['pose_found']:
            return None

        try:
            landmarks = detection_result['landmarks']

            if len(landmarks) < 33:
                return None

            features = {}

            # Extraction des angles des bras
            features['left_arm_angle'] = self._calculate_arm_angle(landmarks, 'left')
            features['right_arm_angle'] = self._calculate_arm_angle(landmarks, 'right')
            features['left_elbow_angle'] = self._calculate_elbow_angle(landmarks, 'left')
            features['right_elbow_angle'] = self._calculate_elbow_angle(landmarks, 'right')

            # Extraction des angles des jambes
            features['left_leg_angle'] = self._calculate_leg_angle(landmarks, 'left')
            features['right_leg_angle'] = self._calculate_leg_angle(landmarks, 'right')
            features['left_knee_angle'] = self._calculate_knee_angle(landmarks, 'left')
            features['right_knee_angle'] = self._calculate_knee_angle(landmarks, 'right')

            # Position des mains
            features['left_hand_height'] = self._calculate_hand_height(landmarks, 'left')
            features['right_hand_height'] = self._calculate_hand_height(landmarks, 'right')
            features['left_hand_lateral'] = self._calculate_hand_lateral(landmarks, 'left')
            features['right_hand_lateral'] = self._calculate_hand_lateral(landmarks, 'right')

            # Position globale du corps
            features['body_lean_angle'] = self._calculate_body_lean(landmarks)
            features['shoulder_tilt'] = self._calculate_shoulder_tilt(landmarks)
            features['hip_width'] = self._calculate_hip_width(landmarks)
            features['shoulder_width'] = self._calculate_shoulder_width(landmarks)

            # Distances relatives
            features['arms_spread'] = self._calculate_arms_spread(landmarks)
            features['legs_spread'] = self._calculate_legs_spread(landmarks)
            features['torso_length'] = self._calculate_torso_length(landmarks)
            features['body_symmetry'] = self._calculate_body_symmetry(landmarks)

            return features

        except Exception as e:
            print(f"Erreur lors de l'analyse de la pose: {e}")
            return None

    def _calculate_angle(self, p1: Tuple[float, float], p2: Tuple[float, float],
                        p3: Tuple[float, float]) -> float:
        """
        Calculate angle between three points (p1-p2-p3).

        Args:
            p1, p2, p3: Points as (x, y) tuples

        Returns:
            Angle in degrees (0-180)
        """
        # Vecteurs
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

        # Calcul de l'angle
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0

        cos_angle = dot_product / (norm_v1 * norm_v2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)

        return float(np.degrees(angle))

    def _calculate_arm_angle(self, landmarks: list, side: str) -> float:
        """Calculate angle of arm relative to body (shoulder-elbow-hip)."""
        shoulder_idx = 11 if side == 'left' else 12
        elbow_idx = 13 if side == 'left' else 14
        hip_idx = 23 if side == 'left' else 24

        shoulder = (landmarks[shoulder_idx][0], landmarks[shoulder_idx][1])
        elbow = (landmarks[elbow_idx][0], landmarks[elbow_idx][1])
        hip = (landmarks[hip_idx][0], landmarks[hip_idx][1])

        return self._calculate_angle(hip, shoulder, elbow)

    def _calculate_elbow_angle(self, landmarks: list, side: str) -> float:
        """Calculate elbow bend angle (shoulder-elbow-wrist)."""
        shoulder_idx = 11 if side == 'left' else 12
        elbow_idx = 13 if side == 'left' else 14
        wrist_idx = 15 if side == 'left' else 16

        shoulder = (landmarks[shoulder_idx][0], landmarks[shoulder_idx][1])
        elbow = (landmarks[elbow_idx][0], landmarks[elbow_idx][1])
        wrist = (landmarks[wrist_idx][0], landmarks[wrist_idx][1])

        return self._calculate_angle(shoulder, elbow, wrist)

    def _calculate_leg_angle(self, landmarks: list, side: str) -> float:
        """Calculate leg angle relative to body (hip-knee-shoulder)."""
        hip_idx = 23 if side == 'left' else 24
        knee_idx = 25 if side == 'left' else 26
        shoulder_idx = 11 if side == 'left' else 12

        hip = (landmarks[hip_idx][0], landmarks[hip_idx][1])
        knee = (landmarks[knee_idx][0], landmarks[knee_idx][1])
        shoulder = (landmarks[shoulder_idx][0], landmarks[shoulder_idx][1])

        return self._calculate_angle(shoulder, hip, knee)

    def _calculate_knee_angle(self, landmarks: list, side: str) -> float:
        """Calculate knee bend angle (hip-knee-ankle)."""
        hip_idx = 23 if side == 'left' else 24
        knee_idx = 25 if side == 'left' else 26
        ankle_idx = 27 if side == 'left' else 28

        hip = (landmarks[hip_idx][0], landmarks[hip_idx][1])
        knee = (landmarks[knee_idx][0], landmarks[knee_idx][1])
        ankle = (landmarks[ankle_idx][0], landmarks[ankle_idx][1])

        return self._calculate_angle(hip, knee, ankle)

    def _calculate_hand_height(self, landmarks: list, side: str) -> float:
        """Calculate hand height relative to shoulder (0=shoulder, 1=below, -1=above)."""
        wrist_idx = 15 if side == 'left' else 16
        shoulder_idx = 11 if side == 'left' else 12
        hip_idx = 23 if side == 'left' else 24

        wrist_y = landmarks[wrist_idx][1]
        shoulder_y = landmarks[shoulder_idx][1]
        hip_y = landmarks[hip_idx][1]

        # Normaliser par rapport à la longueur torse
        torso_length = abs(hip_y - shoulder_y)
        if torso_length == 0:
            return 0.0

        hand_relative = (wrist_y - shoulder_y) / torso_length
        return float(hand_relative)

    def _calculate_hand_lateral(self, landmarks: list, side: str) -> float:
        """Calculate hand lateral position relative to shoulder."""
        wrist_idx = 15 if side == 'left' else 16
        shoulder_idx = 11 if side == 'left' else 12
        other_shoulder_idx = 12 if side == 'left' else 11

        wrist_x = landmarks[wrist_idx][0]
        shoulder_x = landmarks[shoulder_idx][0]
        other_shoulder_x = landmarks[other_shoulder_idx][0]

        # Normaliser par rapport à la largeur des épaules
        shoulder_width = abs(other_shoulder_x - shoulder_x)
        if shoulder_width == 0:
            return 0.0

        hand_relative = (wrist_x - shoulder_x) / shoulder_width
        return float(hand_relative)

    def _calculate_body_lean(self, landmarks: list) -> float:
        """Calculate body lean angle from vertical."""
        left_shoulder = landmarks[11]
        left_hip = landmarks[23]

        # Calcul de l'angle avec la verticale
        dx = left_hip[0] - left_shoulder[0]
        dy = left_hip[1] - left_shoulder[1]

        angle = math.atan2(dx, dy)
        return float(np.degrees(angle))

    def _calculate_shoulder_tilt(self, landmarks: list) -> float:
        """Calculate shoulder tilt angle from horizontal."""
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]

        dx = right_shoulder[0] - left_shoulder[0]
        dy = right_shoulder[1] - left_shoulder[1]

        angle = math.atan2(dy, dx)
        return float(np.degrees(angle))

    def _calculate_hip_width(self, landmarks: list) -> float:
        """Calculate normalized hip width."""
        left_hip = landmarks[23]
        right_hip = landmarks[24]

        return float(abs(right_hip[0] - left_hip[0]))

    def _calculate_shoulder_width(self, landmarks: list) -> float:
        """Calculate normalized shoulder width."""
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]

        return float(abs(right_shoulder[0] - left_shoulder[0]))

    def _calculate_arms_spread(self, landmarks: list) -> float:
        """Calculate how spread apart the arms are."""
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]

        distance = np.sqrt((right_wrist[0] - left_wrist[0])**2 +
                          (right_wrist[1] - left_wrist[1])**2)

        return float(distance)

    def _calculate_legs_spread(self, landmarks: list) -> float:
        """Calculate how spread apart the legs are."""
        left_ankle = landmarks[27]
        right_ankle = landmarks[28]

        return float(abs(right_ankle[0] - left_ankle[0]))

    def _calculate_torso_length(self, landmarks: list) -> float:
        """Calculate normalized torso length."""
        left_shoulder = landmarks[11]
        left_hip = landmarks[23]

        distance = np.sqrt((left_hip[0] - left_shoulder[0])**2 +
                          (left_hip[1] - left_shoulder[1])**2)

        return float(distance)

    def _calculate_body_symmetry(self, landmarks: list) -> float:
        """
        Calculate body symmetry score (0=symmetric, 1=asymmetric).

        Compares left and right side angles.
        """
        left_arm = self._calculate_arm_angle(landmarks, 'left')
        right_arm = self._calculate_arm_angle(landmarks, 'right')
        left_leg = self._calculate_leg_angle(landmarks, 'left')
        right_leg = self._calculate_leg_angle(landmarks, 'right')

        arm_diff = abs(left_arm - right_arm) / 180.0
        leg_diff = abs(left_leg - right_leg) / 180.0

        asymmetry = (arm_diff + leg_diff) / 2.0

        return float(asymmetry)

    def features_to_vector(self, features: Dict[str, float]) -> np.ndarray:
        """
        Convert feature dict to numpy vector.

        Args:
            features: Dictionnaire de features

        Returns:
            Vecteur numpy ordonné
        """
        vector = []
        for name in self.feature_names:
            vector.append(features.get(name, 0.0))

        return np.array(vector, dtype=np.float32)

    def get_feature_description(self, features: Dict[str, float]) -> str:
        """
        Get human-readable description of the pose.

        Args:
            features: Dictionnaire de features

        Returns:
            Description textuelle de la pose
        """
        descriptions = []

        # Analyse des bras
        left_arm = features.get('left_arm_angle', 0)
        right_arm = features.get('right_arm_angle', 0)

        if left_arm > 120 and right_arm > 120:
            descriptions.append("Bras levés")
        elif left_arm < 60 and right_arm < 60:
            descriptions.append("Bras baissés")
        elif abs(left_arm - right_arm) > 40:
            descriptions.append("Bras asymétriques")

        # Analyse des mains
        left_hand_height = features.get('left_hand_height', 0)
        right_hand_height = features.get('right_hand_height', 0)

        if left_hand_height < -0.5 or right_hand_height < -0.5:
            descriptions.append("Mains en l'air")

        # Analyse du corps
        lean = features.get('body_lean_angle', 0)
        if abs(lean) > 10:
            direction = "droite" if lean > 0 else "gauche"
            descriptions.append(f"Penché vers la {direction}")

        # Analyse des jambes
        legs_spread = features.get('legs_spread', 0)
        if legs_spread > 0.3:
            descriptions.append("Jambes écartées")

        if not descriptions:
            return "Pose standard"

        return ", ".join(descriptions)
