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
            # Angles des bras (poids important)
            'left_arm_angle',
            'right_arm_angle',
            'left_elbow_angle',
            'right_elbow_angle',

            # Angles des jambes
            'left_leg_angle',
            'right_leg_angle',
            'left_knee_angle',
            'right_knee_angle',

            # Position des mains (TRÈS IMPORTANT - ajout de plus de features)
            'left_hand_height',
            'right_hand_height',
            'left_hand_lateral',
            'right_hand_lateral',
            'left_hand_distance_to_face',   # Nouvelle feature
            'right_hand_distance_to_face',  # Nouvelle feature
            'hands_distance',                # Nouvelle feature
            'left_hand_above_head',          # Nouvelle feature
            'right_hand_above_head',         # Nouvelle feature
            'hands_crossed',                 # Nouvelle feature

            # Détails des doigts de la main gauche
            'left_thumb_extended',
            'left_index_extended',
            'left_pinky_extended',

            # Détails des doigts de la main droite
            'right_thumb_extended',
            'right_index_extended',
            'right_pinky_extended',

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

            # Position des mains (features de base)
            features['left_hand_height'] = self._calculate_hand_height(landmarks, 'left')
            features['right_hand_height'] = self._calculate_hand_height(landmarks, 'right')
            features['left_hand_lateral'] = self._calculate_hand_lateral(landmarks, 'left')
            features['right_hand_lateral'] = self._calculate_hand_lateral(landmarks, 'right')

            # Nouvelles features pour les mains (plus de détails)
            features['left_hand_distance_to_face'] = self._calculate_hand_to_face_distance(landmarks, 'left')
            features['right_hand_distance_to_face'] = self._calculate_hand_to_face_distance(landmarks, 'right')
            features['hands_distance'] = self._calculate_hands_distance(landmarks)
            features['left_hand_above_head'] = self._is_hand_above_head(landmarks, 'left')
            features['right_hand_above_head'] = self._is_hand_above_head(landmarks, 'right')
            features['hands_crossed'] = self._are_hands_crossed(landmarks)

            # Position des doigts (main gauche)
            features['left_thumb_extended'] = self._is_finger_extended(landmarks, 'left', 'thumb')
            features['left_index_extended'] = self._is_finger_extended(landmarks, 'left', 'index')
            features['left_pinky_extended'] = self._is_finger_extended(landmarks, 'left', 'pinky')

            # Position des doigts (main droite)
            features['right_thumb_extended'] = self._is_finger_extended(landmarks, 'right', 'thumb')
            features['right_index_extended'] = self._is_finger_extended(landmarks, 'right', 'index')
            features['right_pinky_extended'] = self._is_finger_extended(landmarks, 'right', 'pinky')

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

    def _calculate_hand_to_face_distance(self, landmarks: list, side: str) -> float:
        """Calculate distance between hand and face."""
        wrist_idx = 15 if side == 'left' else 16
        nose_idx = 0

        wrist = landmarks[wrist_idx]
        nose = landmarks[nose_idx]

        distance = np.sqrt((wrist[0] - nose[0])**2 + (wrist[1] - nose[1])**2)
        return float(distance)

    def _calculate_hands_distance(self, landmarks: list) -> float:
        """Calculate distance between both hands."""
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]

        distance = np.sqrt((right_wrist[0] - left_wrist[0])**2 +
                          (right_wrist[1] - left_wrist[1])**2)
        return float(distance)

    def _is_hand_above_head(self, landmarks: list, side: str) -> float:
        """Check if hand is above head (returns 1.0 if yes, 0.0 if no)."""
        wrist_idx = 15 if side == 'left' else 16
        nose_idx = 0

        wrist_y = landmarks[wrist_idx][1]
        nose_y = landmarks[nose_idx][1]

        # Retourne 1.0 si la main est au-dessus de la tête
        return 1.0 if wrist_y < nose_y else 0.0

    def _are_hands_crossed(self, landmarks: list) -> float:
        """Check if hands are crossed (returns 1.0 if yes, 0.0 if no)."""
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]

        # Les mains sont croisées si la main gauche est plus à droite que l'épaule droite
        # ou si la main droite est plus à gauche que l'épaule gauche
        left_crossed = left_wrist[0] > right_shoulder[0]
        right_crossed = right_wrist[0] < left_shoulder[0]

        return 1.0 if (left_crossed or right_crossed) else 0.0

    def _is_finger_extended(self, landmarks: list, side: str, finger: str) -> float:
        """
        Check if a finger is extended.
        Returns distance between finger tip and wrist (normalized).
        """
        wrist_idx = 15 if side == 'left' else 16

        # Indices des doigts
        finger_indices = {
            'thumb': 21 if side == 'left' else 22,
            'index': 19 if side == 'left' else 20,
            'pinky': 17 if side == 'left' else 18
        }

        if finger not in finger_indices:
            return 0.0

        finger_idx = finger_indices[finger]
        wrist = landmarks[wrist_idx]
        finger_tip = landmarks[finger_idx]

        # Distance entre le bout du doigt et le poignet
        distance = np.sqrt((finger_tip[0] - wrist[0])**2 +
                          (finger_tip[1] - wrist[1])**2)

        return float(distance)

    def get_feature_description(self, features: Dict[str, float]) -> str:
        """
        Get human-readable description of the pose with focus on hands.

        Args:
            features: Dictionnaire de features

        Returns:
            Description textuelle de la pose
        """
        descriptions = []

        # Analyse des mains (PRIORITÉ)
        left_hand_above = features.get('left_hand_above_head', 0)
        right_hand_above = features.get('right_hand_above_head', 0)

        if left_hand_above > 0.5 and right_hand_above > 0.5:
            descriptions.append("✋ Les deux mains au-dessus de la tête")
        elif left_hand_above > 0.5:
            descriptions.append("✋ Main gauche levée")
        elif right_hand_above > 0.5:
            descriptions.append("✋ Main droite levée")

        # Mains croisées
        if features.get('hands_crossed', 0) > 0.5:
            descriptions.append("✋ Mains croisées")

        # Distance des mains
        hands_dist = features.get('hands_distance', 0)
        if hands_dist > 0.5:
            descriptions.append("✋ Mains écartées")
        elif hands_dist < 0.2:
            descriptions.append("✋ Mains rapprochées")

        # Distance main-visage
        left_hand_to_face = features.get('left_hand_distance_to_face', 1)
        right_hand_to_face = features.get('right_hand_distance_to_face', 1)

        if left_hand_to_face < 0.2:
            descriptions.append("✋ Main gauche près du visage")
        if right_hand_to_face < 0.2:
            descriptions.append("✋ Main droite près du visage")

        # Analyse des bras
        left_arm = features.get('left_arm_angle', 0)
        right_arm = features.get('right_arm_angle', 0)

        if left_arm > 120 and right_arm > 120:
            descriptions.append("Bras levés")
        elif left_arm < 60 and right_arm < 60:
            descriptions.append("Bras baissés")
        elif abs(left_arm - right_arm) > 40:
            descriptions.append("Bras asymétriques")

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
