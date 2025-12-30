"""
Pose Detector Module
Détection de pose corporelle ultra-rapide avec MediaPipe Pose (33 landmarks)
"""

import cv2
import numpy as np
from typing import Optional, List, Dict, Tuple
import urllib.request
import os


class PoseDetector:
    """Detects body pose and extracts 33 pose landmarks using MediaPipe."""

    # Indices des landmarks clés pour l'analyse de poses
    # Basés sur la documentation MediaPipe Pose
    POSE_LANDMARKS = {
        'nose': 0,
        'left_eye_inner': 1,
        'left_eye': 2,
        'left_eye_outer': 3,
        'right_eye_inner': 4,
        'right_eye': 5,
        'right_eye_outer': 6,
        'left_ear': 7,
        'right_ear': 8,
        'mouth_left': 9,
        'mouth_right': 10,
        'left_shoulder': 11,
        'right_shoulder': 12,
        'left_elbow': 13,
        'right_elbow': 14,
        'left_wrist': 15,
        'right_wrist': 16,
        'left_pinky': 17,
        'right_pinky': 18,
        'left_index': 19,
        'right_index': 20,
        'left_thumb': 21,
        'right_thumb': 22,
        'left_hip': 23,
        'right_hip': 24,
        'left_knee': 25,
        'right_knee': 26,
        'left_ankle': 27,
        'right_ankle': 28,
        'left_heel': 29,
        'right_heel': 30,
        'left_foot_index': 31,
        'right_foot_index': 32
    }

    # Connexions pour dessiner le squelette
    POSE_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 7),  # Visage gauche
        (0, 4), (4, 5), (5, 6), (6, 8),  # Visage droit
        (9, 10),  # Bouche
        (11, 12),  # Épaules
        (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),  # Bras gauche
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),  # Bras droit
        (11, 23), (12, 24), (23, 24),  # Torse
        (23, 25), (25, 27), (27, 29), (27, 31),  # Jambe gauche
        (24, 26), (26, 28), (28, 30), (28, 32),  # Jambe droit
    ]

    def __init__(self, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize pose detector.

        Args:
            min_detection_confidence: Confiance minimale pour la détection (0-1)
            min_tracking_confidence: Confiance minimale pour le tracking (0-1)
        """
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python import BaseOptions

        # Download model if not exists
        model_path = self._download_model()

        # Configure pose landmarker options
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False
        )

        # Create pose landmarker
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(options)
        self.frame_timestamp = 0

    def _download_model(self) -> str:
        """Download pose landmarker model if not exists."""
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, 'pose_landmarker_lite.task')

        if not os.path.exists(model_path):
            print("Downloading pose landmarker model...")
            url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
            urllib.request.urlretrieve(url, model_path)
            print("Pose model downloaded successfully!")

        return model_path

    def detect_pose(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Detect pose and extract landmarks from frame.

        Args:
            frame: Image BGR depuis OpenCV

        Returns:
            Dict contenant les landmarks et métadonnées, ou None si pas de pose détectée
            Structure: {
                'landmarks': List[Tuple[float, float, float, float]],  # (x, y, z, visibility)
                'image_shape': Tuple[int, int, int],  # (height, width, channels)
                'pose_found': bool
            }
        """
        if frame is None:
            return None

        try:
            from mediapipe import Image, ImageFormat

            # Conversion BGR vers RGB (requis par MediaPipe)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create MediaPipe Image
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)

            # Process avec MediaPipe (increment timestamp for video mode)
            self.frame_timestamp += 1
            results = self.pose_landmarker.detect_for_video(mp_image, self.frame_timestamp)

            if not results.pose_landmarks:
                return {
                    'landmarks': [],
                    'image_shape': frame.shape,
                    'pose_found': False
                }

            # Extraction des landmarks du premier corps détecté
            pose_landmarks = results.pose_landmarks[0]

            # Conversion en format utilisable (liste de tuples)
            landmarks = []
            for landmark in pose_landmarks:
                landmarks.append((landmark.x, landmark.y, landmark.z, landmark.visibility))

            return {
                'landmarks': landmarks,
                'image_shape': frame.shape,
                'pose_found': True,
                'raw_landmarks': pose_landmarks  # Pour le dessin si nécessaire
            }

        except Exception as e:
            print(f"Erreur lors de la détection de la pose: {e}")
            import traceback
            traceback.print_exc()
            return None

    def draw_pose(self, frame: np.ndarray, detection_result: Dict) -> np.ndarray:
        """
        Draw pose landmarks on frame.

        Args:
            frame: Image BGR depuis OpenCV
            detection_result: Résultat de detect_pose()

        Returns:
            Frame avec pose dessinée
        """
        if not detection_result or not detection_result['pose_found']:
            return frame

        try:
            landmarks = detection_result['landmarks']
            height, width = detection_result['image_shape'][:2]

            # Convertir les landmarks en coordonnées pixel
            landmark_points = []
            for landmark in landmarks:
                x = int(landmark[0] * width)
                y = int(landmark[1] * height)
                visibility = landmark[3]
                landmark_points.append((x, y, visibility))

            # Dessiner les connexions (squelette)
            for connection in self.POSE_CONNECTIONS:
                start_idx, end_idx = connection
                if start_idx < len(landmark_points) and end_idx < len(landmark_points):
                    start_point = landmark_points[start_idx]
                    end_point = landmark_points[end_idx]

                    # Ne dessiner que si les deux points sont visibles
                    if start_point[2] > 0.5 and end_point[2] > 0.5:
                        cv2.line(frame,
                                (start_point[0], start_point[1]),
                                (end_point[0], end_point[1]),
                                (0, 255, 0), 2)

            # Dessiner les landmarks comme des cercles
            for point in landmark_points:
                if point[2] > 0.5:  # Seulement si visible
                    cv2.circle(frame, (point[0], point[1]), 4, (0, 0, 255), -1)

            return frame

        except Exception as e:
            print(f"Erreur lors du dessin de la pose: {e}")
            return frame

    def get_landmark_coords(self, detection_result: Dict,
                           landmark_name: str) -> Optional[Tuple[int, int]]:
        """
        Get pixel coordinates of a specific landmark by name.

        Args:
            detection_result: Résultat de detect_pose()
            landmark_name: Nom du landmark (ex: 'left_wrist')

        Returns:
            Tuple (x, y) en coordonnées pixel, ou None si invalide
        """
        if not detection_result or not detection_result['pose_found']:
            return None

        if landmark_name not in self.POSE_LANDMARKS:
            return None

        landmark_index = self.POSE_LANDMARKS[landmark_name]
        landmarks = detection_result['landmarks']

        if landmark_index >= len(landmarks):
            return None

        height, width = detection_result['image_shape'][:2]

        # Conversion des coordonnées normalisées vers pixel
        x = int(landmarks[landmark_index][0] * width)
        y = int(landmarks[landmark_index][1] * height)

        return x, y

    def close(self):
        """Release MediaPipe resources."""
        if hasattr(self, 'pose_landmarker') and self.pose_landmarker:
            self.pose_landmarker.close()
            print("Pose detector fermé")

    def __del__(self):
        """Destructor pour libérer les ressources."""
        self.close()
