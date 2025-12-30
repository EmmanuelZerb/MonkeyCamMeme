"""
Face Detector Module
Détection faciale ultra-rapide avec MediaPipe Face Mesh (468 landmarks)
"""

import cv2
import numpy as np
from typing import Optional, List, Dict, Tuple
import urllib.request
import os


class FaceDetector:
    """Detects faces and extracts 468 facial landmarks using MediaPipe."""

    # Indices des landmarks clés pour l'analyse d'expressions
    # Basés sur la documentation MediaPipe Face Mesh
    MOUTH_LANDMARKS = {
        'upper_lip': [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
        'lower_lip': [146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
        'mouth_corners': [61, 291],
    }

    EYE_LANDMARKS = {
        'left_eye': [33, 160, 158, 133, 153, 144],
        'right_eye': [362, 385, 387, 263, 373, 380],
    }

    EYEBROW_LANDMARKS = {
        'left_eyebrow': [70, 63, 105, 66, 107],
        'right_eyebrow': [300, 293, 334, 296, 336],
    }

    FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

    def __init__(self, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize face detector.

        Args:
            min_detection_confidence: Confiance minimale pour la détection (0-1)
            min_tracking_confidence: Confiance minimale pour le tracking (0-1)
        """
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python import BaseOptions

        # Download model if not exists
        model_path = self._download_model()

        # Configure face landmarker options
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )

        # Create face landmarker
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
        self.frame_timestamp = 0

    def _download_model(self) -> str:
        """Download face landmarker model if not exists."""
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, 'face_landmarker.task')

        if not os.path.exists(model_path):
            print("Downloading face landmarker model...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
            print("Model downloaded successfully!")

        return model_path

    def detect_face(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Detect face and extract landmarks from frame.

        Args:
            frame: Image BGR depuis OpenCV

        Returns:
            Dict contenant les landmarks et métadonnées, ou None si pas de visage détecté
            Structure: {
                'landmarks': List[Tuple[float, float, float]],  # (x, y, z) normalisés
                'image_shape': Tuple[int, int, int],  # (height, width, channels)
                'face_found': bool
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
            results = self.face_landmarker.detect_for_video(mp_image, self.frame_timestamp)

            if not results.face_landmarks:
                return {
                    'landmarks': [],
                    'image_shape': frame.shape,
                    'face_found': False
                }

            # Extraction des landmarks du premier visage détecté
            face_landmarks = results.face_landmarks[0]

            # Conversion en format utilisable (liste de tuples)
            landmarks = []
            for landmark in face_landmarks:
                landmarks.append((landmark.x, landmark.y, landmark.z))

            return {
                'landmarks': landmarks,
                'image_shape': frame.shape,
                'face_found': True,
                'raw_landmarks': face_landmarks  # Pour le dessin si nécessaire
            }

        except Exception as e:
            print(f"Erreur lors de la détection du visage: {e}")
            import traceback
            traceback.print_exc()
            return None

    def draw_landmarks(self, frame: np.ndarray, detection_result: Dict) -> np.ndarray:
        """
        Draw facial landmarks on frame.

        Args:
            frame: Image BGR depuis OpenCV
            detection_result: Résultat de detect_face()

        Returns:
            Frame avec landmarks dessinés
        """
        if not detection_result or not detection_result['face_found']:
            return frame

        try:
            # Draw landmarks manually
            landmarks = detection_result['landmarks']
            height, width = detection_result['image_shape'][:2]

            # Draw all landmarks as small circles
            for landmark in landmarks:
                x = int(landmark[0] * width)
                y = int(landmark[1] * height)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            # Draw face oval
            for i in range(len(self.FACE_OVAL)):
                start_idx = self.FACE_OVAL[i]
                end_idx = self.FACE_OVAL[(i + 1) % len(self.FACE_OVAL)]

                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    start_point = (
                        int(landmarks[start_idx][0] * width),
                        int(landmarks[start_idx][1] * height)
                    )
                    end_point = (
                        int(landmarks[end_idx][0] * width),
                        int(landmarks[end_idx][1] * height)
                    )
                    cv2.line(frame, start_point, end_point, (255, 255, 255), 1)

            return frame

        except Exception as e:
            print(f"Erreur lors du dessin des landmarks: {e}")
            return frame

    def get_landmark_coords(self, detection_result: Dict,
                           landmark_index: int) -> Optional[Tuple[int, int]]:
        """
        Get pixel coordinates of a specific landmark.

        Args:
            detection_result: Résultat de detect_face()
            landmark_index: Index du landmark (0-467)

        Returns:
            Tuple (x, y) en coordonnées pixel, ou None si invalide
        """
        if not detection_result or not detection_result['face_found']:
            return None

        if landmark_index >= len(detection_result['landmarks']):
            return None

        landmarks = detection_result['landmarks']
        height, width = detection_result['image_shape'][:2]

        # Conversion des coordonnées normalisées vers pixel
        x = int(landmarks[landmark_index][0] * width)
        y = int(landmarks[landmark_index][1] * height)

        return x, y

    def get_multiple_landmark_coords(self, detection_result: Dict,
                                     landmark_indices: List[int]) -> List[Tuple[int, int]]:
        """
        Get pixel coordinates for multiple landmarks.

        Args:
            detection_result: Résultat de detect_face()
            landmark_indices: Liste d'indices des landmarks

        Returns:
            Liste de tuples (x, y) en coordonnées pixel
        """
        coords = []
        for idx in landmark_indices:
            coord = self.get_landmark_coords(detection_result, idx)
            if coord:
                coords.append(coord)

        return coords

    def close(self):
        """Release MediaPipe resources."""
        if hasattr(self, 'face_landmarker') and self.face_landmarker:
            self.face_landmarker.close()
            print("Face detector fermé")

    def __del__(self):
        """Destructor pour libérer les ressources."""
        self.close()
