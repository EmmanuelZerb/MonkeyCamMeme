"""
Camera Handler Module
Gère la capture vidéo depuis la webcam avec OpenCV
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class CameraHandler:
    """Handles webcam capture and frame processing."""

    def __init__(self, camera_index: int = 0, fps: int = 30):
        """
        Initialize camera handler.

        Args:
            camera_index: Index de la caméra (0 par défaut pour webcam principale)
            fps: Frames per second cibles
        """
        self.camera_index = camera_index
        self.target_fps = fps
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False

    def start(self) -> bool:
        """
        Start camera capture.

        Returns:
            True si la caméra a été initialisée avec succès, False sinon
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_index)

            if not self.cap.isOpened():
                print(f"Erreur: Impossible d'ouvrir la caméra {self.camera_index}")
                return False

            # Configuration de la caméra pour performance optimale
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

            self.is_running = True
            print(f"Caméra {self.camera_index} initialisée avec succès")
            return True

        except Exception as e:
            print(f"Erreur lors de l'initialisation de la caméra: {e}")
            return False

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a single frame from camera.

        Returns:
            Tuple (success, frame) où success indique si la lecture a réussi
            et frame contient l'image capturée (BGR format)
        """
        if not self.is_running or self.cap is None:
            return False, None

        try:
            ret, frame = self.cap.read()

            if not ret:
                print("Erreur: Impossible de lire la frame")
                return False, None

            # Flip horizontal pour effet miroir (plus naturel pour l'utilisateur)
            frame = cv2.flip(frame, 1)

            return True, frame

        except Exception as e:
            print(f"Erreur lors de la lecture de la frame: {e}")
            return False, None

    def get_frame_dimensions(self) -> Tuple[int, int]:
        """
        Get current frame dimensions.

        Returns:
            Tuple (width, height) des dimensions de la frame
        """
        if self.cap is None:
            return 640, 480

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        return width, height

    def get_actual_fps(self) -> float:
        """
        Get actual FPS from camera.

        Returns:
            FPS actuels de la caméra
        """
        if self.cap is None:
            return 0.0

        return self.cap.get(cv2.CAP_PROP_FPS)

    def release(self) -> None:
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()
            self.is_running = False
            print("Caméra libérée")

    def __del__(self):
        """Destructor pour s'assurer que la caméra est libérée."""
        self.release()
