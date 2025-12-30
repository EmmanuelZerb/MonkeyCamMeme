"""
UI Manager Module
Interface PyQt5 moderne pour MemeMotion
"""

import cv2
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QLabel, QPushButton, QFrame, QMessageBox)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont
from typing import Optional, Tuple

from .camera_handler import CameraHandler
from .face_detector import FaceDetector
from .expression_analyzer import ExpressionAnalyzer
from .pose_detector import PoseDetector
from .pose_analyzer import PoseAnalyzer
from .hand_detector import HandDetector
from .meme_matcher import MemeMatcher


class MemeMotionUI(QMainWindow):
    """Main window for MemeMotion application."""

    def __init__(self):
        """Initialize the main window."""
        super().__init__()

        # Configuration de la fenêtre
        self.setWindowTitle("MemeMotion")
        self.setGeometry(100, 100, 1200, 700)

        # Initialisation des composants
        self.camera_handler: Optional[CameraHandler] = None
        self.face_detector: Optional[FaceDetector] = None
        self.expression_analyzer: Optional[ExpressionAnalyzer] = None
        self.pose_detector: Optional[PoseDetector] = None
        self.pose_analyzer: Optional[PoseAnalyzer] = None
        self.hand_detector: Optional[HandDetector] = None
        self.meme_matcher: Optional[MemeMatcher] = None

        # État de l'application
        self.current_frame: Optional[np.ndarray] = None
        self.current_meme_image: Optional[np.ndarray] = None
        self.current_match: Optional[Tuple] = None
        self.last_displayed_meme: Optional[str] = None  # Pour éviter les affichages répétés

        # Timer pour le refresh
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Construction de l'UI
        self.init_ui()

    def init_ui(self):
        """Initialize UI components."""
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Layout principal
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # === SECTION VIDEO ===
        video_layout = QHBoxLayout()

        # Frame gauche: Webcam de l'utilisateur
        self.user_frame = QFrame()
        self.user_frame.setFrameStyle(QFrame.Box | QFrame.Sunken)
        self.user_frame.setLineWidth(2)
        user_layout = QVBoxLayout()
        self.user_frame.setLayout(user_layout)

        user_title = QLabel("Vous")
        user_title.setAlignment(Qt.AlignCenter)
        user_title.setFont(QFont("Arial", 12))
        user_title.setStyleSheet("color: #888; padding: 5px;")
        user_layout.addWidget(user_title)

        self.user_video_label = QLabel()
        self.user_video_label.setAlignment(Qt.AlignCenter)
        self.user_video_label.setMinimumSize(500, 400)
        self.user_video_label.setStyleSheet("background-color: #1a1a1a;")
        user_layout.addWidget(self.user_video_label)

        video_layout.addWidget(self.user_frame)

        # Frame droite: Meme de référence
        self.meme_frame = QFrame()
        self.meme_frame.setFrameStyle(QFrame.Box | QFrame.Sunken)
        self.meme_frame.setLineWidth(2)
        meme_layout = QVBoxLayout()
        self.meme_frame.setLayout(meme_layout)

        meme_title = QLabel("Match")
        meme_title.setAlignment(Qt.AlignCenter)
        meme_title.setFont(QFont("Arial", 12))
        meme_title.setStyleSheet("color: #888; padding: 5px;")
        meme_layout.addWidget(meme_title)

        self.meme_video_label = QLabel()
        self.meme_video_label.setAlignment(Qt.AlignCenter)
        self.meme_video_label.setMinimumSize(500, 400)
        self.meme_video_label.setStyleSheet("background-color: #1a1a1a;")
        meme_layout.addWidget(self.meme_video_label)

        video_layout.addWidget(self.meme_frame)

        main_layout.addLayout(video_layout)

        # === SECTION INFO ===
        info_layout = QHBoxLayout()

        # Label du nom du meme
        self.meme_name_label = QLabel("")
        self.meme_name_label.setAlignment(Qt.AlignCenter)
        self.meme_name_label.setFont(QFont("Arial", 16))
        self.meme_name_label.setStyleSheet("color: #aaa; padding: 10px;")
        info_layout.addWidget(self.meme_name_label)

        main_layout.addLayout(info_layout)

        # === SECTION BOUTONS ===
        button_layout = QHBoxLayout()

        # Bouton Quitter
        self.quit_button = QPushButton("Quitter")
        self.quit_button.setFont(QFont("Arial", 11))
        self.quit_button.setMinimumHeight(40)
        self.quit_button.clicked.connect(self.close_application)
        self.quit_button.setStyleSheet("""
            QPushButton {
                background-color: #333;
                color: #aaa;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #444;
                color: #fff;
            }
            QPushButton:pressed {
                background-color: #222;
            }
        """)
        button_layout.addWidget(self.quit_button)

        main_layout.addLayout(button_layout)

        # Label de notification
        self.notification_label = QLabel("")
        self.notification_label.setAlignment(Qt.AlignCenter)
        self.notification_label.setFont(QFont("Arial", 10))
        self.notification_label.setStyleSheet("color: #666; padding: 5px;")
        main_layout.addWidget(self.notification_label)

        # Style général de la fenêtre
        self.setStyleSheet("QMainWindow { background-color: #1e1e1e; }")

    def initialize_components(self, meme_folder: str, metadata_file: str) -> bool:
        """
        Initialize all components (camera, detectors, etc.).

        Args:
            meme_folder: Dossier des memes
            metadata_file: Fichier de métadonnées

        Returns:
            True si initialisation réussie
        """
        try:
            # Initialisation de la caméra
            self.camera_handler = CameraHandler(camera_index=0, fps=30)
            if not self.camera_handler.start():
                QMessageBox.critical(self, "Erreur",
                                   "Impossible d'initialiser la caméra!")
                return False

            # Initialisation du détecteur de visage
            self.face_detector = FaceDetector(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

            # Initialisation de l'analyseur d'expressions
            self.expression_analyzer = ExpressionAnalyzer()

            # Initialisation du détecteur de pose
            self.pose_detector = PoseDetector(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

            # Initialisation de l'analyseur de poses
            self.pose_analyzer = PoseAnalyzer()

            # Initialisation du détecteur de mains
            self.hand_detector = HandDetector(
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3
            )

            # Initialisation du matcher de memes
            self.meme_matcher = MemeMatcher(meme_folder, metadata_file)
            if not self.meme_matcher.load_or_generate_metadata():
                QMessageBox.warning(self, "Avertissement",
                                  "Aucun meme chargé. Ajoutez des images dans assets/memes/")

            # Démarrage du timer de refresh (30 FPS = ~33ms)
            self.timer.start(33)

            return True

        except Exception as e:
            QMessageBox.critical(self, "Erreur d'initialisation", str(e))
            return False

    def update_frame(self):
        """Update video frames and perform matching."""
        if not self.camera_handler:
            return

        # Lecture de la frame
        success, frame = self.camera_handler.read_frame()
        if not success or frame is None:
            return

        self.current_frame = frame.copy()

        # Détection du visage, pose ET mains (mais on affiche seulement visage + mains)
        face_detection = self.face_detector.detect_face(frame)
        pose_detection = self.pose_detector.detect_pose(frame)
        hands_detection = self.hand_detector.detect_hands(frame)

        # Dessin des détections sur la frame
        frame_annotated = frame.copy()

        # Analyse des features
        face_features = None
        pose_features = None

        if face_detection and face_detection['face_found']:
            # Dessin des landmarks faciaux
            frame_annotated = self.face_detector.draw_landmarks(frame_annotated, face_detection)
            # Analyse de l'expression
            face_features = self.expression_analyzer.analyze_expression(face_detection)

        # Analyse de la pose (pour le matching seulement, pas d'affichage)
        if pose_detection and pose_detection['pose_found']:
            pose_features = self.pose_analyzer.analyze_pose(pose_detection)

        # Dessin des mains
        if hands_detection and hands_detection['hands_found']:
            frame_annotated = self.hand_detector.draw_hands(frame_annotated, hands_detection)

        # Analyser les mains pour le matching
        hands_features = None
        if hands_detection and hands_detection['hands_found']:
            hands_features = self.meme_matcher._analyze_hands(hands_detection)

        # Si au moins visage OU mains sont détectés
        if face_features is not None or hands_features is not None:
            # Matching avec les memes (incluant la pose en arrière-plan)
            match = self.meme_matcher.find_best_match(face_features, pose_features, hands_features)

            if match:
                meme_id, meme_name, score, meme_path = match
                self.current_match = match

                self.meme_name_label.setText(meme_name)

                # Chargement et affichage automatique du meme si score suffisant
                if self.meme_matcher.should_auto_display(score):
                    # Ne recharger que si c'est un nouveau meme
                    if self.last_displayed_meme != meme_id:
                        meme_image = self.meme_matcher.get_meme_image(meme_id)
                        if meme_image is not None:
                            self.current_meme_image = meme_image
                            self.display_meme_image(meme_image)
                            self.last_displayed_meme = meme_id
                            self.show_notification(meme_name)
                else:
                    # Si le score est trop bas, effacer le meme affiché
                    if self.last_displayed_meme is not None:
                        self.clear_meme_display()
                        self.last_displayed_meme = None

            # Affichage de la frame utilisateur
            self.display_user_frame(frame_annotated)
        else:
            # Aucune détection - afficher la frame sans annotations et sans meme
            self.display_user_frame(frame)
            self.meme_name_label.setText("")
            if self.last_displayed_meme is not None:
                self.clear_meme_display()
                self.last_displayed_meme = None

    def display_user_frame(self, frame: np.ndarray):
        """Display user's webcam frame."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        # Redimensionner pour s'adapter au label
        scaled_pixmap = pixmap.scaled(self.user_video_label.size(),
                                      Qt.KeepAspectRatio,
                                      Qt.SmoothTransformation)
        self.user_video_label.setPixmap(scaled_pixmap)

    def display_meme_image(self, meme_image: np.ndarray):
        """Display matching meme image."""
        rgb_image = cv2.cvtColor(meme_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        # Redimensionner pour s'adapter au label
        scaled_pixmap = pixmap.scaled(self.meme_video_label.size(),
                                      Qt.KeepAspectRatio,
                                      Qt.SmoothTransformation)
        self.meme_video_label.setPixmap(scaled_pixmap)

    def clear_meme_display(self):
        """Clear meme display."""
        self.meme_video_label.clear()
        self.meme_video_label.setStyleSheet("background-color: #1a1a1a;")

    def show_notification(self, message: str):
        """Show temporary notification."""
        self.notification_label.setText(message)
        # Clear notification après 3 secondes
        QTimer.singleShot(3000, lambda: self.notification_label.setText(""))

    def close_application(self):
        """Close application properly."""
        reply = QMessageBox.question(self, 'Quitter',
                                    'Quitter?',
                                    QMessageBox.Yes | QMessageBox.No,
                                    QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.close()

    def closeEvent(self, event):
        """Handle window close event."""
        # Arrêt du timer
        self.timer.stop()

        # Libération des ressources
        if self.camera_handler:
            self.camera_handler.release()

        if self.face_detector:
            self.face_detector.close()

        if self.pose_detector:
            self.pose_detector.close()

        if self.hand_detector:
            self.hand_detector.close()

        if self.meme_matcher:
            self.meme_matcher.close()

        event.accept()
