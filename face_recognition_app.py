import cv2
import numpy as np
import mediapipe as mp
import pickle
import os
from typing import Dict, List, Tuple, Optional

class MediaPipeFaceDatabase:
    """Face database using MediaPipe face detection"""
    
    def __init__(self, db_file: str = "face_database_mp.pkl"):
        self.db_file = db_file
        self.face_data: Dict[str, np.ndarray] = {}
        self.load_database()
    
    def load_database(self) -> None:
        """Load face database from file"""
        try:
            if os.path.exists(self.db_file):
                with open(self.db_file, 'rb') as f:
                    self.face_data = pickle.load(f)
                print(f"Loaded {len(self.face_data)} faces from database")
            else:
                print("No existing database found. Creating new database.")
                self.face_data = {}
        except Exception as e:
            print(f"Error loading database: {e}")
            self.face_data = {}
    
    def save_database(self) -> None:
        """Save face database to file"""
        try:
            with open(self.db_file, 'wb') as f:
                pickle.dump(self.face_data, f)
            print(f"Database saved with {len(self.face_data)} faces")
        except Exception as e:
            print(f"Error saving database: {e}")
    
    def extract_face_features(self, face_image: np.ndarray) -> np.ndarray:
        """Extract simple features from face image"""
        # Convert to grayscale
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image
        
        # Resize to standard size
        resized = cv2.resize(gray, (64, 64))
        
        # Calculate histogram features
        hist = cv2.calcHist([resized], [0], None, [256], [0, 256])
        hist = hist.flatten()
        
        # Normalize
        hist = hist / (hist.sum() + 1e-7)
        
        return hist
    
    def add_face(self, name: str, face_image: np.ndarray) -> bool:
        """Add a new face to the database"""
        try:
            features = self.extract_face_features(face_image)
            self.face_data[name] = features
            self.save_database()
            print(f"Added {name} to database")
            return True
        except Exception as e:
            print(f"Error adding face to database: {e}")
            return False
    
    def find_match(self, face_image: np.ndarray, threshold: float = 0.3) -> Optional[str]:
        """Find a matching face in the database"""
        if not self.face_data:
            return None
        
        try:
            features = self.extract_face_features(face_image)
            
            best_match = None
            best_score = float('inf')
            
            for name, stored_features in self.face_data.items():
                # Calculate correlation coefficient
                score = np.corrcoef(features, stored_features)[0, 1]
                distance = 1 - score  # Convert correlation to distance
                
                if distance < best_score:
                    best_score = distance
                    best_match = name
            
            if best_score < threshold:
                return best_match
            
            return None
        except Exception as e:
            print(f"Error finding match: {e}")
            return None
    
    def get_all_names(self) -> List[str]:
        """Get all names in the database"""
        return list(self.face_data.keys())
    
    def remove_face(self, name: str) -> bool:
        """Remove a face from the database"""
        try:
            if name in self.face_data:
                del self.face_data[name]
                self.save_database()
                print(f"Removed {name} from database")
                return True
            else:
                print(f"{name} not found in database")
                return False
        except Exception as e:
            print(f"Error removing face: {e}")
            return False
    
    def get_database_size(self) -> int:
        """Get the number of faces in database"""
        return len(self.face_data)


class MediaPipeFaceRecognitionApp:
    """Face recognition app using MediaPipe"""
    
    def __init__(self):
        self.database = MediaPipeFaceDatabase()
        self.camera = None
        self.running = False
        
        # Initialize MediaPipe
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5)
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
    
    def initialize_camera(self) -> bool:
        """Initialize the camera"""
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                print("Error: Could not access camera")
                return False
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            print("Camera initialized successfully")
            return True
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using MediaPipe"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        face_locations = []
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                
                # Convert relative coordinates to absolute
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)
                
                # Convert to (top, right, bottom, left) format
                face_locations.append((y, x + w, y + h, x))
        
        return face_locations
    
    def create_face_mesh_preview(self, frame: np.ndarray) -> np.ndarray:
        """Create a preview frame with MediaPipe face mesh and landmarks"""
        preview_frame = frame.copy()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with face mesh
        mesh_results = self.face_mesh.process(rgb_frame)
        
        # Draw face mesh
        if mesh_results.multi_face_landmarks:
            for face_landmarks in mesh_results.multi_face_landmarks:
                # Draw face mesh
                self.mp_drawing.draw_landmarks(
                    preview_frame,
                    face_landmarks,
                    self.mp_face_mesh.FACEMESH_CONTOURS,
                    None,
                    self.mp_drawing_styles.get_default_face_mesh_contours_style())
                
                # Draw face mesh tesselation
                self.mp_drawing.draw_landmarks(
                    preview_frame,
                    face_landmarks,
                    self.mp_face_mesh.FACEMESH_TESSELATION,
                    None,
                    self.mp_drawing_styles.get_default_face_mesh_tesselation_style())
                
                # Draw irises
                self.mp_drawing.draw_landmarks(
                    preview_frame,
                    face_landmarks,
                    self.mp_face_mesh.FACEMESH_IRISES,
                    None,
                    self.mp_drawing_styles.get_default_face_mesh_iris_style())
        
        # Process with face detection for bounding boxes
        detection_results = self.face_detection.process(rgb_frame)
        if detection_results.detections:
            for detection in detection_results.detections:
                self.mp_drawing.draw_detection(preview_frame, detection)
        
        return preview_frame
    
    def extract_face_region(self, frame: np.ndarray, face_location: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract face region from frame"""
        top, right, bottom, left = face_location
        
        # Add some padding
        padding = 20
        top = max(0, top - padding)
        bottom = min(frame.shape[0], bottom + padding)
        left = max(0, left - padding)
        right = min(frame.shape[1], right + padding)
        
        face_image = frame[top:bottom, left:right]
        return face_image
    
    def draw_face_box(self, frame: np.ndarray, face_location: Tuple[int, int, int, int], 
                     name: str, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """Draw a box around the face with name label"""
        top, right, bottom, left = face_location
        
        # Draw rectangle around face
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        
        # Draw label background
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        
        # Draw label text
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)
        
        return frame
    
    def capture_and_add_face(self, name: str) -> bool:
        """Capture a face and add it to the database"""
        if not self.camera:
            print("Camera not initialized")
            return False
        
        print(f"Capturing face for {name}. Please look at the camera...")
        
        # Give user time to position themselves
        for i in range(3, 0, -1):
            ret, frame = self.camera.read()
            if ret:
                cv2.putText(frame, f"Capturing in {i}...", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Face Capture', frame)
                cv2.waitKey(1000)
        
        # Capture the face
        ret, frame = self.camera.read()
        if not ret:
            print("Failed to capture frame")
            return False
        
        # Detect faces
        face_locations = self.detect_faces(frame)
        
        if not face_locations:
            print("No face detected. Please try again.")
            cv2.destroyWindow('Face Capture')
            return False
        
        if len(face_locations) > 1:
            print("Multiple faces detected. Please ensure only one person is in frame.")
            cv2.destroyWindow('Face Capture')
            return False
        
        # Extract face image
        face_image = self.extract_face_region(frame, face_locations[0])
        
        if face_image.size > 0:
            success = self.database.add_face(name, face_image)
            cv2.destroyWindow('Face Capture')
            return success
        
        cv2.destroyWindow('Face Capture')
        return False
    
    def run_recognition(self) -> None:
        """Main recognition loop"""
        if not self.initialize_camera():
            return
        
        self.running = True
        print("Face recognition started. Press 'q' to quit, 'a' to add new face, 'm' to toggle mesh preview.")
        
        frame_count = 0
        process_this_frame = True
        show_mesh_preview = False
        
        while self.running:
            ret, frame = self.camera.read()
            if not ret:
                print("Failed to read from camera")
                break
            
            # Create main recognition frame
            main_frame = frame.copy()
            
            # Process every other frame for better performance
            if process_this_frame:
                # Detect faces
                face_locations = self.detect_faces(frame)
                
                # Process each face
                for face_location in face_locations:
                    # Extract face image
                    face_image = self.extract_face_region(frame, face_location)
                    
                    if face_image.size > 0:
                        # Try to find match in database
                        match_name = self.database.find_match(face_image)
                        
                        if match_name:
                            # Known person
                            message = f"Hello {match_name}!"
                            color = (0, 255, 0)  # Green
                            print(f"Recognized: {match_name}")
                        else:
                            # Unknown person
                            message = "I do not know you. Please introduce yourself."
                            color = (0, 0, 255)  # Red
                            print("Unknown person detected")
                        
                        # Draw face box and message
                        main_frame = self.draw_face_box(main_frame, face_location, message, color)
            
            process_this_frame = not process_this_frame
            
            # Display instructions
            cv2.putText(main_frame, "Press 'q' to quit, 'a' to add face, 'm' to toggle mesh", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(main_frame, f"Database: {self.database.get_database_size()} faces", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show main recognition window
            cv2.imshow('Face Recognition App', main_frame)
            
            # Show MediaPipe face mesh preview if enabled
            if show_mesh_preview:
                mesh_frame = self.create_face_mesh_preview(frame)
                cv2.putText(mesh_frame, "MediaPipe Face Mesh & Landmarks", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(mesh_frame, "Press 'm' to hide this preview", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow('MediaPipe Face Mesh Preview', mesh_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
            elif key == ord('a'):
                self.add_new_person()
            elif key == ord('m'):
                show_mesh_preview = not show_mesh_preview
                if not show_mesh_preview:
                    cv2.destroyWindow('MediaPipe Face Mesh Preview')
                print(f"Face mesh preview: {'ON' if show_mesh_preview else 'OFF'}")
        
        self.cleanup()
    
    def add_new_person(self) -> None:
        """Add a new person to the database"""
        print("\n--- Adding New Person ---")
        name = input("Enter the person's name: ").strip()
        
        if not name:
            print("Name cannot be empty")
            return
        
        if name in self.database.get_all_names():
            print(f"{name} already exists in database")
            return
        
        success = self.capture_and_add_face(name)
        if success:
            print(f"Successfully added {name} to database!")
        else:
            print(f"Failed to add {name} to database")
    
    def cleanup(self) -> None:
        """Clean up resources"""
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        print("Application closed")


def main():
    """Main function"""
    print("=== Face Recognition App with MediaPipe Face Mesh ===")
    print("This app detects faces and matches them with the database.")
    print("If a person is recognized, it says 'Hello [Name]'")
    print("If not recognized, it asks for introduction.")
    print()
    print("🎮 Controls:")
    print("  • Press 'q' to quit")
    print("  • Press 'a' to add new person")
    print("  • Press 'm' to toggle MediaPipe face mesh preview")
    print()
    
    app = MediaPipeFaceRecognitionApp()
    
    try:
        app.run_recognition()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        app.cleanup()


if __name__ == "__main__":
    main()