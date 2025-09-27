import cv2
import numpy as np
import mediapipe as mp
import pickle
import os
from typing import Dict, List, Tuple, Optional

class MediaPipeFaceDatabase:
    """Face database using MediaPipe face detection with multi-angle support"""
    
    def __init__(self, db_file: str = "face_database_mp.pkl"):
        self.db_file = db_file
        # Changed to store multiple face samples per person: Dict[name, List[features]]
        self.face_data: Dict[str, List[np.ndarray]] = {}
        self.load_database()
    
    def load_database(self) -> None:
        """Load face database from file"""
        try:
            if os.path.exists(self.db_file):
                with open(self.db_file, 'rb') as f:
                    loaded_data = pickle.load(f)
                    
                    # Handle backward compatibility - convert old format to new
                    if loaded_data and isinstance(list(loaded_data.values())[0], np.ndarray):
                        # Old format: Dict[str, np.ndarray] -> convert to Dict[str, List[np.ndarray]]
                        self.face_data = {name: [features] for name, features in loaded_data.items()}
                        print(f"Converted old database format. Loaded {len(self.face_data)} people from database")
                    else:
                        # New format: Dict[str, List[np.ndarray]]
                        self.face_data = loaded_data
                        total_samples = sum(len(samples) for samples in self.face_data.values())
                        print(f"Loaded {len(self.face_data)} people with {total_samples} total face samples")
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
            total_samples = sum(len(samples) for samples in self.face_data.values())
            print(f"Database saved with {len(self.face_data)} people and {total_samples} total face samples")
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
        """Add a new face sample to the database"""
        try:
            features = self.extract_face_features(face_image)
            
            if name not in self.face_data:
                self.face_data[name] = []
            
            self.face_data[name].append(features)
            self.save_database()
            
            sample_count = len(self.face_data[name])
            print(f"Added face sample #{sample_count} for {name} to database")
            return True
        except Exception as e:
            print(f"Error adding face to database: {e}")
            return False
    
    def find_match(self, face_image: np.ndarray, threshold: float = 0.3) -> Optional[str]:
        """Find a matching face in the database using all stored samples"""
        if not self.face_data:
            return None
        
        try:
            features = self.extract_face_features(face_image)
            
            best_match = None
            best_score = float('inf')
            
            for name, stored_samples in self.face_data.items():
                # Compare against all samples for this person and take the best match
                person_best_score = float('inf')
                
                for stored_features in stored_samples:
                    try:
                        # Calculate correlation coefficient
                        score = np.corrcoef(features, stored_features)[0, 1]
                        distance = 1 - score  # Convert correlation to distance
                        
                        if distance < person_best_score:
                            person_best_score = distance
                    except:
                        # Skip this sample if correlation fails
                        continue
                
                # Use the best score from all samples of this person
                if person_best_score < best_score:
                    best_score = person_best_score
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
        """Get the number of people in database"""
        return len(self.face_data)
    
    def get_total_samples(self) -> int:
        """Get the total number of face samples in database"""
        return sum(len(samples) for samples in self.face_data.values())
    
    def get_person_sample_count(self, name: str) -> int:
        """Get the number of samples for a specific person"""
        return len(self.face_data.get(name, []))


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
        """Capture multiple face angles and add them to the database"""
        if not self.camera:
            print("Camera not initialized")
            return False
        
        print(f"\n🎯 Multi-Angle Face Capture for {name}")
        print("=" * 50)
        print("📸 We'll capture your face from 5 different angles:")
        print("   1. Looking straight ahead (frontal)")
        print("   2. Turn head slightly to your left")
        print("   3. Turn head slightly to your right")
        print("   4. Tilt head slightly up")
        print("   5. Tilt head slightly down")
        print("\n💡 Tips: Keep good lighting, stay in frame, and follow instructions")
        input("\nPress ENTER to start the capture process...")
        
        angles = [
            ("FRONTAL", "Look straight at the camera 📷"),
            ("LEFT_TURN", "Turn your head slightly to your LEFT ⬅️"),
            ("RIGHT_TURN", "Turn your head slightly to your RIGHT ➡️"),
            ("TILT_UP", "Tilt your head slightly UP ⬆️"),
            ("TILT_DOWN", "Tilt your head slightly DOWN ⬇️")
        ]
        
        successful_captures = 0
        
        for i, (angle_name, instruction) in enumerate(angles, 1):
            print(f"\n📸 Capture {i}/5 - {angle_name}")
            print(f"🎯 {instruction}")
            
            # Give user time to position themselves
            for countdown in range(5, 0, -1):
                ret, frame = self.camera.read()
                if ret:
                    # Draw instruction on frame
                    cv2.putText(frame, f"Capture {i}/5: {angle_name}", (20, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(frame, instruction, (20, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(frame, f"Capturing in {countdown}...", (20, 450), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Show current successful captures
                    cv2.putText(frame, f"Successful: {successful_captures}/5", (400, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    cv2.imshow('Multi-Angle Face Capture', frame)
                    cv2.waitKey(1000)
            
            # Capture the face
            ret, frame = self.camera.read()
            if not ret:
                print(f"❌ Failed to capture frame for {angle_name}")
                continue
            
            # Detect faces
            face_locations = self.detect_faces(frame)
            
            if not face_locations:
                print(f"❌ No face detected for {angle_name}. Skipping this angle.")
                continue
            
            if len(face_locations) > 1:
                print(f"❌ Multiple faces detected for {angle_name}. Skipping this angle.")
                continue
            
            # Extract and save face image
            face_image = self.extract_face_region(frame, face_locations[0])
            
            if face_image.size > 0:
                # Show captured face
                capture_display = frame.copy()
                cv2.putText(capture_display, f"✅ {angle_name} CAPTURED!", (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Multi-Angle Face Capture', capture_display)
                cv2.waitKey(1500)  # Show success message
                
                success = self.database.add_face(name, face_image)
                if success:
                    successful_captures += 1
                    print(f"✅ {angle_name} captured successfully!")
                else:
                    print(f"❌ Failed to save {angle_name}")
            else:
                print(f"❌ Invalid face image for {angle_name}")
        
        cv2.destroyWindow('Multi-Angle Face Capture')
        
        print(f"\n📊 CAPTURE SUMMARY:")
        print(f"   Successfully captured: {successful_captures}/5 angles")
        print(f"   Total samples for {name}: {self.database.get_person_sample_count(name)}")
        
        if successful_captures >= 3:
            print(f"✅ Great! {name} has been added with {successful_captures} face angles.")
            print("🎯 This will significantly improve recognition accuracy!")
            return True
        else:
            print(f"⚠️  Only {successful_captures} angles captured. Consider trying again for better accuracy.")
            return successful_captures > 0
    
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
            cv2.putText(main_frame, f"Database: {self.database.get_database_size()} people, {self.database.get_total_samples()} samples", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
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
    print("=== 🚀 Multi-Angle Face Recognition App with MediaPipe ===")
    print("🎯 ENHANCED FEATURES:")
    print("   📸 Multi-angle face capture (5 different poses)")
    print("   🧠 Improved recognition accuracy with multiple face samples")
    print("   📊 Smart matching against all stored face angles")
    print("   🎨 MediaPipe face mesh visualization")
    print()
    print("📋 How it works:")
    print("   • Recognizes faces using multiple stored angles per person")
    print("   • When adding a person, captures 5 different face angles")
    print("   • Significantly improved accuracy in various lighting and poses")
    print()
    print("🎮 Controls:")
    print("  • Press 'q' to quit")
    print("  • Press 'a' to add new person (multi-angle capture)")
    print("  • Press 'm' to toggle MediaPipe face mesh preview")
    print()
    print("💡 Multi-angle capture includes:")
    print("   1. Frontal view  2. Left turn  3. Right turn  4. Tilt up  5. Tilt down")
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