"""
🤖 Face Recognition App - Quick Demo
"""

from face_recognition_app import MediaPipeFaceRecognitionApp
import cv2

def quick_demo():
    """Start the face recognition system"""
    print("🎉 FACE RECOGNITION APP")
    print("=" * 40)
    print("✅ Recognizes faces and says 'Hello [Name]!'")
    print("❓ Says 'I don't know you' for unknown faces")
    print("➕ Press 'a' to add new people")
    print("🚪 Press 'q' to quit")
    print()
    
    # Test camera
    print("🔍 Testing camera...")
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("❌ Camera not accessible!")
        return False
    camera.release()
    print("✅ Camera ready!")
    print()
    
    input("Press ENTER to start...")
    
    app = MediaPipeFaceRecognitionApp()
    try:
        app.run_recognition()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    finally:
        app.cleanup()
    
    return True

def show_info():
    """Show app information"""
    print("\n📖 APP INFO:")
    print("=" * 30)
    print("📁 Files:")
    print("   • face_recognition_app.py - Main app")
    print("   • demo.py - This demo")
    print()
    print("🎯 How to run:")
    print("   • python face_recognition_app.py")
    print("   • python demo.py")
    print()
    print("� Tips:")
    print("   • Good lighting helps")
    print("   • Face the camera directly")
    print("   • One person when adding faces")

if __name__ == "__main__":
    print("🤖 Welcome to the Face Recognition System!")
    print()
    
    while True:
        print("Choose an option:")
        print("1. 🚀 Start Face Recognition")
        print("2. 📖 Show Info")
        print("3. 🚪 Exit")
        print()
        
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == '1':
            quick_demo()
        elif choice == '2':
            show_info()
        elif choice == '3':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")
        
        print("\n" + "="*60 + "\n")