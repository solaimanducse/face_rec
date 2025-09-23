import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import threading
from face_recognition_app import MediaPipeFaceDatabase, MediaPipeFaceRecognitionApp
import numpy as np

class MediaPipeGUI:
    """GUI version using MediaPipe for face recognition"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Face Recognition App - MediaPipe")
        self.root.geometry("900x700")
        self.root.configure(bg='#2c3e50')
        
        # Force window to appear on top and focus
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after(100, lambda: self.root.attributes('-topmost', False))
        
        # Initialize MediaPipe face recognition
        self.face_app = MediaPipeFaceRecognitionApp()
        self.database = self.face_app.database
        
        self.is_running = False
        self.video_thread = None
        
        print("Setting up GUI interface...")
        self.setup_ui()
        print("GUI setup complete!")
        
    def setup_ui(self):
        """Setup the user interface"""
        # Title
        title_label = tk.Label(self.root, text="🤖 Face Recognition System", 
                              font=('Arial', 24, 'bold'), fg='white', bg='#2c3e50')
        title_label.pack(pady=15)
        
        # Subtitle
        subtitle_label = tk.Label(self.root, text="MediaPipe Edition", 
                                 font=('Arial', 12, 'italic'), fg='#bdc3c7', bg='#2c3e50')
        subtitle_label.pack(pady=(0, 10))
        
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(expand=True, fill='both', padx=20, pady=10)
        
        # Video frame
        video_container = tk.Frame(main_frame, bg='#34495e', relief='sunken', bd=3)
        video_container.pack(pady=10)
        
        self.video_frame = tk.Label(video_container, bg='black', width=640, height=480,
                                   text="📷 Camera Feed\n\nClick 'Start Camera' to begin",
                                   font=('Arial', 16), fg='white')
        self.video_frame.pack(padx=10, pady=10)
        
        # Status frame
        status_frame = tk.Frame(main_frame, bg='#2c3e50')
        status_frame.pack(pady=10, fill='x')
        
        # Status indicators
        self.status_label = tk.Label(status_frame, text="🔴 Camera: Stopped", 
                                   font=('Arial', 12, 'bold'), fg='#e74c3c', bg='#2c3e50')
        self.status_label.pack(side='left')
        
        self.db_count_label = tk.Label(status_frame, 
                                      text=f"📊 Database: {self.database.get_database_size()} faces", 
                                      font=('Arial', 12), fg='#3498db', bg='#2c3e50')
        self.db_count_label.pack(side='right')
        
        # Button frame
        button_frame = tk.Frame(main_frame, bg='#2c3e50')
        button_frame.pack(pady=15)
        
        # Start/Stop camera button
        self.start_btn = tk.Button(button_frame, text="🎥 Start Camera", 
                                  command=self.toggle_camera, bg='#27ae60', fg='white',
                                  font=('Arial', 12, 'bold'), padx=25, pady=8,
                                  relief='raised', bd=2)
        self.start_btn.pack(side='left', padx=8)
        
        # Add person button
        self.add_btn = tk.Button(button_frame, text="👤 Add New Person", 
                                command=self.add_person_dialog, bg='#3498db', fg='white',
                                font=('Arial', 12, 'bold'), padx=25, pady=8,
                                relief='raised', bd=2)
        self.add_btn.pack(side='left', padx=8)
        
        # Database info button
        self.info_btn = tk.Button(button_frame, text="📋 Database Info", 
                                 command=self.show_database_info, bg='#f39c12', fg='white',
                                 font=('Arial', 12, 'bold'), padx=25, pady=8,
                                 relief='raised', bd=2)
        self.info_btn.pack(side='left', padx=8)
        
        # Clear database button
        self.clear_btn = tk.Button(button_frame, text="🗑️ Clear Database", 
                                  command=self.clear_database, bg='#e74c3c', fg='white',
                                  font=('Arial', 12, 'bold'), padx=25, pady=8,
                                  relief='raised', bd=2)
        self.clear_btn.pack(side='left', padx=8)
        
        # Recognition result frame
        result_frame = tk.Frame(main_frame, bg='#34495e', relief='sunken', bd=2)
        result_frame.pack(pady=10, fill='x')
        
        result_title = tk.Label(result_frame, text="🎯 Recognition Result:", 
                               font=('Arial', 12, 'bold'), fg='white', bg='#34495e')
        result_title.pack(pady=(10, 5))
        
        self.result_label = tk.Label(result_frame, text="Start camera to begin recognition", 
                                    font=('Arial', 14, 'bold'), fg='#bdc3c7', bg='#34495e')
        self.result_label.pack(pady=(0, 10))
        
        # Instructions
        instructions = tk.Text(main_frame, height=4, bg='#34495e', fg='white', 
                              font=('Arial', 10), relief='sunken', bd=2)
        instructions.pack(pady=10, fill='x')
        instructions.insert('1.0', 
            "📖 Instructions:\n"
            "• Click 'Start Camera' to begin face recognition\n"
            "• Green box = Known person ('Hello [Name]!')\n"
            "• Red box = Unknown person ('I do not know you. Please introduce yourself.')\n"
            "• Use 'Add New Person' to register new faces in the database")
        instructions.config(state='disabled')
        
    def toggle_camera(self):
        """Start or stop the camera"""
        if not self.is_running:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        """Start the camera and recognition"""
        try:
            if not self.face_app.initialize_camera():
                messagebox.showerror("Error", "Could not access camera")
                return
            
            self.is_running = True
            self.start_btn.config(text="⏹️ Stop Camera", bg='#e74c3c')
            self.status_label.config(text="🟢 Camera: Running", fg='#27ae60')
            
            # Start video thread
            self.video_thread = threading.Thread(target=self.video_loop)
            self.video_thread.daemon = True
            self.video_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera: {e}")
    
    def stop_camera(self):
        """Stop the camera"""
        self.is_running = False
        if self.face_app.camera:
            self.face_app.camera.release()
        
        self.start_btn.config(text="🎥 Start Camera", bg='#27ae60')
        self.status_label.config(text="🔴 Camera: Stopped", fg='#e74c3c')
        self.result_label.config(text="Camera stopped", fg='#bdc3c7')
        self.video_frame.config(image='', text="📷 Camera Feed\n\nClick 'Start Camera' to begin")
        
    def video_loop(self):
        """Main video processing loop"""
        frame_count = 0
        process_this_frame = True
        
        while self.is_running:
            ret, frame = self.face_app.camera.read()
            if not ret:
                break
                
            # Process every other frame for performance
            if process_this_frame:
                # Detect faces
                face_locations = self.face_app.detect_faces(frame)
                
                # Process each face
                for face_location in face_locations:
                    # Extract face image
                    face_image = self.face_app.extract_face_region(frame, face_location)
                    
                    if face_image.size > 0:
                        # Try to find match in database
                        match_name = self.database.find_match(face_image)
                        
                        if match_name:
                            # Known person
                            message = f"Hello {match_name}!"
                            color = (0, 255, 0)  # Green
                            self.update_result_label(f"✅ Recognized: {match_name}", '#27ae60')
                        else:
                            # Unknown person
                            message = "Unknown Person"
                            color = (0, 0, 255)  # Red
                            self.update_result_label("❓ I do not know you. Please introduce yourself.", '#e74c3c')
                        
                        # Draw face box and message
                        frame = self.face_app.draw_face_box(frame, face_location, message, color)
            
            process_this_frame = not process_this_frame
            
            # Convert frame to display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_pil = frame_pil.resize((640, 480), Image.Resampling.LANCZOS)
            frame_tk = ImageTk.PhotoImage(frame_pil)
            
            # Update GUI
            self.video_frame.config(image=frame_tk, text="")
            self.video_frame.image = frame_tk
            
            frame_count += 1
    
    def update_result_label(self, text, color):
        """Update the result label in main thread"""
        self.root.after(0, lambda: self.result_label.config(text=text, fg=color))
    
    def add_person_dialog(self):
        """Show dialog to add new person"""
        if not self.is_running:
            messagebox.showwarning("Warning", "Please start the camera first")
            return
        
        # Create dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Add New Person")
        dialog.geometry("400x200")
        dialog.configure(bg='#2c3e50')
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() - dialog.winfo_width()) // 2
        y = (dialog.winfo_screenheight() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")
        
        tk.Label(dialog, text="👤 Add New Person to Database", 
                font=('Arial', 14, 'bold'), fg='white', bg='#2c3e50').pack(pady=15)
        
        tk.Label(dialog, text="Enter person's name:", font=('Arial', 12), 
                fg='white', bg='#2c3e50').pack(pady=5)
        
        name_entry = tk.Entry(dialog, font=('Arial', 12), width=25)
        name_entry.pack(pady=10)
        name_entry.focus()
        
        button_frame = tk.Frame(dialog, bg='#2c3e50')
        button_frame.pack(pady=15)
        
        def capture_face():
            name = name_entry.get().strip()
            if not name:
                messagebox.showerror("Error", "Name cannot be empty")
                return
            
            if name in self.database.get_all_names():
                messagebox.showerror("Error", f"{name} already exists in database")
                return
            
            dialog.destroy()
            self.capture_and_add_face(name)
        
        tk.Button(button_frame, text="📸 Capture Face", command=capture_face, 
                 bg='#27ae60', fg='white', font=('Arial', 11, 'bold'), 
                 padx=15, pady=5).pack(side='left', padx=10)
        tk.Button(button_frame, text="❌ Cancel", command=dialog.destroy, 
                 bg='#e74c3c', fg='white', font=('Arial', 11, 'bold'), 
                 padx=15, pady=5).pack(side='left', padx=10)
        
        # Bind Enter key
        dialog.bind('<Return>', lambda e: capture_face())
    
    def capture_and_add_face(self, name):
        """Capture face and add to database"""
        if not self.face_app.camera:
            messagebox.showerror("Error", "Camera not available")
            return
        
        # Show countdown
        for i in range(3, 0, -1):
            self.update_result_label(f"📸 Capturing {name} in {i}...", '#f39c12')
            self.root.update()
            self.root.after(1000)
        
        # Capture frame
        ret, frame = self.face_app.camera.read()
        if not ret:
            messagebox.showerror("Error", "Failed to capture frame")
            return
        
        # Detect faces
        face_locations = self.face_app.detect_faces(frame)
        
        if not face_locations:
            messagebox.showerror("Error", "No face detected. Please try again.")
            return
        
        if len(face_locations) > 1:
            messagebox.showerror("Error", "Multiple faces detected. Please ensure only one person is in frame.")
            return
        
        # Extract face image
        face_image = self.face_app.extract_face_region(frame, face_locations[0])
        
        if face_image.size > 0:
            success = self.database.add_face(name, face_image)
            if success:
                messagebox.showinfo("Success", f"✅ Successfully added {name} to database!")
                self.db_count_label.config(text=f"📊 Database: {self.database.get_database_size()} faces")
                self.update_result_label(f"✅ Added {name} to database", '#27ae60')
            else:
                messagebox.showerror("Error", f"❌ Failed to add {name} to database")
        else:
            messagebox.showerror("Error", "Could not extract face from image")
    
    def show_database_info(self):
        """Show database information"""
        names = self.database.get_all_names()
        count = len(names)
        
        # Create info dialog
        info_dialog = tk.Toplevel(self.root)
        info_dialog.title("Database Information")
        info_dialog.geometry("400x300")
        info_dialog.configure(bg='#2c3e50')
        info_dialog.transient(self.root)
        
        # Center the dialog
        info_dialog.update_idletasks()
        x = (info_dialog.winfo_screenwidth() - info_dialog.winfo_width()) // 2
        y = (info_dialog.winfo_screenheight() - info_dialog.winfo_height()) // 2
        info_dialog.geometry(f"+{x}+{y}")
        
        tk.Label(info_dialog, text="📊 Database Information", 
                font=('Arial', 16, 'bold'), fg='white', bg='#2c3e50').pack(pady=15)
        
        tk.Label(info_dialog, text=f"Total faces: {count}", 
                font=('Arial', 12), fg='#3498db', bg='#2c3e50').pack(pady=5)
        
        if names:
            tk.Label(info_dialog, text="Registered people:", 
                    font=('Arial', 12, 'bold'), fg='white', bg='#2c3e50').pack(pady=(15, 5))
            
            # Create scrollable list
            frame = tk.Frame(info_dialog, bg='#2c3e50')
            frame.pack(pady=5, padx=20, fill='both', expand=True)
            
            scrollbar = tk.Scrollbar(frame)
            scrollbar.pack(side='right', fill='y')
            
            listbox = tk.Listbox(frame, yscrollcommand=scrollbar.set, 
                               font=('Arial', 11), bg='#34495e', fg='white',
                               selectbackground='#3498db')
            listbox.pack(side='left', fill='both', expand=True)
            scrollbar.config(command=listbox.yview)
            
            for name in sorted(names):
                listbox.insert('end', f"👤 {name}")
        else:
            tk.Label(info_dialog, text="No faces in database", 
                    font=('Arial', 12, 'italic'), fg='#bdc3c7', bg='#2c3e50').pack(pady=20)
        
        tk.Button(info_dialog, text="✅ Close", command=info_dialog.destroy, 
                 bg='#27ae60', fg='white', font=('Arial', 11, 'bold')).pack(pady=15)
    
    def clear_database(self):
        """Clear all faces from database"""
        if not self.database.get_all_names():
            messagebox.showinfo("Info", "Database is already empty")
            return
        
        result = messagebox.askyesno("Confirm", 
                                   "⚠️ Are you sure you want to delete all faces from the database?\n\n"
                                   "This action cannot be undone!")
        if result:
            # Clear database
            self.database.face_data.clear()
            self.database.save_database()
            self.db_count_label.config(text="📊 Database: 0 faces")
            self.update_result_label("🗑️ Database cleared", '#f39c12')
            messagebox.showinfo("Success", "Database cleared successfully!")
    
    def run(self):
        """Run the GUI application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def on_closing(self):
        """Handle window closing"""
        if self.is_running:
            self.stop_camera()
        self.root.destroy()

def main():
    """Main function for GUI"""
    print("🚀 Starting Face Recognition GUI (MediaPipe Edition)...")
    app = MediaPipeGUI()
    app.run()

if __name__ == "__main__":
    main()