"""
Face management UI for adding and managing faces in the database
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
from typing import Optional, List, Dict

from services.face_service import FaceRecognitionService
from config.config import config
from utils.logger import ui_logger


class FaceManagerDialog:
    """Dialog để quản lý face database"""

    def __init__(self, parent, face_service: FaceRecognitionService):
        self.parent = parent
        self.face_service = face_service
        self.dialog = None
        self.face_list = None
        self.current_image = None
        self.current_image_path = None

        self.create_dialog()
        self.refresh_face_list()

        ui_logger.info("👤 Face manager dialog created")

    def create_dialog(self):
        """Tạo dialog UI"""
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("👤 Quản lý Face Database")
        self.dialog.geometry("900x700")
        self.dialog.resizable(True, True)
        self.dialog.transient(self.parent)
        self.dialog.grab_set()

        # Center dialog
        self.center_dialog()

        # Configure grid weights
        self.dialog.grid_rowconfigure(1, weight=1)
        self.dialog.grid_columnconfigure(0, weight=1)

        # Create sections
        self.create_header()
        self.create_main_content()
        self.create_buttons()
        self.create_status_bar()

        # Bind events
        self.dialog.protocol("WM_DELETE_WINDOW", self.on_close)

    def center_dialog(self):
        """Center dialog on screen"""
        self.dialog.update_idletasks()
        width = self.dialog.winfo_width()
        height = self.dialog.winfo_height()
        x = (self.dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (height // 2)
        self.dialog.geometry(f"{width}x{height}+{x}+{y}")

    def create_header(self):
        """Tạo header section"""
        header_frame = ttk.Frame(self.dialog)
        header_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=20)

        # Title
        title_label = ttk.Label(
            header_frame,
            text="👤 Quản lý Face Database",
            font=('Arial', 16, 'bold')
        )
        title_label.pack(pady=(0, 10))

        # Statistics
        self.stats_label = ttk.Label(
            header_frame,
            text="Loading statistics...",
            font=('Arial', 10),
            foreground='gray'
        )
        self.stats_label.pack()

    def create_main_content(self):
        """Tạo main content area"""
        main_frame = ttk.Frame(self.dialog)
        main_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)

        # Left panel - Face list
        self.create_face_list_panel(main_frame)

        # Right panel - Face details
        self.create_face_details_panel(main_frame)

    def create_face_list_panel(self, parent):
        """Tạo panel danh sách faces"""
        list_frame = ttk.LabelFrame(parent, text="📋 Danh sách Faces", padding=10)
        list_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        list_frame.grid_rowconfigure(1, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)

        # Search frame
        search_frame = ttk.Frame(list_frame)
        search_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        search_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(search_frame, text="🔍 Search:").grid(row=0, column=0, padx=(0, 5))

        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(search_frame, textvariable=self.search_var)
        self.search_entry.grid(row=0, column=1, sticky="ew", padx=(0, 5))
        self.search_entry.bind('<KeyRelease>', self.on_search)

        ttk.Button(search_frame, text="🔄", command=self.refresh_face_list, width=3).grid(row=0, column=2)

        # Face list
        list_container = ttk.Frame(list_frame)
        list_container.grid(row=1, column=0, sticky="nsew")
        list_container.grid_rowconfigure(0, weight=1)
        list_container.grid_columnconfigure(0, weight=1)

        # Treeview
        columns = ('Name', 'Age', 'Gender', 'Created')
        self.face_tree = ttk.Treeview(list_container, columns=columns, show='headings', height=15)

        # Configure columns
        self.face_tree.heading('Name', text='Tên')
        self.face_tree.heading('Age', text='Tuổi')
        self.face_tree.heading('Gender', text='Giới tính')
        self.face_tree.heading('Created', text='Ngày tạo')

        self.face_tree.column('Name', width=120, anchor='w')
        self.face_tree.column('Age', width=60, anchor='center')
        self.face_tree.column('Gender', width=80, anchor='center')
        self.face_tree.column('Created', width=100, anchor='center')

        # Scrollbar
        scrollbar = ttk.Scrollbar(list_container, orient=tk.VERTICAL, command=self.face_tree.yview)
        self.face_tree.configure(yscrollcommand=scrollbar.set)

        self.face_tree.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        # Bind events
        self.face_tree.bind('<<TreeviewSelect>>', self.on_face_select)
        self.face_tree.bind('<Double-1>', self.edit_face)
        self.face_tree.bind('<Button-3>', self.show_face_context_menu)

        # Context menu
        self.create_face_context_menu()

    def create_face_context_menu(self):
        """Tạo context menu cho face list"""
        self.face_context_menu = tk.Menu(self.dialog, tearoff=0)
        self.face_context_menu.add_command(label="✏️ Edit", command=self.edit_face)
        self.face_context_menu.add_command(label="🗑️ Delete", command=self.delete_face)
        self.face_context_menu.add_separator()
        self.face_context_menu.add_command(label="📋 Copy Name", command=self.copy_face_name)
        self.face_context_menu.add_command(label="ℹ️ Info", command=self.show_face_info)

    def show_face_context_menu(self, event):
        """Hiển thị context menu cho face"""
        item = self.face_tree.identify_row(event.y)
        if item:
            self.face_tree.selection_set(item)
            self.face_context_menu.post(event.x_root, event.y_root)

    def create_face_details_panel(self, parent):
        """Tạo panel chi tiết face"""
        details_frame = ttk.LabelFrame(parent, text="👤 Chi tiết Face", padding=10)
        details_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        details_frame.grid_rowconfigure(3, weight=1)
        details_frame.grid_columnconfigure(0, weight=1)

        # Face image preview
        self.image_label = ttk.Label(
            details_frame,
            text="📷 Chọn face để xem chi tiết",
            font=('Arial', 12),
            background='lightgray',
            anchor='center'
        )
        self.image_label.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        # Face info
        info_frame = ttk.Frame(details_frame)
        info_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        info_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(info_frame, text="Name:").grid(row=0, column=0, sticky="w", padx=(0, 10))
        self.name_var = tk.StringVar()
        self.name_entry = ttk.Entry(info_frame, textvariable=self.name_var, state='readonly')
        self.name_entry.grid(row=0, column=1, sticky="ew", pady=2)

        ttk.Label(info_frame, text="Age:").grid(row=1, column=0, sticky="w", padx=(0, 10))
        self.age_var = tk.StringVar()
        self.age_entry = ttk.Entry(info_frame, textvariable=self.age_var, state='readonly')
        self.age_entry.grid(row=1, column=1, sticky="ew", pady=2)

        ttk.Label(info_frame, text="Gender:").grid(row=2, column=0, sticky="w", padx=(0, 10))
        self.gender_var = tk.StringVar()
        self.gender_entry = ttk.Entry(info_frame, textvariable=self.gender_var, state='readonly')
        self.gender_entry.grid(row=2, column=1, sticky="ew", pady=2)

        ttk.Label(info_frame, text="Created:").grid(row=3, column=0, sticky="w", padx=(0, 10))
        self.created_var = tk.StringVar()
        self.created_entry = ttk.Entry(info_frame, textvariable=self.created_var, state='readonly')
        self.created_entry.grid(row=3, column=1, sticky="ew", pady=2)

        # Action buttons
        action_frame = ttk.Frame(details_frame)
        action_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))

        ttk.Button(action_frame, text="✏️ Edit", command=self.edit_face).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="🗑️ Delete", command=self.delete_face).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="📋 Copy", command=self.copy_face_name).pack(side=tk.LEFT, padx=5)

        # Additional info
        self.additional_info = tk.Text(
            details_frame,
            height=8,
            width=40,
            wrap=tk.WORD,
            state='disabled'
        )
        self.additional_info.grid(row=3, column=0, sticky="nsew", pady=(10, 0))

        # Scrollbar for text
        info_scrollbar = ttk.Scrollbar(details_frame, orient=tk.VERTICAL, command=self.additional_info.yview)
        self.additional_info.configure(yscrollcommand=info_scrollbar.set)
        info_scrollbar.grid(row=3, column=1, sticky="ns", pady=(10, 0))

    def create_buttons(self):
        """Tạo button panel"""
        button_frame = ttk.Frame(self.dialog)
        button_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=20)

        # Left buttons
        left_buttons = ttk.Frame(button_frame)
        left_buttons.pack(side=tk.LEFT)

        ttk.Button(left_buttons, text="➕ Add Face", command=self.add_face, width=12).pack(side=tk.LEFT, padx=5)
        ttk.Button(left_buttons, text="💾 Backup", command=self.backup_database, width=12).pack(side=tk.LEFT, padx=5)
        ttk.Button(left_buttons, text="📥 Restore", command=self.restore_database, width=12).pack(side=tk.LEFT, padx=5)

        # Right buttons
        right_buttons = ttk.Frame(button_frame)
        right_buttons.pack(side=tk.RIGHT)

        ttk.Button(right_buttons, text="📊 Statistics", command=self.show_statistics, width=12).pack(side=tk.RIGHT,
                                                                                                    padx=5)
        ttk.Button(right_buttons, text="❌ Close", command=self.on_close, width=12).pack(side=tk.RIGHT, padx=5)

    def create_status_bar(self):
        """Tạo status bar"""
        self.status_label = ttk.Label(
            self.dialog,
            text="Ready",
            font=('Arial', 9),
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_label.grid(row=3, column=0, sticky="ew", padx=20, pady=(0, 10))

    def refresh_face_list(self):
        """Refresh face list"""
        self.update_status("Refreshing face list...")

        try:
            # Clear existing items
            for item in self.face_tree.get_children():
                self.face_tree.delete(item)

            # Get faces from service
            faces = self.face_service.get_face_list()

            for face in faces:
                # Format data for display
                age = face.get('age', 0)
                age_str = str(age) if age > 0 else 'N/A'

                gender = face.get('gender', 0)
                gender_str = 'Male' if gender == 0 else 'Female'

                created = face.get('created_at', '')
                created_str = created[:10] if created else 'N/A'  # Date only

                # Insert into tree
                self.face_tree.insert('', tk.END, values=(
                    face['name'],
                    age_str,
                    gender_str,
                    created_str
                ))

            # Update statistics
            self.update_statistics()

            count = len(faces)
            self.update_status(f"✅ Loaded {count} faces")
            ui_logger.info(f"👤 Refreshed face list: {count} faces")

        except Exception as e:
            ui_logger.error(f"❌ Error refreshing face list: {e}")
            self.update_status(f"❌ Error: {str(e)}")

    def update_statistics(self):
        """Update statistics display"""
        try:
            stats = self.face_service.get_statistics()

            stats_text = (
                f"📊 Total: {stats['total_faces']} faces | "
                f"👨 Male: {stats['male_count']} | "
                f"👩 Female: {stats['female_count']} | "
                f"📈 Avg Age: {stats['average_age']}"
            )

            self.stats_label.config(text=stats_text)

        except Exception as e:
            ui_logger.error(f"❌ Error updating statistics: {e}")
            self.stats_label.config(text="❌ Error loading statistics")

    def on_search(self, event=None):
        """Handle search"""
        query = self.search_var.get().lower()

        # Show/hide items based on search
        for item in self.face_tree.get_children():
            values = self.face_tree.item(item, 'values')
            name = values[0].lower()

            if query in name:
                self.face_tree.reattach(item, '', 'end')
            else:
                self.face_tree.detach(item)

    def on_face_select(self, event):
        """Handle face selection"""
        selection = self.face_tree.selection()
        if not selection:
            self.clear_face_details()
            return

        item = self.face_tree.item(selection[0])
        face_name = item['values'][0]

        self.show_face_details(face_name)

    def show_face_details(self, face_name: str):
        """Show face details"""
        try:
            face_info = self.face_service.get_face_info(face_name)
            if not face_info:
                self.clear_face_details()
                return

            # Update fields
            self.name_var.set(face_name)
            self.age_var.set(face_info.get('age', 'N/A'))
            self.gender_var.set('Male' if face_info.get('gender', 0) == 0 else 'Female')
            self.created_var.set(face_info.get('created_at', 'N/A')[:19])

            # Update additional info
            self.additional_info.config(state='normal')
            self.additional_info.delete(1.0, tk.END)

            info_text = f"""Face Information:

Embedding Size: {face_info.get('embedding_size', 'N/A')}
Face Count: {face_info.get('face_count', 'N/A')}
Last Updated: {face_info.get('updated_at', 'N/A')[:19]}

Landmarks: {'Available' if face_info.get('landmarks') else 'Not Available'}
Bounding Box: {'Available' if face_info.get('bbox') else 'Not Available'}

Recognition Threshold: {config.RECOGNITION_THRESHOLD}
"""

            self.additional_info.insert(1.0, info_text)
            self.additional_info.config(state='disabled')

        except Exception as e:
            ui_logger.error(f"❌ Error showing face details: {e}")
            self.clear_face_details()

    def clear_face_details(self):
        """Clear face details"""
        self.name_var.set('')
        self.age_var.set('')
        self.gender_var.set('')
        self.created_var.set('')

        self.additional_info.config(state='normal')
        self.additional_info.delete(1.0, tk.END)
        self.additional_info.config(state='disabled')

    def add_face(self):
        """Add new face"""
        AddFaceDialog(self.dialog, self.face_service, self.refresh_face_list)

    def edit_face(self):
        """Edit selected face"""
        selection = self.face_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a face to edit!")
            return

        item = self.face_tree.item(selection[0])
        face_name = item['values'][0]

        # TODO: Implement edit face dialog
        messagebox.showinfo("Edit Face", f"Edit functionality for '{face_name}' will be implemented")

    def delete_face(self):
        """Delete selected face"""
        selection = self.face_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a face to delete!")
            return

        item = self.face_tree.item(selection[0])
        face_name = item['values'][0]

        # Confirm deletion
        result = messagebox.askyesno(
            "Confirm Delete",
            f"Are you sure you want to delete face '{face_name}'?\n\nThis action cannot be undone."
        )

        if result:
            if self.face_service.remove_face_from_database(face_name):
                self.refresh_face_list()
                self.update_status(f"✅ Deleted face: {face_name}")
                ui_logger.info(f"🗑️ Deleted face: {face_name}")
            else:
                messagebox.showerror("Error", f"Failed to delete face '{face_name}'")

    def copy_face_name(self):
        """Copy face name to clipboard"""
        selection = self.face_tree.selection()
        if not selection:
            return

        item = self.face_tree.item(selection[0])
        face_name = item['values'][0]

        self.dialog.clipboard_clear()
        self.dialog.clipboard_append(face_name)
        self.update_status(f"📋 Copied name: {face_name}")

    def show_face_info(self):
        """Show detailed face info"""
        selection = self.face_tree.selection()
        if not selection:
            return

        item = self.face_tree.item(selection[0])
        face_name = item['values'][0]

        face_info = self.face_service.get_face_info(face_name)
        if face_info:
            info_text = f"""Face Information: {face_name}

Created: {face_info.get('created_at', 'N/A')}
Updated: {face_info.get('updated_at', 'N/A')}
Age: {face_info.get('age', 'N/A')}
Gender: {'Male' if face_info.get('gender', 0) == 0 else 'Female'}
Embedding Size: {face_info.get('embedding_size', 'N/A')}
Face Count: {face_info.get('face_count', 'N/A')}
"""
            messagebox.showinfo(f"Face Info - {face_name}", info_text)

    def backup_database(self):
        """Backup face database"""
        file_path = filedialog.asksaveasfilename(
            title="Backup Face Database",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if file_path:
            if self.face_service.backup_face_database(file_path):
                messagebox.showinfo("Success", f"✅ Face database backed up to:\n{file_path}")
                self.update_status(f"✅ Backup saved: {file_path}")
            else:
                messagebox.showerror("Error", "Failed to backup face database")

    def restore_database(self):
        """Restore face database"""
        result = messagebox.askyesno(
            "Confirm Restore",
            "Restoring will replace the current face database.\n\nDo you want to continue?"
        )

        if not result:
            return

        file_path = filedialog.askopenfilename(
            title="Restore Face Database",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if file_path:
            if self.face_service.restore_face_database(file_path):
                messagebox.showinfo("Success", f"✅ Face database restored from:\n{file_path}")
                self.refresh_face_list()
                self.update_status(f"✅ Restored from: {file_path}")
            else:
                messagebox.showerror("Error", "Failed to restore face database")

    def show_statistics(self):
        """Show detailed statistics"""
        try:
            stats = self.face_service.get_statistics()

            stats_text = f"""Face Database Statistics:

Total Faces: {stats['total_faces']}
Male: {stats['male_count']}
Female: {stats['female_count']}
Unknown Gender: {stats['unknown_gender']}

Average Age: {stats['average_age']} years
Recognition Threshold: {stats['recognition_threshold']}
Database Size: {stats['database_size']} bytes

System Status:
InsightFace Initialized: {'Yes' if stats['is_initialized'] else 'No'}
"""

            messagebox.showinfo("Face Database Statistics", stats_text)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load statistics:\n{str(e)}")

    def update_status(self, message: str):
        """Update status message"""
        self.status_label.config(text=message)

    def on_close(self):
        """Close dialog"""
        self.dialog.destroy()
        ui_logger.info("👤 Face manager dialog closed")


class AddFaceDialog:
    """Dialog để thêm face mới"""

    def __init__(self, parent, face_service: FaceRecognitionService, callback=None):
        self.parent = parent
        self.face_service = face_service
        self.callback = callback
        self.dialog = None
        self.image_path = None
        self.preview_image = None

        self.create_dialog()

    def create_dialog(self):
        """Tạo add face dialog"""
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("➕ Add New Face")
        self.dialog.geometry("500x600")
        self.dialog.resizable(False, False)
        self.dialog.transient(self.parent)
        self.dialog.grab_set()

        # Center dialog
        self.center_dialog()

        # Create UI
        self.create_add_face_ui()

        # Bind events
        self.dialog.protocol("WM_DELETE_WINDOW", self.on_cancel)

    def center_dialog(self):
        """Center dialog on screen"""
        self.dialog.update_idletasks()
        width = self.dialog.winfo_width()
        height = self.dialog.winfo_height()
        x = (self.dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (height // 2)
        self.dialog.geometry(f"{width}x{height}+{x}+{y}")

    def create_add_face_ui(self):
        """Tạo UI cho add face"""
        main_frame = ttk.Frame(self.dialog, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(main_frame, text="➕ Add New Face", font=('Arial', 14, 'bold'))
        title_label.pack(pady=(0, 20))

        # Image selection
        image_frame = ttk.LabelFrame(main_frame, text="📷 Select Image", padding=10)
        image_frame.pack(fill=tk.X, pady=(0, 20))

        ttk.Button(image_frame, text="📂 Browse Image", command=self.browse_image).pack(pady=5)

        # Image preview
        self.preview_label = ttk.Label(
            image_frame,
            text="No image selected",
            background='lightgray',
            anchor='center',
            font=('Arial', 10)
        )
        self.preview_label.pack(fill=tk.X, pady=(10, 0))

        # Face name
        name_frame = ttk.LabelFrame(main_frame, text="👤 Face Name", padding=10)
        name_frame.pack(fill=tk.X, pady=(0, 20))

        self.name_var = tk.StringVar()
        self.name_entry = ttk.Entry(name_frame, textvariable=self.name_var, font=('Arial', 12))
        self.name_entry.pack(fill=tk.X, pady=5)
        self.name_entry.focus()

        # Validation info
        self.validation_label = ttk.Label(
            main_frame,
            text="Please select an image and enter a name",
            font=('Arial', 9),
            foreground='gray'
        )
        self.validation_label.pack(pady=(0, 20))

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)

        ttk.Button(button_frame, text="❌ Cancel", command=self.on_cancel).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="✅ Add Face", command=self.on_add_face).pack(side=tk.RIGHT, padx=5)

    def browse_image(self):
        """Browse for image file"""
        file_path = filedialog.askopenfilename(
            title="Select Face Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            self.image_path = file_path
            self.load_preview_image()
            self.validate_image()

    def load_preview_image(self):
        """Load and display preview image"""
        if not self.image_path:
            return

        try:
            # Load image
            image = cv2.imread(self.image_path)
            if image is None:
                self.validation_label.config(text="❌ Cannot load image file", foreground='red')
                return

            # Resize for preview
            height, width = image.shape[:2]
            max_size = 200

            if width > height:
                new_width = max_size
                new_height = int(height * max_size / width)
            else:
                new_height = max_size
                new_width = int(width * max_size / height)

            resized = cv2.resize(image, (new_width, new_height))

            # Convert to RGB for PIL
            rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)

            # Convert to PhotoImage
            self.preview_image = ImageTk.PhotoImage(pil_image)

            # Update preview label
            self.preview_label.config(image=self.preview_image, text='')

        except Exception as e:
            ui_logger.error(f"❌ Error loading preview image: {e}")
            self.validation_label.config(text=f"❌ Error loading image: {str(e)}", foreground='red')

    def validate_image(self):
        """Validate selected image"""
        if not self.image_path:
            return

        try:
            # Load image
            image = cv2.imread(self.image_path)
            if image is None:
                self.validation_label.config(text="❌ Invalid image file", foreground='red')
                return

            # Validate with face service
            validation = self.face_service.validate_face_image(image)

            if validation['valid']:
                face_info = validation.get('face_size', (0, 0))
                self.validation_label.config(
                    text=f"✅ Valid face detected ({face_info[0]}x{face_info[1]})",
                    foreground='green'
                )
            else:
                self.validation_label.config(
                    text=f"❌ {validation['message']}",
                    foreground='red'
                )

        except Exception as e:
            ui_logger.error(f"❌ Error validating image: {e}")
            self.validation_label.config(text=f"❌ Validation error: {str(e)}", foreground='red')

    def on_add_face(self):
        """Add face to database"""
        name = self.name_var.get().strip()

        if not name:
            messagebox.showwarning("Warning", "Please enter a name for the face!")
            return

        if not self.image_path:
            messagebox.showwarning("Warning", "Please select an image!")
            return

        try:
            # Load image
            image = cv2.imread(self.image_path)
            if image is None:
                messagebox.showerror("Error", "Cannot load image file!")
                return

            # Add to database
            if self.face_service.add_face_to_database(name, image):
                messagebox.showinfo("Success", f"✅ Face '{name}' added successfully!")

                # Call callback to refresh parent list
                if self.callback:
                    self.callback()

                self.dialog.destroy()
                ui_logger.info(f"✅ Added face: {name}")

            else:
                messagebox.showerror("Error", f"Failed to add face '{name}' to database!")

        except Exception as e:
            ui_logger.error(f"❌ Error adding face: {e}")
            messagebox.showerror("Error", f"Error adding face:\n{str(e)}")

    def on_cancel(self):
        """Cancel adding face"""
        self.dialog.destroy()