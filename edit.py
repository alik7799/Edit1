import os
import json
import numpy as np
from PIL import Image as PILImage
from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup
from kivy.graphics import Color, Line, RoundedRectangle
from kivy.core.window import Window
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.widget import Widget
import platform
from kivy.uix.textinput import TextInput
from kivy.uix.filechooser import FileChooserIconView
import string
import platform
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout

class RoundedButton(Button):
    def __init__(self, bg_color=(0.2, 0.6, 0.86, 1), radius=16, **kwargs):
        super().__init__(**kwargs)
        self.bg_color = bg_color
        self.radius = radius
        self.background_normal = ''
        self.background_color = (0, 0, 0, 0)
        self.color = (1, 1, 1, 1)
        self.font_size = 16
        with self.canvas.before:
            Color(*self.bg_color)
            self.rect = RoundedRectangle(radius=[self.radius])
        self.bind(pos=self.update_rect, size=self.update_rect)

    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size

def rgb_to_lab(rgb_color):
    import cv2
    rgb_pixel = np.uint8([[rgb_color]])
    lab_pixel = cv2.cvtColor(rgb_pixel, cv2.COLOR_RGB2LAB)
    return lab_pixel[0][0]

def delta_e(color1_rgb, color2_rgb):
    lab1 = rgb_to_lab(color1_rgb)
    lab2 = rgb_to_lab(color2_rgb)
    return np.linalg.norm(lab1 - lab2)

class SelectableImage(Image):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.allow_stretch = True
        self.keep_ratio = True
        self.start_pos = None
        self.end_pos = None
        self.selection_confirmed = False
        self.selection_canvas = None

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            self.start_pos = touch.pos
            with self.canvas:
                Color(1, 0.5, 0, 0.5)
                if self.selection_canvas:
                    self.canvas.remove(self.selection_canvas)
                self.selection_canvas = Line(rectangle=(touch.x, touch.y, 0, 0), width=2)
            return True
        return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        if self.selection_canvas and self.start_pos:
            x1, y1 = self.start_pos
            x2, y2 = touch.pos
            x = min(x1, x2)
            y = min(y1, y2)
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            self.selection_canvas.rectangle = (x, y, w, h)
            self.end_pos = touch.pos
            return True
        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        if self.selection_canvas:
            self.end_pos = touch.pos
            self.selection_confirmed = True
            return True
        return super().on_touch_up(touch)

class ImageApp(App):
    
    
    def build(self):
        Window.clearcolor = (0.15, 0.27, 0.42, 1)
        self.image_paths = []
        self.current_index = 0
        self.ref_rgb = [140, 57, 54]
        self.thresholds = {"fresh": 30, "warning": 35, "spoiled": 100}
        self.white_reference = None
        self.root = FloatLayout()
        self.analysis_table = {} # ذخیره داده‌ها: filename → [status, delta_E, R, G, B

        self.image_widget = SelectableImage(
            size_hint=(0.95, 0.6),
            pos_hint={"center_x": 0.5, "top": 0.95},
        )
        self.root.add_widget(self.image_widget)

        self.status_label = Label(
            text="Load images → Navigate → Select region → Confirm → Analyze",
            size_hint=(1, 0.08),
            pos_hint={"center_x": 0.5, "y": 0.03},
            color=(1, 1, 1, 1),
            font_size=16,
        )
        self.root.add_widget(self.status_label)

        self.load_btn = RoundedButton(
            text="📷 Load Images",
            size_hint=(0.3, 0.07),
            pos_hint={"x": 0.05, "y": 0.28},
        )
        self.prev_btn = RoundedButton(
            text="◀️ Prev",
            size_hint=(0.15, 0.07),
            pos_hint={"x": 0.05, "y": 0.18},
        )
        self.next_btn = RoundedButton(
            text="Next ▶️",
            size_hint=(0.15, 0.07),
            pos_hint={"right": 0.95, "y": 0.18},
        )
        self.confirm_btn = RoundedButton(
            text="✅ Confirm",
            size_hint=(0.3, 0.07),
            pos_hint={"center_x": 0.5, "y": 0.28},
            bg_color=(0.18, 0.63, 0.17, 1),
        )
        self.analyze_btn = RoundedButton(
            text="📊 Analyze",
            size_hint=(0.3, 0.07),
            pos_hint={"right": 0.95, "y": 0.28},
            bg_color=(0.96, 0.47, 0.2, 1),
        )
        self.analyze_btn.disabled = True
        

        self.set_white_btn = RoundedButton(
            text="🎯 Set White Ref.",
            size_hint=(0.3, 0.07),
            pos_hint={"center_x": 0.5, "y": 0.18},
            bg_color=(0.4, 0.4, 1, 1),
            )
        self.set_white_btn.bind(on_press=self.set_white_reference)
        self.root.add_widget(self.set_white_btn)
        
        self.clear_white_btn = RoundedButton(
            text="❌ Clear White Ref.",
            size_hint=(0.3, 0.07),
            pos_hint={"center_x": 0.5, "y": 0.10},
            bg_color=(0.7, 0.2, 0.2, 1),
            )   
        self.clear_white_btn.bind(on_press=self.clear_white_reference)
        self.root.add_widget(self.clear_white_btn)
        
        self.get_rgb_btn = RoundedButton(
            text="🎯 Get RGB",
            size_hint=(0.25, 0.07),
            pos_hint={"right": 0.95, "y": 0.08},
            bg_color=(0.3, 0.7, 0.9, 1)
            )
        self.get_rgb_btn.bind(on_press=self.get_selected_rgb)
        self.root.add_widget(self.get_rgb_btn)

        self.rgb_label = Label(
            text="RGB: - , - , -",
            size_hint=(0.4, 0.05),
            pos_hint={"center_x": 0.77, "y": 0.02},
            color=(1, 1, 1, 1),
            font_size=15,
        )
        self.root.add_widget(self.rgb_label)


        self.image_index_label = Label(
            text="Image 0 / 0",
            size_hint=(0.2, 0.05),
            pos_hint={"center_x": 0.9, "y": 0.87},
            color=(1, 1, 1, 1),
            font_size=17,
        )
        self.filename_label = Label(
            text="",
            size_hint=(0.6, 0.05),
            pos_hint={"center_x": 0.1, "y": 0.87},
            color=(1, 1, 1, 1),
            font_size=17,
        )
        self.root.add_widget(self.filename_label)

        self.load_btn.bind(on_press=self.show_filechooser)
        self.confirm_btn.bind(on_press=self.confirm_selection)
        self.analyze_btn.bind(on_press=self.analyze_roi)
        self.prev_btn.bind(on_press=self.prev_image)
        self.next_btn.bind(on_press=self.next_image)

        self.root.add_widget(self.load_btn)
        self.root.add_widget(self.confirm_btn)
        self.root.add_widget(self.analyze_btn)
        self.root.add_widget(self.prev_btn)
        self.root.add_widget(self.next_btn)
        self.root.add_widget(self.image_index_label)
        
        self.settings_btn = RoundedButton(
            text="⚙️ Settings",
            size_hint=(0.25, 0.07),
            pos_hint={"x": 0.05, "y": 0.08},
            bg_color=(0.5, 0.5, 0.5, 1)
        )
        self.settings_btn.bind(on_press=self.open_settings_popup)
        self.root.add_widget(self.settings_btn)


        return self.root
    
    def update_table_entry(self, filename, delta_e_value=None, rgb=None, status=None):
        entry = self.analysis_table.get(filename, [None, None, None, None, None])

        if delta_e_value is not None:
            entry[1] = round(delta_e_value, 2)
            if rgb is not None:
                entry[2:5] = rgb
                if status is not None:
                    entry[0] = status

        self.analysis_table[filename] = entry

    def show_filechooser(self, instance):
    # ساخت ویجت انتخاب فایل
        filechooser = FileChooserIconView(multiselect=True, size_hint=(1, 0.85))
    # گرفتن لیست درایوها (ویندوز)
        drives = [f"{d}:/" for d in string.ascii_uppercase if os.path.exists(f"{d}:/")]

    # نوار درایوها
        drive_bar = BoxLayout(size_hint=(1, 0.1), spacing=5, padding=5)
        for drive in drives:
            btn = RoundedButton(
                text=f"🖴 {drive}",
                size_hint=(None, None),
                size=(80, 40),
                font_size=14,
                bg_color=(0.2, 0.5, 0.8, 1)
                )
        # استفاده از path به عنوان مقدار پیش‌فرض برای جلوگیری از خطا
            btn.bind(on_press=lambda instance, path=drive: setattr(filechooser, 'path', path))
            drive_bar.add_widget(btn)

    # دکمه‌های OK و Cancel
        btns = BoxLayout(size_hint=(1, 0.1), spacing=10)
        btn_ok = RoundedButton(text="✅ OK", size_hint=(0.5, 1))
        btn_cancel = RoundedButton(text="❌ Cancel", size_hint=(0.5, 1))
        btns.add_widget(btn_ok)
        btns.add_widget(btn_cancel)

    # ساخت لایه کلی
        layout = BoxLayout(orientation='vertical', spacing=10, padding=10)
        layout.add_widget(drive_bar)
        layout.add_widget(filechooser)
        layout.add_widget(btns)

        popup = Popup(title="Select Images", content=layout, size_hint=(0.95, 0.95))

        def load_and_close(instance):
            if filechooser.selection:
                self.image_paths = filechooser.selection
                self.current_index = 0
                self.ref_rgb = [140, 58, 54]
                self.thresholds = {
                    "fresh": 30,
                    "warning": 35,
                    "spoiled": 100
                    }
            self.update_image()
            popup.dismiss()

        btn_ok.bind(on_press=load_and_close)
        btn_cancel.bind(on_press=popup.dismiss)
        popup.open()
        
 
    def clear_white_reference(self, instance):
        self.white_reference = None
        self.status_label.text = "White reference cleared."

    def update_image(self):
        if self.image_paths:
            path = self.image_paths[self.current_index]
            self.image_widget.source = path
            self.status_label.text = f"Loaded image {self.current_index + 1} of {len(self.image_paths)}"
            self.image_index_label.text = f"Image {self.current_index + 1} / {len(self.image_paths)}"
            filename = os.path.basename(path)
            self.filename_label.text = f"Filename : {filename}"
            self.analyze_btn.disabled = True
            self.image_widget.selection_confirmed = False
            
    def open_settings_popup(self, instance):
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        input_fields = {}

        def create_input(label_text, default):
            box = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
            label = Label(text=label_text, size_hint_x=0.5, color=(1,1,1,1))
            input_box = TextInput(text=str(default), multiline=False, input_filter='float', size_hint_x=0.5)
            box.add_widget(label)
            box.add_widget(input_box)
            layout.add_widget(box)
            return input_box

    # Create input fields
        input_fields['R'] = create_input("Reference R:", self.ref_rgb[0])
        input_fields['G'] = create_input("Reference G:", self.ref_rgb[1])
        input_fields['B'] = create_input("Reference B:", self.ref_rgb[2])
        input_fields['fresh'] = create_input("Max ∆E for Fresh:", self.thresholds["fresh"])
        input_fields['warning'] = create_input("Max ∆E for Warning:", self.thresholds["warning"])
        input_fields['spoiled'] = create_input("Max ∆E for Spoiled:", self.thresholds["spoiled"])

    # Save and Cancel buttons
        btns = BoxLayout(size_hint_y=None, height=40, spacing=10)
        save_btn = RoundedButton(text="Save", bg_color=(0.1, 0.6, 0.1, 1))
        cancel_btn = RoundedButton(text="Cancel", bg_color=(0.6, 0.1, 0.1, 1))
        btns.add_widget(save_btn)
        btns.add_widget(cancel_btn)
        layout.add_widget(btns)

        popup = Popup(title="Adjust Reference RGB and ΔE Thresholds", content=layout, size_hint=(0.85, 0.8))

        def save_settings(instance):
            try:
                self.ref_rgb = [float(input_fields['R'].text), float(input_fields['G'].text),float(input_fields['B'].text)]
                self.thresholds["fresh"] = float(input_fields['fresh'].text)
                self.thresholds["warning"] = float(input_fields['warning'].text)
                self.thresholds["spoiled"] = float(input_fields['spoiled'].text)
                self.status_label.text = "Settings updated."
                popup.dismiss()
            except:
                self.status_label.text = "Invalid input in settings."

        save_btn.bind(on_press=save_settings)
        cancel_btn.bind(on_press=popup.dismiss)
        popup.open()
        
    def show_data_table(self):
        layout = BoxLayout(orientation='vertical', spacing=10, padding=10)

        scroll = ScrollView(size_hint=(1, 1))
        table = GridLayout(cols=6, spacing=5, size_hint_y=None)
        table.bind(minimum_height=table.setter('height'))

        headers = ["Filename", "ΔE", "Status", "R", "G", "B"]
        for h in headers:
            table.add_widget(Label(text=h, bold=True, color=(1,1,1,1)))

        for row in self.analysis_table:
            for item in row:
                table.add_widget(Label(text=str(item), color=(1,1,1,1)))

        scroll.add_widget(table)
        layout.add_widget(scroll)

        btn_close = RoundedButton(text="Close", size_hint_y=None, height=40)
        layout.add_widget(btn_close)

        popup = Popup(title="Data Table", content=layout, size_hint=(0.9, 0.9))
        btn_close.bind(on_press=popup.dismiss)
        popup.open()
        
    def show_data_table_window(self):

        layout = BoxLayout(orientation='vertical', spacing=10, padding=10)
        headers = ["Filename", "Status", "ΔE", "R", "G", "B"]   
        table = GridLayout(cols=len(headers), spacing=5, size_hint_y=None)
        table.bind(minimum_height=table.setter('height'))

        for head in headers:
            table.add_widget(Label(text=head, bold=True, color=(1, 1, 1, 1), font_size=16))

        for filename, row in self.analysis_table.items():
            table.add_widget(Label(text=filename, color=(1, 1, 1, 1)))
        for val in row:
            table.add_widget(Label(text=str(val) if val is not None else "-", color=(1, 1, 1, 1)))

        scroll = ScrollView(size_hint=(1, 1))
        scroll.add_widget(table)

        layout.add_widget(scroll)

    # دکمه‌های پایین جدول
        buttons = BoxLayout(size_hint_y=None, height=50, spacing=10)
        close_btn = RoundedButton(text="❌ Close", bg_color=(0.6, 0.1, 0.1, 1))
        save_btn = RoundedButton(text="💾 Save to CSV", bg_color=(0.2, 0.6, 0.2, 1))
        buttons.add_widget(save_btn)
        buttons.add_widget(close_btn)

        layout.add_widget(buttons)
        
        popup = Popup(title="Analysis Table", content=layout, size_hint=(0.9, 0.9))
        close_btn.bind(on_press=popup.dismiss)
        save_btn.bind(on_press=lambda x: self.export_table_to_csv(popup))
        popup.open()



    def prev_image(self, instance):
        if self.image_paths and self.current_index > 0:
            self.current_index -= 1
            self.update_image()

    def next_image(self, instance):
        if self.image_paths and self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self.update_image()
            
    def set_white_reference(self, instance):
        if not self.image_widget.start_pos or not self.image_widget.end_pos:
            self.status_label.text = "Please select a region first."
            return

        x1, y1 = self.image_widget.start_pos
        x2, y2 = self.image_widget.end_pos

        img = PILImage.open(self.image_paths[self.current_index])
        img_w, img_h = img.size
        widget_w, widget_h = self.image_widget.size
        scale_x = img_w / widget_w
        scale_y = img_h / widget_h

        left = int(min(x1, x2) * scale_x)
        right = int(max(x1, x2) * scale_x)
        top = int(img_h - max(y1, y2) * scale_y)
        bottom = int(img_h - min(y1, y2) * scale_y)
        
        roi = img.crop((left, top, right, bottom))
        roi_np = np.array(roi)
        self.white_reference = roi_np.mean(axis=(0, 1)).astype(float)[:3]

        self.status_label.text = f"White reference set: RGB {self.white_reference}"
    
    def confirm_selection(self, instance):
        if not self.image_widget.start_pos or not self.image_widget.end_pos:
            self.status_label.text = "Please select a region first."
            return
        self.status_label.text = "Selection confirmed. Ready to analyze."
        self.analyze_btn.disabled = False
        
    def get_selected_rgb(self, instance):
        if not self.image_widget.start_pos or not self.image_widget.end_pos:
            self.status_label.text = "Please select a region first."
            return

        x1, y1 = self.image_widget.start_pos
        x2, y2 = self.image_widget.end_pos

        img = PILImage.open(self.image_paths[self.current_index]).convert("RGB")
        img_w, img_h = img.size
        widget_w, widget_h = self.image_widget.size
        scale_x = img_w / widget_w
        scale_y = img_h / widget_h

        left = int(min(x1, x2) * scale_x)
        right = int(max(x1, x2) * scale_x)
        top = int(img_h - max(y1, y2) * scale_y)
        bottom = int(img_h - min(y1, y2) * scale_y)

        roi = img.crop((left, top, right, bottom))
        roi_np = np.array(roi)
        mean_color = roi_np.mean(axis=(0, 1)).astype(int)[:3]

        self.rgb_label.text = f"RGB: {mean_color[0]}, {mean_color[1]}, {mean_color[2]}"
        self.status_label.text = "RGB extracted from selected region"
        filename = os.path.basename(self.image_paths[self.current_index])
        self.update_table_entry(filename, rgb=list(mean_color))
        
    def save_table_to_csv(self, instance):
        try:
            if not self.analysis_table:
                self.status_label.text = "No data to save."
                return

            filechooser = FileChooserIconView(path=os.getcwd())
            layout = BoxLayout(orientation='vertical')
            layout.add_widget(filechooser)

            name_input = TextInput(hint_text="Enter file name", multiline=False, size_hint_y=None, height=40)
            layout.add_widget(name_input)

            btns = BoxLayout(size_hint_y=None, height=40)
            save_btn = RoundedButton(text="Save")
            cancel_btn = RoundedButton(text="Cancel")
            btns.add_widget(save_btn)
            btns.add_widget(cancel_btn)
            layout.add_widget(btns)

            popup = Popup(title="Save Table to CSV", content=layout, size_hint=(0.85, 0.85))
            def do_save(_):
                    if filechooser.selection and name_input.text.strip():
                        folder = filechooser.selection[0]
                        filename = name_input.text.strip()
                        if not filename.endswith(".csv"):
                            filename += ".csv"
                        filepath = os.path.join(folder, filename)
                        with open(filepath, "w", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow(["Filename", "ΔE", "Status", "R", "G", "B"])
                            for row in self.analysis_table:
                                writer.writerow(row)
                        self.status_label.text = f"CSV saved to: {filepath}"
                        popup.dismiss()
                    else:
                            self.status_label.text = "Select a folder and enter a file name."

            save_btn.bind(on_press=do_save)
            cancel_btn.bind(on_press=popup.dismiss)
            popup.open()

        except Exception as e:
                self.status_label.text = f"Error saving CSV: {e}"
    def export_table_to_csv(self, popup_to_close=None):
        filechooser = FileChooserIconView(path=os.getcwd(), dirselect=False)
        layout = BoxLayout(orientation='vertical', spacing=10)
        layout.add_widget(filechooser)

        name_input = TextInput(hint_text="Enter file name (without .csv)", multiline=False, size_hint_y=None, height=40)
        layout.add_widget(name_input)

        btns = BoxLayout(size_hint_y=None, height=40, spacing=10)
        save_btn = RoundedButton(text="Save")
        cancel_btn = RoundedButton(text="Cancel")
        btns.add_widget(save_btn)
        btns.add_widget(cancel_btn)
        layout.add_widget(btns)
        
        popup = Popup(title="Save Table to CSV", content=layout, size_hint=(0.9, 0.9))

        def save_file(instance):
            if filechooser.selection and name_input.text.strip():
                folder = os.path.dirname(filechooser.selection[0]) if os.path.isfile(filechooser.selection[0]) else filechooser.selection[0]
                filename = os.path.join(folder, name_input.text.strip() + ".csv")
                try:
                    with open(filename, "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(["Filename", "Status", "ΔE", "R", "G", "B"])
                        for fname, row in self.analysis_table.items():
                            writer.writerow([fname] + row)
                    self.status_label.text = f"Saved CSV to: {filename}"
                    popup.dismiss()
                    if popup_to_close:
                        popup_to_close.dismiss()
                except Exception as e:
                    self.status_label.text = f"Error saving CSV: {e}"

        save_btn.bind(on_press=save_file)
        cancel_btn.bind(on_press=popup.dismiss)
        popup.open()
    def export_csv(self, popup_to_close):
        filechooser = FileChooserIconView(path=os.getcwd(), dirselect=True)
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(filechooser)
    
        btns = BoxLayout(size_hint_y=None, height=40)
        ok_btn = RoundedButton(text="Export")
        cancel_btn = RoundedButton(text="Cancel")
        btns.add_widget(ok_btn)
        btns.add_widget(cancel_btn)
        layout.add_widget(btns)
        
        popup = Popup(title="Choose Folder to Save CSV", content=layout, size_hint=(0.9, 0.9))

        def save_to_folder(instance):
            if filechooser.selection:
                dest_folder = filechooser.selection[0]
                filename = "analysis_data.csv" # یا بگیر از کاربر با TextInput
                dest_path = os.path.join(dest_folder, filename)
            
                try:
                    with open(dest_path, "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(["Filename", "ΔE", "Status", "R", "G", "B"])
                        writer.writerows(self.analysis_table)
                    self.status_label.text = f"CSV saved to: {dest_path}"
                except Exception as e:
                    self.status_label.text = f"Failed to save CSV: {str(e)}"

            popup.dismiss()
            popup_to_close.dismiss()

        ok_btn.bind(on_press=save_to_folder)
        cancel_btn.bind(on_press=popup.dismiss)
        popup.open()
    


    def analyze_roi(self, instance):
        if not self.image_paths:
            self.status_label.text = "Load an image first."
            return
        if not self.image_widget.selection_confirmed:
            self.status_label.text = "Confirm selection first."
            return

        x1, y1 = self.image_widget.start_pos
        x2, y2 = self.image_widget.end_pos

        img = PILImage.open(self.image_paths[self.current_index]).convert("RGB")
        img_w, img_h = img.size
        widget_w, widget_h = self.image_widget.size
        scale_x = img_w / widget_w
        scale_y = img_h / widget_h

        left = int(min(x1, x2) * scale_x)
        right = int(max(x1, x2) * scale_x)
        top = int(img_h - max(y1, y2) * scale_y)
        bottom = int(img_h - min(y1, y2) * scale_y)

        roi = img.crop((left, top, right, bottom))
        roi_np = np.array(roi)
        mean_color = roi_np.mean(axis=(0, 1)).astype(float)[:3]

# Apply white reference correction if available
        if self.white_reference is not None:
            correction = np.array([255, 255, 255]) / self.white_reference
            corrected = roi_np[:, :, :3] * correction
            corrected = np.clip(corrected, 0, 255)
            mean_color = corrected.mean(axis=(0, 1)).astype(float)

        # محاسبه میانگین رنگ در RGB
        mean_color = roi_np.mean(axis=(0, 1)).astype(int)[:3]

        # مقایسه با رنگ مرجع در فضای LAB
        initial_color = tuple(self.ref_rgb) # مقدار مرجع (در صورت RGB بودن، باید اصلاح بشه)
        de = delta_e(initial_color, mean_color)

        if de < self.thresholds["fresh"]:
            status = "Fresh"
        elif de < self.thresholds["warning"]:
            status = "Warning"
        else:
            status = "Spoiled"

        self.status_label.text = f"ΔE = {de:.2f} — Status: {status}"
        filename = os.path.basename(self.image_paths[self.current_index])
        self.update_table_entry(filename, delta_e_value=de, status=status)
        self.show_data_table_window()


if __name__ == "__main__":
    ImageApp().run()