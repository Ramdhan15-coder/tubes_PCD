import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QComboBox, QTextEdit
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QTimer

class ImageProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Aplikasi PCD Final - Video Game")
        self.setGeometry(100, 100, 1400, 860) 

        # --- MODEL YOLO (loaded once) ---
        self.yolo_net = None
        self.yolo_classes = None
        self.yolo_output_layers = None
        self.yolo_model_loaded = False

        # --- VIDEO PROCESSING ---
        self.video_cap = None
        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self.process_video_frame)

        label_font = QFont("Arial", 12)
        label_font.setBold(True)

        self.title_original = QLabel("ORIGINAL", self)
        self.title_original.setGeometry(50, 20, 600, 30)
        self.title_original.setAlignment(Qt.AlignCenter)
        self.title_original.setFont(label_font)
        
        self.title_processed = QLabel("PROCESSED", self)
        self.title_processed.setGeometry(750, 20, 600, 30)
        self.title_processed.setAlignment(Qt.AlignCenter)
        self.title_processed.setFont(label_font)

        # --- UI SETUP ---
        self.original_label = QLabel(self)
        self.original_label.setGeometry(50, 50, 600, 450); self.original_label.setStyleSheet("border: 1px solid black; background-color: #f0f0f0;"); self.original_label.setAlignment(Qt.AlignCenter); self.original_label.setText("Original Image / Video")
        self.processed_label = QLabel(self); self.processed_label.setGeometry(750, 50, 600, 450); self.processed_label.setStyleSheet("border: 1px solid black; background-color: #f0f0f0;"); self.processed_label.setAlignment(Qt.AlignCenter); self.processed_label.setText("Processed Image / Video")
        
        y_pos_main_actions, x_pos_col1 = 520, 50
        self.btn_open = QPushButton("1. Open Image", self); self.btn_open.setGeometry(x_pos_col1, y_pos_main_actions, 150, 30); self.btn_open.clicked.connect(self.open_image_file)
        self.btn_extract_features = QPushButton("2. Extract Features", self); self.btn_extract_features.setGeometry(x_pos_col1, y_pos_main_actions + 40, 150, 30); self.btn_extract_features.clicked.connect(self.extract_features)
        self.btn_detect_faces = QPushButton("3. Detect Face (Haar)", self); self.btn_detect_faces.setGeometry(x_pos_col1, y_pos_main_actions + 80, 150, 30); self.btn_detect_faces.clicked.connect(self.detect_faces)
        self.btn_detect_objects_yolo_img = QPushButton("4. Detect Objects (Img)", self) 
        self.btn_detect_objects_yolo_img.setGeometry(x_pos_col1, y_pos_main_actions + 120, 150, 30)
        self.btn_detect_objects_yolo_img.clicked.connect(self.detect_objects_yolo_on_static_image)
        self.btn_open_video = QPushButton("5. Open Video", self) 
        self.btn_open_video.setGeometry(x_pos_col1, y_pos_main_actions + 160, 150, 30) 
        self.btn_open_video.clicked.connect(self.open_video_file)

        x_pos_filters = 220; y_pos_filters = 520
        self.combo_sharpen = QComboBox(self); self.combo_sharpen.setGeometry(x_pos_filters, y_pos_filters, 200, 30); self.combo_sharpen.addItems(["Kernel Sharpening", "Unsharp Masking", "High Boost Filtering", "Laplace Operator", "HD Sharpening", "Gaussian Blur (Smoothing)", "Edge Detection (Sobel)", "Edge Detection (Laplacian)", "Edge Detection (Canny)", "Emboss"])
        self.btn_sharpen = QPushButton("Apply Spatial Filter", self); self.btn_sharpen.setGeometry(x_pos_filters + 210, y_pos_filters, 150, 30); self.btn_sharpen.clicked.connect(self.apply_sharpening)
        self.combo_histogram = QComboBox(self); self.combo_histogram.setGeometry(x_pos_filters, y_pos_filters + 40, 200, 30); self.combo_histogram.addItems(["Histogram Equalization", "CLAHE (Color)"])
        self.btn_histogram = QPushButton("Apply Contrast", self); self.btn_histogram.setGeometry(x_pos_filters + 210, y_pos_filters + 40, 150, 30); self.btn_histogram.clicked.connect(self.apply_histogram_equalization)
        self.combo_frequency = QComboBox(self); self.combo_frequency.setGeometry(x_pos_filters, y_pos_filters + 80, 200, 30); self.combo_frequency.addItems(["High Pass (FFT)", "Low Pass (FFT)"])
        self.btn_frequency = QPushButton("Apply Frequency Filter", self); self.btn_frequency.setGeometry(x_pos_filters + 210, y_pos_filters + 80, 150, 30); self.btn_frequency.clicked.connect(self.apply_frequency_filter)
        self.btn_histogram_viewer = QPushButton("Show Histogram", self); self.btn_histogram_viewer.setGeometry(x_pos_filters, y_pos_filters + 120, 200, 30); self.btn_histogram_viewer.clicked.connect(self.show_histogram)
        
        x_pos_utils = 850; y_pos_utils = 520
        self.btn_gray = QPushButton("Convert to Grayscale", self); self.btn_gray.setGeometry(x_pos_utils, y_pos_utils, 150, 30); self.btn_gray.clicked.connect(self.convert_grayscale)
        self.btn_binary = QPushButton("Convert to Binary", self); self.btn_binary.setGeometry(x_pos_utils, y_pos_utils + 40, 150, 30); self.btn_binary.clicked.connect(self.convert_binary)
        self.btn_save_img = QPushButton("Save Image/Frame", self); self.btn_save_img.setGeometry(x_pos_utils + 160, y_pos_utils, 150, 30); self.btn_save_img.clicked.connect(self.save_image)
        self.btn_export_data = QPushButton("Export Pixel Data", self); self.btn_export_data.setGeometry(x_pos_utils + 160, y_pos_utils + 40, 150, 30); self.btn_export_data.clicked.connect(self.export_pixel_data)
        
        self.text_edit = QTextEdit(self)
        self.text_edit.setGeometry(50, 715, 1300, 130)
        self.text_edit.setFont(QFont("Courier New", 13)) 
        self.text_edit.setReadOnly(True)

        self.img = None; self.processed_img = None

    def display_image(self, image_data, label_widget):
        if image_data is None: label_widget.setText("No Image/Video"); return
        if len(image_data.shape) == 2: image_data = cv2.cvtColor(image_data, cv2.COLOR_GRAY2RGB)
        h, w, c = image_data.shape; bpl = 3 * w
        qimg = QImage(image_data.data, w, h, bpl, QImage.Format_RGB888); pixmap = QPixmap.fromImage(qimg)
        label_widget.setPixmap(pixmap.scaled(label_widget.width(), label_widget.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def open_image_file(self):
        if self.video_timer.isActive(): self.stop_video_processing()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.img = cv2.imread(file_path)
            if self.img is None: self.text_edit.setText(f"Gagal memuat gambar: {file_path}"); return
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB); self.processed_img = self.img.copy()
            self.display_image(self.img, self.original_label); self.display_image(self.processed_img, self.processed_label)
            self.text_edit.setText("Gambar berhasil dimuat! Silakan pilih aksi."); self.show_pixel_values(self.img, "Original Image")

    def open_video_file(self):
        if self.video_timer.isActive(): self.stop_video_processing(); return
        video_path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Videos (*.mp4 *.avi *.mkv *.mov)")
        if video_path:
            self.video_cap = cv2.VideoCapture(video_path)
            if not self.video_cap.isOpened(): self.text_edit.setText(f"Error: Tidak bisa membuka video di {video_path}."); self.video_cap = None; return
            if not self.load_yolo_model_if_needed(): self.stop_video_processing("Gagal memuat model YOLO untuk video."); return
            self.text_edit.setText(f"Memutar video: {video_path}.\nKlik tombol 'Open Video' lagi untuk berhenti.")
            self.original_label.setText("Original Video Frame"); self.processed_label.setText("Processed Video Frame")
            self.video_timer.start(33) 
        else: self.text_edit.setText("Pemilihan video dibatalkan.")

    def process_video_frame(self):
        if self.video_cap and self.video_cap.isOpened():
            ret, frame = self.video_cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.display_image(frame_rgb, self.original_label)
                processed_frame = self.process_frame_with_yolo(frame_rgb)
                self.display_image(processed_frame, self.processed_label)
                self.processed_img = processed_frame 
            else: self.stop_video_processing("Video selesai atau gagal membaca frame.")
        else: self.stop_video_processing() 
            
    def stop_video_processing(self, message="Pemutaran video dihentikan."):
        self.video_timer.stop()
        if self.video_cap is not None: self.video_cap.release()
        self.video_cap = None; self.text_edit.setText(message)
        self.original_label.setText("Original Image / Video"); self.processed_label.setText("Processed Image / Video")
        self.img = None; self.processed_img = None

    def load_yolo_model_if_needed(self):
        if not self.yolo_model_loaded:
            self.text_edit.setText("Memuat model YOLO... Mohon tunggu."); QApplication.processEvents()
            try:
                self.yolo_net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
                with open("coco.names", "r") as f: self.yolo_classes = [line.strip() for line in f.readlines()]
                layer_names = self.yolo_net.getLayerNames()
                try: self.yolo_output_layers = [layer_names[i - 1] for i in self.yolo_net.getUnconnectedOutLayers()]
                except AttributeError: self.yolo_output_layers = [layer_names[i[0] - 1] for i in self.yolo_net.getUnconnectedOutLayers()]
                self.yolo_model_loaded = True; self.text_edit.append("\nModel YOLO berhasil dimuat."); return True
            except Exception as e:
                self.text_edit.setText(f"Error: Gagal memuat model YOLO. Pastikan file .weights, .cfg, dan coco.names ada.\nDetail: {e}")
                self.yolo_model_loaded = False; return False
        return True

    def process_frame_with_yolo(self, frame_to_process):
        if not self.yolo_model_loaded or self.yolo_net is None: return frame_to_process
        img_for_detection = frame_to_process.copy(); height, width, _ = img_for_detection.shape
        blob = cv2.dnn.blobFromImage(img_for_detection, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.yolo_net.setInput(blob); outs = self.yolo_net.forward(self.yolo_output_layers)
        class_ids, confidences, boxes = [], [], []
        CONF_THRESHOLD, NMS_THRESHOLD = 0.5, 0.4
        for out in outs:
            for detection in out:
                scores = detection[5:]; class_id = np.argmax(scores); confidence = scores[class_id]
                if confidence > CONF_THRESHOLD:
                    center_x, center_y, w_box, h_box = int(detection[0]*width), int(detection[1]*height), int(detection[2]*width), int(detection[3]*height)
                    x, y = int(center_x - w_box / 2), int(center_y - h_box / 2)
                    boxes.append([x, y, w_box, h_box]); confidences.append(float(confidence)); class_ids.append(class_id)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
        font = cv2.FONT_HERSHEY_PLAIN; colors = np.random.uniform(0, 255, size=(len(self.yolo_classes), 3))
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w_box, h_box = boxes[i]
                if 0 <= class_ids[i] < len(self.yolo_classes): 
                    label = str(self.yolo_classes[class_ids[i]])
                    confidence_label = f" {confidences[i]*100:.2f}%"; color = colors[class_ids[i]]
                    cv2.rectangle(img_for_detection, (x, y), (x + w_box, y + h_box), color, 2)
                    cv2.putText(img_for_detection, label + confidence_label, (x, y - 5), font, 1.5, color, 2)
        return img_for_detection

    def detect_objects_yolo_on_static_image(self):
        if self.img is None: self.text_edit.setText("Buka gambar dulu broo!"); return
        if not self.load_yolo_model_if_needed(): return
        self.text_edit.setText("Memproses deteksi objek pada gambar dengan YOLOv4-tiny..."); QApplication.processEvents()
        processed_image = self.process_frame_with_yolo(self.img.copy())
        self.processed_img = processed_image; self.display_image(self.processed_img, self.processed_label)
        self.text_edit.append("\nDeteksi objek pada gambar selesai.")

    def detect_faces(self):
        if self.img is None: self.text_edit.setText("Buka gambar dulu broo!"); return
        cascade_file = 'haarcascade_frontalface_default.xml'
        try:
            face_cascade = cv2.CascadeClassifier(cascade_file)
            if face_cascade.empty(): raise IOError(f"File tidak ditemukan: {cascade_file}")
        except Exception as e: self.text_edit.setText(str(e)); return
        img_for_detection = self.img.copy(); gray_img = cv2.cvtColor(img_for_detection, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces: cv2.rectangle(img_for_detection, (x, y), (x+w, y+h), (0, 255, 0), 3)
        self.processed_img = img_for_detection; self.display_image(self.processed_img, self.processed_label)
        self.text_edit.clear(); self.text_edit.append(f"--- Face Detection Result ---\nMenggunakan model: {cascade_file}\nTerdeteksi {len(faces)} wajah.")
        if len(faces) == 0: self.text_edit.append("\nTips: Coba 'lbpcascade_animeface.xml' jika wajah anime.")

    def show_pixel_values(self, image, label):
        if image is None: self.text_edit.append(f"\nTidak ada data pixel untuk {label}"); return
        self.text_edit.append(f"\n--- {label} Pixel Values (Contoh 5x5) ---")
        if len(image.shape) == 2:
            for row in image[:5, :5]: self.text_edit.append("  ".join(f"{val:3}" for val in row))
        else:
            for row in image[:5, :5, :]: self.text_edit.append("  ".join(f"[{r:03},{g:03},{b:03}]" for r, g, b in row))
        self.text_edit.append("")
        
    def apply_sharpening(self): 
        if self.img is None: self.text_edit.setText("Buka gambar dulu!"); return
        method = self.combo_sharpen.currentText(); processed = self.img.copy(); gray_mode = False
        if method == "Kernel Sharpening": kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]); processed = cv2.filter2D(self.img, -1, kernel)
        elif method == "Unsharp Masking": blurred = cv2.GaussianBlur(self.img, (5,5), 1.0); processed = cv2.addWeighted(self.img, 1.5, blurred, -0.5, 0)
        elif method == "High Boost Filtering": blurred = cv2.GaussianBlur(self.img, (5,5), 1.0); k=1.5; processed = cv2.addWeighted(self.img, 1+k, blurred, -k, 0)
        elif method == "Laplace Operator": laplacian = cv2.Laplacian(self.img, cv2.CV_64F); sharpened = cv2.convertScaleAbs(laplacian); processed = cv2.addWeighted(self.img, 1.0, sharpened, 1.0, 0)
        elif method == "HD Sharpening": processed = cv2.detailEnhance(self.img, sigma_s=10, sigma_r=0.15)
        elif method == "Gaussian Blur (Smoothing)": processed = cv2.GaussianBlur(self.img, (5,5), 0)
        elif method in ["Edge Detection (Sobel)", "Edge Detection (Laplacian)", "Edge Detection (Canny)"]:
            gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY); gray_mode = True
            if method == "Edge Detection (Sobel)": sobelx = cv2.Sobel(gray, cv2.CV_64F, 1,0, ksize=3); sobely = cv2.Sobel(gray, cv2.CV_64F, 0,1, ksize=3); edges = cv2.magnitude(sobelx, sobely); processed = cv2.convertScaleAbs(edges)
            elif method == "Edge Detection (Laplacian)": lap = cv2.Laplacian(gray, cv2.CV_64F); processed = cv2.convertScaleAbs(lap)
            elif method == "Edge Detection (Canny)": processed = cv2.Canny(gray, 100, 200)
        elif method == "Emboss": kernel = np.array([[-2,-1,0],[-1,1,1],[0,1,2]]); embossed = cv2.filter2D(self.img, -1, kernel); processed = cv2.convertScaleAbs(embossed)
        self.processed_img = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB) if gray_mode else processed
        self.display_image(self.processed_img, self.processed_label); self.show_pixel_values(self.processed_img, f"Processed ({method})")

    def apply_frequency_filter(self):
        if self.img is None: self.text_edit.setText("Buka gambar dulu!"); return
        method = self.combo_frequency.currentText(); gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT); dft_shift = np.fft.fftshift(dft)
        rows, cols = gray.shape; crow, ccol = rows//2, cols//2; r=30; mask = np.zeros((rows,cols,2), np.uint8)
        if method == "High Pass (FFT)": mask[:,:]=1; mask[crow-r:crow+r, ccol-r:ccol+r]=0
        elif method == "Low Pass (FFT)": mask[crow-r:crow+r, ccol-r:ccol+r]=1
        fshift = dft_shift*mask; f_ishift = np.fft.ifftshift(fshift); img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1]); img_back = cv2.normalize(img_back, None, 0,255, cv2.NORM_MINMAX)
        self.processed_img = cv2.cvtColor(np.uint8(img_back), cv2.COLOR_GRAY2RGB)
        self.display_image(self.processed_img, self.processed_label); self.show_pixel_values(self.processed_img, f"Processed ({method})")

    def apply_histogram_equalization(self):
        if self.img is None: self.text_edit.setText("Buka gambar dulu!"); return
        method = self.combo_histogram.currentText()
        if method == "Histogram Equalization": gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY); eq = cv2.equalizeHist(gray); self.processed_img = cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB)
        elif method == "CLAHE (Color)": lab = cv2.cvtColor(self.img, cv2.COLOR_RGB2LAB); l,a,b = cv2.split(lab); clahe=cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)); cl=clahe.apply(l); merged=cv2.merge((cl,a,b)); self.processed_img=cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        self.display_image(self.processed_img, self.processed_label); self.show_pixel_values(self.processed_img, f"Processed ({method})")

    def show_histogram(self):
        current_img_for_hist = self.processed_img if self.processed_img is not None else self.img
        if current_img_for_hist is None: self.text_edit.setText("Buka atau proses gambar dulu!"); return
        plt.figure("Histogram"); img_hist = current_img_for_hist.copy()
        if len(img_hist.shape)==2 or img_hist.shape[2]==1: plt.hist(img_hist.ravel(),256,[0,256], color='gray'); plt.title("Grayscale Histogram")
        else:
            color = ('r','g','b'); plt.title("RGB Histogram")
            for i,col_char in enumerate(color): plt.hist(img_hist[:,:,i].ravel(),256,[0,256], color=col_char, alpha=0.5, label=col_char.upper())
            plt.legend()
        plt.xlabel("Pixel Value"); plt.ylabel("Frequency"); plt.tight_layout(); plt.show()

    def convert_grayscale(self):
        if self.img is None: self.text_edit.setText("Buka gambar dulu!"); return
        gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY); self.processed_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        self.display_image(self.processed_img, self.processed_label); self.show_pixel_values(gray, "Grayscale")

    def convert_binary(self):
        if self.img is None: self.text_edit.setText("Buka gambar dulu!"); return
        gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY); _, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
        self.processed_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB); self.display_image(self.processed_img, self.processed_label); self.show_pixel_values(binary, "Binary")

    def save_image(self):
        img_to_save = self.processed_img if self.processed_img is not None else self.img
        if img_to_save is None: self.text_edit.setText("Tidak ada gambar untuk disimpan!"); return
        filePath, _ = QFileDialog.getSaveFileName(self, "Save Image/Frame", "", "PNG(*.png);;JPEG(*.jpg *.jpeg)")
        if filePath: cv2.imwrite(filePath, cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR)); self.text_edit.setText(f"Gambar disimpan ke {filePath}")

    def export_pixel_data(self):
        img_to_export = self.processed_img if self.processed_img is not None else self.img
        if img_to_export is None: self.text_edit.setText("Tidak ada gambar untuk ekspor data pixel!"); return
        if len(img_to_export.shape) == 3: h,w,c = img_to_export.shape; reshaped = img_to_export.reshape(-1,3); df = pd.DataFrame(reshaped, columns=["R","G","B"])
        else: h,w = img_to_export.shape; reshaped = img_to_export.reshape(-1,1); df = pd.DataFrame(reshaped, columns=["Grayscale"])
        filePath, _ = QFileDialog.getSaveFileName(self, "Save Pixel Data", "", "Excel Files (*.xlsx);;Text Files (*.txt)")
        if filePath:
            try:
                if filePath.endswith(".txt"): df.to_csv(filePath, index=False, sep='\t')
                else: df.to_excel(filePath, index=False)
                self.text_edit.setText(f"Data pixel disimpan ke {filePath}")
            except Exception as e: self.text_edit.setText(f"Error menyimpan data: {e}")

    def extract_features(self):
        current_image = self.processed_img if self.processed_img is not None else self.img
        if current_image is None : self.text_edit.setText("Buka gambar atau video dulu!"); return
        self.text_edit.clear(); self.text_edit.append("--- Menganalisis Fitur Citra ---")
        mean, std_dev = cv2.meanStdDev(current_image)
        self.text_edit.append(f"\nMean (R,G,B): [{mean[2][0]:.2f}, {mean[1][0]:.2f}, {mean[0][0]:.2f}]")
        self.text_edit.append(f"Std Dev (R,G,B): [{std_dev[2][0]:.2f}, {std_dev[1][0]:.2f}, {std_dev[0][0]:.2f}]")
        if len(current_image.shape) == 3 and current_image.shape[2] == 3: gray_img = cv2.cvtColor(current_image, cv2.COLOR_RGB2GRAY)
        elif len(current_image.shape) == 2: gray_img = current_image
        else: self.text_edit.append("Format tidak didukung untuk ORB."); return
        orb = cv2.ORB_create(500); keypoints, _ = orb.detectAndCompute(gray_img, None)
        self.text_edit.append(f"\nJumlah Keypoint ORB: {len(keypoints)}")
        img_with_keypoints = cv2.drawKeypoints(current_image.copy(), keypoints, None, color=(0,255,0), flags=0)
        self.display_image(img_with_keypoints, self.processed_label); self.text_edit.append("\nKeypoints ditampilkan di panel 'Processed Image'.")
    
    def closeEvent(self, event):
        self.stop_video_processing("Aplikasi ditutup.")
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessingApp()
    window.show()
    sys.exit(app.exec_())