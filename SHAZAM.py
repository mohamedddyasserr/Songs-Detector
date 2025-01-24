import os
import sys
import json
import numpy as np
import soundfile as sf
from scipy.fftpack import dct
from sklearn.metrics.pairwise import cosine_similarity
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                                QLabel, QFileDialog, QSlider, QTableWidget, QTableWidgetItem, QMessageBox,QProgressBar, QFrame)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap, QPalette, QColor, QMouseEvent, QPen
import imagehash
from PIL import Image
import matplotlib.pyplot as plt
import os
import re

# folder of songs
SONGS_FOLDER = 'E:\\DSP\\Task5\\songs'

class ProcessSongsThread(QThread): 
    progress = pyqtSignal(int) 
    def __init__(self, folder_path, json_file): 
        super().__init__() 
        self.folder_path = folder_path 
        self.json_file = json_file 
        
    def run(self): 
        print("Thread started")
        process_songs(self.folder_path, self.json_file, self.progress)
        self.progress.emit(100)
        print("Thread finished")

    

# Function to convert audio file to array
def audio_to_array(file_path):
    audio_array, framerate = sf.read(file_path)
    if len(audio_array.shape) == 2:
        audio_array = audio_array.mean(axis=1)  # Convert to mono by averaging channels
    return audio_array, framerate

# Function to extract features from audio data
def extract_features(audio_array):
    N = len(audio_array)  # Length of the audio array
    
    # Apply a Hamming window to the audio signal to reduce spectral leakage
    windowed = audio_array * np.hamming(N)
    
    # make a Fast Fourier Transform (FFT) on the windowed signal to get the frequency spectrum
    transformed = np.abs(np.fft.rfft(windowed))
    
    # get the power spectrum (squared magnitudes of the FFT results)
    power = transformed**2
    
    # Apply a logarithmic scale to the power spectrum to match human hearing perception
    log_power = np.log(power + 1e-10) 
    
    # Compute the Mel-Frequency Cepstral Coefficients (MFCCs) from the log power spectrum
    # Use the Discrete Cosine Transform (DCT) to obtain the cepstral coefficients
    mfccs = dct(log_power, type=2, axis=0, norm='ortho')[:13]

    # performs a Discrete Cosine Transform on the logarithmic power spectrum, normalizes the result, 
    # and selects the first 13 coefficients to obtain the MFCCs. 
    # These MFCCs are a compact representation of the audio signal, capturing the most important features for tasks like audio comparison and recognition.
    # MFCCs are a set of coefficients that collectively represent the short-term power spectrum of a sound.

    return mfccs.tolist()


# Function to generate perceptual hash from audio array
def generate_perceptual_hash(audio_array):
    plt.specgram(audio_array, NFFT=2048, noverlap=1024)
    plt.axis('off')
    plt.savefig("temp_spectrogram.png", bbox_inches='tight', pad_inches=0)
    plt.close()
    image = Image.open("temp_spectrogram.png")
    phash = imagehash.phash(image)
    os.remove("temp_spectrogram.png")
    return str(phash)  # Convert to string for JSON serialization

# Function to generate fingerprint
def generate_fingerprint(file_path):    
    audio_array, _ = audio_to_array(file_path)
    features = extract_features(audio_array)
    phash = generate_perceptual_hash(audio_array)
    return {"features": features, "phash": phash}

# Function to process all songs and save fingerprints to JSON
def process_songs(folder_path, json_file, progress_callback): 
    fingerprints = {} 
    files = [f for f in os.listdir(folder_path) if f.endswith('.wav')] 
    total_files = len(files) 
    for idx, file in enumerate(files): 
        file_path = os.path.join(folder_path, file) 
        fingerprint = generate_fingerprint(file_path) 
        song_name, group_number, song_type = extract_info_from_filename(file) 
        fingerprint.update({"song_name": song_name, "group_number": group_number, "type": song_type})
        fingerprints[file] = fingerprint 
        with open(json_file, 'w') as f: 
            json.dump(fingerprints, f, indent=4)
        print("Processing complete") # Debug statement

# Function to load fingerprints from JSON
def load_fingerprints(json_file):
    with open(json_file, 'r') as f:
        fingerprints = json.load(f)
    return fingerprints

# Function to calculate similarity using cosine similarity and Hamming distance
def calculate_similarity(fingerprint1, fingerprint2):
    features1, phash1 = fingerprint1["features"], fingerprint1["phash"]
    features2, phash2 = fingerprint2["features"], fingerprint2["phash"]
    feature_similarity = cosine_similarity([features1], [features2]).mean()
    hash_similarity = 1 - (imagehash.hex_to_hash(phash1) - imagehash.hex_to_hash(phash2)) / len(imagehash.hex_to_hash(phash1).hash.flatten())
    return (feature_similarity + hash_similarity) / 2

# Function to find closest songs
def find_closest_songs(fingerprints, target_fingerprint):
    similarities = {}
    for key, fingerprint in fingerprints.items():
        similarity = calculate_similarity(target_fingerprint, fingerprint)
        similarities[key] = similarity * 100  # Convert to percentage
    sorted_songs = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    return sorted_songs

# Function to create weighted average of two audio files
def weighted_average(file1, file2, weight1, weight2):
    audio_array1, framerate1 = audio_to_array(file1)
    audio_array2, framerate2 = audio_to_array(file2)
    if framerate1 != framerate2:
        audio_array2 = resample_audio(audio_array2, framerate2, framerate1)
    length = min(len(audio_array1), len(audio_array2)) # Ensure both arrays are of the same length 
    audio_array1 = audio_array1[:length] 
    audio_array2 = audio_array2[:length]
    weighted_avg = weight1 * audio_array1 + weight2 * audio_array2
    return weighted_avg, framerate1

# Function to resample audio array to a different sample rate
def resample_audio(audio_array, original_rate, target_rate): 
    duration = len(audio_array) / original_rate 
    target_length = int(duration * target_rate) 
    resampled_audio = np.interp(np.linspace(0, len(audio_array), target_length), np.arange(len(audio_array)), audio_array)
    return resampled_audio


def extract_info_from_filename(filename):
    """
       Extracts song name, group number, and type from the filename. 
       Assumes filenames are in the format 'Groupnumber_songName_type.ext'. 
    """ 
    name_part, ext = os.path.splitext(filename) 
    parts = name_part.split('_') 
    if len(parts) >= 3: 
        group_number = parts[0] # e.g., "Group1" 
        song_name = '_'.join(parts[1:-1]).replace("-", " ").strip() # e.g., "SomeSongName" 
        song_type = parts[-1].replace("(", "").replace(")", "").replace("_", " ").strip() # e.g., "vocals" 
        # Ensure proper formatting 
        group_number = group_number.replace("Group", "Group ").strip()
    else: 
        group_number, song_name, song_type = "", "", "" 
    return song_name, group_number, song_type


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.apply_dark_theme()

    def initUI(self):
        self.main_layout = QHBoxLayout()
        self.right_layout = QVBoxLayout()
        self.bottom_layout = QHBoxLayout()

        self.left_layout = QVBoxLayout()

        # Logo display 
        self.logo_label = QLabel(self) 
        self.logo_pixmap = QPixmap("E:\DSP\Task5\logo.png") # Update with the actual logo path 
        self.logo_label.setPixmap(self.logo_pixmap) 
        self.left_layout.addWidget(self.logo_label)

        self.target_label = QLabel("Select Your Song:")
        self.left_layout.addWidget(self.target_label)
        self.target_button = QPushButton("Browse")
        self.target_button.clicked.connect(self.select_target_file)
        self.left_layout.addWidget(self.target_button)

        self.find_button = QPushButton("Find Similar Songs")
        self.find_button.clicked.connect(self.find_songs)
        self.left_layout.addWidget(self.find_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.left_layout.addWidget(self.progress_bar)

        self.mix_label1 = QLabel("Select First Song to Mix:")
        self.left_layout.addWidget(self.mix_label1)
        self.mix_button1 = QPushButton("Browse")
        self.mix_button1.clicked.connect(self.select_mix_file1)
        self.left_layout.addWidget(self.mix_button1)

        self.mix_label2 = QLabel("Select Second Song to Mix:")
        self.left_layout.addWidget(self.mix_label2)
        self.mix_button2 = QPushButton("Browse")
        self.mix_button2.clicked.connect(self.select_mix_file2)
        self.left_layout.addWidget(self.mix_button2)

        self.slider_label = QLabel("Set Weight Percentage for Both Songs:")
        self.left_layout.addWidget(self.slider_label)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(50)
        self.slider.valueChanged.connect(self.update_slider_labels)
        self.left_layout.addWidget(self.slider)

        self.slider_layout = QHBoxLayout() 
        self.weight1_label = QLabel("First Song: 50%") 
        self.weight2_label = QLabel("Second Song: 50%") 
        self.slider_layout.addWidget(self.weight1_label)
        self.slider_layout.addWidget(self.weight2_label)
        self.left_layout.addLayout(self.slider_layout)

        self.mix_button = QPushButton("Mix and Find Similar Songs")
        self.mix_button.clicked.connect(self.mix_and_find)
        self.left_layout.addWidget(self.mix_button)

        self.main_layout.addLayout(self.left_layout)

        self.result_table = QTableWidget()
        self.result_table.setColumnCount(4)
        self.result_table.setHorizontalHeaderLabels(["Song Name", "Type", "Group Number", "Similarity (%)"]) 
        self.result_table.setMinimumWidth(1500) # Set minimum width for the table
        self.result_table.setColumnWidth(0, 375)  
        self.result_table.setColumnWidth(1, 375) 
        self.result_table.setColumnWidth(2, 375) 
        self.result_table.setColumnWidth(3, 375) 
        self.right_layout.addWidget(self.result_table)

        self.details_box = QFrame()
        self.details_box.setFrameShape(QFrame.Box) 
        self.details_box.setFrameShadow(QFrame.Raised) 
        self.details_box.setLineWidth(2)
        self.details_box.setStyleSheet("background-color: #1E1E1E; color: #FFFFFF;")
        self.details_box.setFixedSize(700, 225) # Set fixed size to ensure rectangle shape

        self.details_layout = QHBoxLayout(self.details_box) # Layout for details at the bottom 
        self.details_left_layout = QVBoxLayout() 
        self.details_right_layout = QVBoxLayout() 
        self.song_info_label = QLabel("You are searching for:") 
        self.song_name_label = QLabel("") 
        self.song_name_label.setStyleSheet("font-size: 25px; font-weight: bold; font-style: italic;")
        self.song_picture_label = QLabel("")
        self.song_picture_label.setFixedSize(200, 200)
        self.details_left_layout.addWidget(self.song_info_label) 
        self.details_left_layout.addWidget(self.song_name_label) 
        self.details_right_layout.addWidget(self.song_picture_label)
        self.details_layout.addLayout(self.details_left_layout) 
        self.details_layout.addLayout(self.details_right_layout) 
        self.right_layout.addWidget(self.details_box, alignment=Qt.AlignCenter)
        self.right_layout.addLayout(self.details_layout)
        

        self.main_layout.addLayout(self.right_layout)

        self.setLayout(self.main_layout)
        self.setWindowTitle("Our SHAZAM!")

    

    def update_slider_labels(self): 
        weight1 = self.slider.value() 
        weight2 = 100 - weight1 
        self.weight1_label.setText(f"First File: {weight1}%") 
        self.weight2_label.setText(f"Second File: {weight2}%")

    def select_target_file(self):
        self.target_file, _ = QFileDialog.getOpenFileName(self, "Select Target Song", "", "WAV files (*.wav)")
        self.target_label.setText(f"Selected Your Song: {"Loaded Successfully!"}")

    def select_mix_file1(self):
        self.mix_file1, _ = QFileDialog.getOpenFileName(self, "Select First File", "", "WAV files (*.wav)")
        top_song_name, top_group_number, top_song_type = extract_info_from_filename(self.mix_file1) 
        self.mix_label1.setText(f"Selected First File: \n {top_song_name , (top_song_type)}")
        
        
        # self.mix_label1.setText(f"Selected First File: {"Loaded Successfully!"}")

    def select_mix_file2(self):
        self.mix_file2, _ = QFileDialog.getOpenFileName(self, "Select Second File", "", "WAV files (*.wav)")
        top_song_name, top_group_number, top_song_type = extract_info_from_filename(self.mix_file2) 
        self.mix_label2.setText(f"Selected First File: \n {(top_song_name) , (top_song_type)}")
        

        # self.mix_label2.setText(f"Selected Second File: {"Loaded Successfully!"}")

    def find_songs(self):
        json_file = os.path.join(SONGS_FOLDER, "fingerprints.json")
       
            
        self.process_thread = ProcessSongsThread(SONGS_FOLDER, json_file) 
        self.process_thread.progress.connect(self.update_progress) 
        self.process_thread.finished.connect(self.on_find_songs_complete) 
        self.process_thread.start()

        print("Thread started in find_songs")

    def update_progress(self, value):
        print(f"Updating progress bar: {value}%")
        self.progress_bar.setValue(value)

    def on_find_songs_complete(self): 
        print("Thread finished and find_songs_complete called")
        json_file = os.path.join(SONGS_FOLDER, "fingerprints.json") 
        fingerprints = load_fingerprints(json_file) 
        target_fingerprint = generate_fingerprint(self.target_file) 
        closest_songs = find_closest_songs(fingerprints, target_fingerprint) 

        self.result_table.setRowCount(0) 

        if closest_songs: 
            # Display top matching song info 
            top_song_filename, _ = closest_songs[0] 
            top_song_name, top_group_number, top_song_type = extract_info_from_filename(top_song_filename) 
            self.song_name_label.setText(f"                    {top_song_name}") # Display the image associated with the top matching song

            top_group_folder = os.path.join(SONGS_FOLDER, f'Team_{top_group_number.strip().replace("Group ", "")}') 
            image_path = os.path.join(top_group_folder, 'image.png')

            print(f"Looking for image at: {image_path}")
            if os.path.exists(image_path): 
                top_song_image = QPixmap(image_path) 
                top_song_image = top_song_image.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.song_picture_label.setPixmap(top_song_image) 
            else: 
                self.song_picture_label.setText("No image available")

        for i, (file_name, similarity) in enumerate(closest_songs):  
            song_name, group_number, song_type = extract_info_from_filename(file_name)

            self.result_table.insertRow(i) 
            self.result_table.setItem(i, 0, QTableWidgetItem(song_name)) 
            self.result_table.setItem(i, 1, QTableWidgetItem(song_type)) 
            self.result_table.setItem(i, 2, QTableWidgetItem(group_number)) 
            self.result_table.setItem(i, 3, QTableWidgetItem(f"{similarity:.2f}%"))
        self.progress_bar.setValue(0) # Reset the progress bar


    def mix_and_find(self):
        try:
            weight1 = self.slider.value() / 100
            weight2 = 1 - weight1
            mixed_audio, framerate = weighted_average(self.mix_file1, self.mix_file2, weight1, weight2)

            # Save the mixed audio to a temporary file
            mixed_file_path = "mixed_audio.wav"
            sf.write(mixed_file_path, mixed_audio, framerate)

            # Generate fingerprint for the mixed audio
            mixed_fingerprint = generate_fingerprint(mixed_file_path)

            # Load fingerprints from JSON and find closest matches to the mixed audio
            json_file = os.path.join(SONGS_FOLDER, "fingerprints.json")
            fingerprints = load_fingerprints(json_file)
            closest_songs = find_closest_songs(fingerprints, mixed_fingerprint)

            # Display the results in the table
            self.result_table.setRowCount(0)


            if closest_songs: 
                # Display top matching song info 
                top_song_filename, _ = closest_songs[0] 
                top_song_name, top_group_number, top_song_type = extract_info_from_filename(top_song_filename) 
                self.song_name_label.setText(f"                    {top_song_name}") # Display the image associated with the top matching song

                top_group_folder = os.path.join(SONGS_FOLDER, f'Team_{top_group_number.strip().replace("Group ", "")}') 
                image_path = os.path.join(top_group_folder, 'image.png')

                print(f"Looking for image at: {image_path}")
                if os.path.exists(image_path): 
                    top_song_image = QPixmap(image_path) 
                    top_song_image = top_song_image.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.song_picture_label.setPixmap(top_song_image) 
                else: 
                    self.song_picture_label.setText("No image available")


            for i, (file_name, similarity) in enumerate(closest_songs):  # Show top 10 closest songs
                song_name, group_number, song_type = extract_info_from_filename(file_name)

                self.result_table.insertRow(i) 
                self.result_table.setItem(i, 0, QTableWidgetItem(song_name)) 
                self.result_table.setItem(i, 1, QTableWidgetItem(song_type)) 
                self.result_table.setItem(i, 2, QTableWidgetItem(group_number)) 
                self.result_table.setItem(i, 3, QTableWidgetItem(f"{similarity:.2f}%"))
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def apply_dark_theme(self):
        dark_palette = QPalette()

        # Background
        dark_palette.setColor(QPalette.Window, QColor(45, 45, 45))
        dark_palette.setColor(QPalette.WindowText, Qt.white)

        # Base color
        dark_palette.setColor(QPalette.Base, QColor(30, 30, 30))
        dark_palette.setColor(QPalette.AlternateBase, QColor(45, 45, 45))

        # Tooltips
        dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
        dark_palette.setColor(QPalette.ToolTipText, Qt.white)

        # Text
        dark_palette.setColor(QPalette.Text, Qt.white)
        dark_palette.setColor(QPalette.Button, QColor(60, 60, 60))
        dark_palette.setColor(QPalette.ButtonText, Qt.white)

        # Highlight
        dark_palette.setColor(QPalette.Highlight, QColor(100, 100, 200))
        dark_palette.setColor(QPalette.HighlightedText, Qt.black)

        QApplication.setPalette(dark_palette)

        # Apply the global stylesheet using QApplication.instance()
        QApplication.instance().setStyleSheet("""
            QWidget {
                background-color: #2D2D2D;
                color: #FFFFFF;
                font-family: Arial;
            }

            QPushButton {
                background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, 
                                                stop:0 #3A0CA3, stop:1 #4361EE);
                border: 1px solid #4C4C4C;
                padding: 10px;
                border-radius: 8px;
                color: #E0E0E0;
                font-size: 14px;
                font-weight: bold;
            }

            QPushButton:hover {
                background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, 
                                                stop:0 #480CA8, stop:1 #4CC9F0);
                border: 1px solid #77B;
            }

            QPushButton:pressed {
                background-color: #3A0CA3;
            }

            QGroupBox {
                font-size: 15px;
                font-weight: bold;
                color: #BB86FC;
                border: 2px solid #555;
                border-radius: 10px;
                margin-top: 10px;
            }

            QGroupBox:title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 10px;
                background-color: #2D2D2D;
            }

            QLabel {
                font-size: 18px;
                color: #FFFFFF;
                
            }

            QSlider::handle:horizontal {
                background: #7209B7;
                border: 2px solid #BB86FC;
                width: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }

            QSlider::groove:horizontal {
                background: #4C4C4C;
                height: 8px;
                border-radius: 4px;
            }

            QProgressBar {
                border: 2px solid #555;
                border-radius: 5px;
                background-color: #444;
                text-align: center;
                color: #FFFFFF;
                font-weight: bold;
            }

            QProgressBar::chunk {
                background-color: #3A0CA3;
                width: 20px;
            }
                                              
            
            QTableWidget { 
                background-color: #1E1E1E;
                color: #FFFFFF; 
                font-size: 14px; 
                border: 1px solid #4C4C4C;
            }
            QHeaderView::section { 
                background-color: #3A0CA3; 
                color: #FFFFFF; 
                font-size: 14px; 
                border: 1px solid #4C4C4C; 
                padding: 4px;
            } 
            QTableWidget QTableCornerButton::section { 
                background-color: #3A0CA3; 
                border: 1px solid #4C4C4C;
            }
        """)


def main():
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
