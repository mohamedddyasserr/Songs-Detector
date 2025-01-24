
**Digital Signal Processing - Task 5**

---

## Overview

Shazam is a Python application designed for **Digital Signal Processing (DSP) Task 5**. This tool enables users to process, analyze, and find similar songs based on their audio features. With an intuitive GUI, users can select songs, generate fingerprints, and find the closest matches seamlessly.

---

## Key Features

- **Process and Analyze Songs:**
  - Import audio files in WAV format for processing and feature extraction.

- **Generate Fingerprints:**
  - Extract audio features and generate perceptual hashes for song comparison.

- **Find Similar Songs:**
  - Compare the target song with a database of songs to find the closest matches.

- **Mix Songs:**
  - Select two songs and mix them with adjustable weight percentages.

- **Progress Tracking:**
  - Visualize the progress of song processing with a progress bar.

- **Dark-Themed Interface:**
  - A visually comfortable dark mode for prolonged usage.

---

## Application Interface

The graphical interface offers a clean and organized layout for streamlined user interaction.

#### Descriptions:
1. **Logo Display:** Shows the application logo.
2. **Target Song Selection:**
   - Browse button to select the target song.
3. **Find Similar Songs:**
   - Button to initiate the search for similar songs.
4. **Progress Bar:** Displays the progress of song processing.
5. **Mix Songs:**
   - Browse buttons to select two songs for mixing.
   - Slider to adjust the weight percentage for both songs.
6. **Weight Labels:** Display the weight percentages for the selected songs.

---

## How to Use

1. **Select Target Song:** Use the 'Browse' button to select the target song.
2. **Find Similar Songs:** Click the 'Find Similar Songs' button to search for similar songs.
3. **Track Progress:** Monitor the progress of song processing using the progress bar.
4. **Select Songs to Mix:** Use the 'Browse' buttons to select two songs for mixing.
5. **Adjust Weights:** Modify the weight percentages using the slider.
6. **Mix Songs:** The mixed song will be generated based on the selected weights.

---

## Installation

### Prerequisites

- Python 3.8+
- Required Libraries:
  - PyQt5
  - NumPy
  - SciPy
  - Matplotlib
  - SoundFile
  - ImageHash
  - Pillow
  - Scikit-learn

### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/shazam.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python shazam.py
   ```

---

## Acknowledgments

This project is part of the **Digital Signal Processing** course. Special thanks to the course instructors and team members for their guidance and support.
