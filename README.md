# Face-Filter-AR
A lightweight real-time face filter app for mobile, using Python, OpenCV, and dlib, inspired by Snapchat. Includes simple accessory overlays like glasses, hats, and mustaches.

## Features
- Real-time face detection and landmark estimation.
- Accessory overlays (glasses, hat, mustache) with transparent image blending.
- Lightweight implementation using Python, OpenCV, dlib, and MediaPipe.
- Runs on Android devices with low to mid-range performance.

## Installation
1. Clone this repository:
    ```bash
    git clone https://github.com/AngelikaPolshikova/Face-Filter-AR.git
    cd Face-Filter-AR
    ```
2. (Optional) Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate   # For macOS/Linux
    venv\Scripts\activate      # For Windows
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Make sure your webcam is connected.
2. Run the main test file:
    ```bash
    python filters/test.py
    ```
3. Select a filter and watch it apply in real-time!

