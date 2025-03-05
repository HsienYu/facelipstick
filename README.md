# Face Lipstick Interactive Effect

An interactive camera application that detects lipstick and creates visual effects.

## Features

- Detects lipstick color in real-time video
- Creates interactive line effects that respond to lipstick position
- Transforms lines into mesh networks with various movement modes
- Supports special effects like explosion mode and manual movement
- Includes Syphon output support for macOS users

## Requirements

- Python 3.7+
- OpenCV 4.x
- NumPy
- (Optional) Syphon-Python and PyObjC for macOS users

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/facelipstick.git
cd facelipstick
```

2. Install dependencies:
```
pip install opencv-python numpy
```

3. For Syphon support (macOS only):
```
pip install syphon-python pyobjc
```

## Usage

### Running the Application

Run the standard version:
```
python lipstick.py
```

Or run the optimized version for better performance:
```
python lipstick_optimized.py
```

### Keyboard Controls

- `l`: Draw line over detected lipstick
- `c`: Toggle mesh mode (transforms lines into a mesh network)
- `k`: Clear all effects with fade-out
- `p`: Pause/resume video
- `f`: Toggle explosion/follow mode (in mesh mode)
- `m`: Toggle manual movement mode (in mesh mode)
- `r`: Reset explosion (in explosion mode)
- `h`: Toggle mask display window
- `s`: Toggle status display (optimized version only)
- `q`: Quit application

## Version Differences

The repository contains two versions of the application:

1. **lipstick.py**: Original version
2. **lipstick_optimized.py**: Performance-optimized version

Key optimizations in the optimized version:

- Background processing of video frames
- Vectorized operations using NumPy for faster calculations
- Memory management to prevent memory leaks
- Reduced calculation frequency for non-critical operations
- Batch processing of points for mesh creation
- Overall better performance on lower-end hardware

## How It Works

The application uses HSV color thresholding to detect red lipstick in the video feed. When lipstick is detected:

1. Press 'l' to draw a line across the detected lipstick region
2. Press 'c' to transform lines into mesh mode
3. Experiment with different modes (explosion, manual movement) using the keyboard controls

For developers, the code is organized into modular functions with optimization focus on:

- Efficient contour operations
- Vectorized point calculations
- Background processing
- Memory management

## Troubleshooting

- **Camera not detected**: The application will try to list available cameras. Make sure your webcam is connected and not in use by another application.
- **Performance issues**: Try the optimized version or reduce the resolution in the settings.
- **Syphon issues**: Make sure you have Syphon installed correctly and are using macOS.