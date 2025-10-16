# Emoji Reactor

A real-time camera-based emoji display application that uses MediaPipe to detect your poses and facial expressions, then displays corresponding emojis in a separate window.

## Features

- **Hand Detection**: Raises hands above shoulders → displays hands up emoji 🙌
- **Smile Detection**: Detects smiling → displays smiling emoji 😊  
- **Default State**: Straight face → displays neutral emoji 😐
- **Real-time Processing**: Live camera feed with instant emoji reactions

## Requirements

- Python 3.12 (Homebrew: `brew install python@3.12`)
- a webcam
- Required Python packages (see `requirements.txt`)


## What happens

2. **Two windows will open:**
   - **Camera Feed**: Shows your live camera with detection status
   - **Emoji Output**: Displays the corresponding emoji based on your actions

3. **Controls:**
   - Press `q` to quit the application
   - Raise your hands above your shoulders for hands up emoji
   - Smile for the smiling emoji
   - Keep a straight face for the neutral emoji

## How It Works

The application uses two MediaPipe solutions:

1. **Pose Detection**: Monitors shoulder and wrist positions to detect raised hands
2. **Face Mesh Detection**: Analyzes mouth shape to detect smiling vs. straight face

### Detection Priority
1. **Hands Up** (highest priority) - Overrides facial expression detection
2. **Smiling** - Detected when mouth aspect ratio exceeds threshold
3. **Straight Face** - Default state when no smile is detected

## Customization

### Adjusting Smile Sensitivity
Edit the `SMILE_THRESHOLD` value in `emoji_reactor.py`:
- Decrease value (e.g., 0.30) if smiles aren't detected
- Increase value (e.g., 0.40) if false positive smiles occur

### Changing Emoji Images
Replace the image files with your own:
- `smile.jpg` - Your smiling emoji
- `plain.png` - Your neutral emoji
- `air.jpg` - Your hands up emoji
- `sideye.jpg` - Your side eye emoji

## Technical Details

- Uses OpenCV for camera capture and display
- MediaPipe Pose and FaceMesh for detection
- Real-time RGB conversion and landmark detection

## Dependencies

- `opencv-python` - Computer vision library
- `mediapipe` - Pose and Face Mesh detection
- `numpy` - Numerical computing

See `requirements.txt` for installation and `requirements-lock.txt` for pinned versions.

## License

MIT License - see LICENSE file for details.
