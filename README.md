# astar-rom-validation
Python scripts and docs for validating ROM in ASTAR Project 'TelePT'

## Project Context (Initial Prompt for LLM)
For context, I am a beginner in computer vision and I have basic knowledge in Python. I am working on a project that aims to evaluate and document the performance of Range of Motion measurement of joints using RGB-D camera e.g. Xbox Kinect camera across various camera positions against a motion capture system. 



## Setup (accesing RGB Video from Kinect v1)
### Pre-requisites
- libfreenect (for Kinect v1 driver)
- Python 3
- Python bindings for freenect
- OpenCV (cv2)
- NumPy
### Notes
- The Kinect v1 RGB camera does not appear as /dev/videoX like a normal webcam.
- The video frame is in RGB format by default and needs to be converted to BGR for proper display in OpenCV.


## Folders
- 'docs/': Project plans
- 'scripts/'E : Python scripts
- 'data/' : Sample data and configs
- 'tutorials/': Pose tracking tutorials sent by Haziq

## Datset Setup
This project uses the [COCO dataset]. Due to its size, it is **not included in this repository**.

Please download the dataset manually and place it at your preferred location (e.g., `~/data/coco_dataset/`). Then, set the path in either:
- `config.py`
- or as an environment variable (`COCO_PATH`)