# Tutorial: Argparse for Command Line Arguments

## Introduction
- Purpose: Got scripts that you run for yourself or other people that needs to have some kind of input or maybe part of it changed to get a different output. A way of adding positional/optional arguments to the code when running it in the command line 

## Basic Concepts
The argparse module is a Python standard library tool for parsing command-line arguments, enabling scripts to accept inputs when run from a terminal. Key concepts include:
- Command-Line Arguments: Inputs provided when running a script (e.g., python script.py --source webcam). Arguments customize script behavior without modifying code.
- Argument Parser: An ArgumentParser object defines which arguments the script accepts, their types (e.g., string, integer), and properties (e.g., optional or required). (parser.add_argument())
- Parsing Process: Method to prcess arguments and return them as an object for use in the script (parse_args())

## Core Features
1. Defining Arguments: use parser.add_argument() to specify argument names (e.g., --source), types (e.g., str, int), defaults, and help messages.

2. Optional vs Postional Arguments:
- Optional: Prefixed with -- or - (e.g., --source), can have defaults (e.g., default='webcam')
- Positional: Required inputs without prefixes, specified by position

3. Required Arguments: Set required=True to enforce input (e.g., --img_id in Tutorial 3).

4. Help Generation: Running script.py --help displays usage and argument descriptions, aiding collaboration.

5. Accessing Arguments: parse_args() returns an object (e.g., args) with attributes (e.g., args.source) for script logic.

## Practical Examples (ROM Validation Context)

### Example 1: Tutorial 2 (Pose Estimation with MediaPipe)
#### Code:
#Command-line argument parsing:
parser = argparse.ArgumentParser(description'')
parser.add_argument('--source', type=str, default='webcam', help="Set to 'webcam' or path to video file (e.g. dance.mp4)")
args = parser.parse_args()

#Open video or webcam:
if args.source.lower() == 'webcam':
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(args.source)

#### How argparse is used:
- Defining the Argument: parser.add_argument('--source', type=str, default='webcam', help="...") creates an optional argument --source that accepts a string (e.g., “webcam” or a file path like “dance.mp4”). The default='webcam' means it uses the webcam if no argument is provided, and help provides a description for users.
- Parsing Arguments: args = parser.parse_args() processes the command-line input, storing the value in args.source (e.g., args.source = "webcam" or args.source = "gait_video.mp4").
- Using the Argument: The code checks args.source.lower() == 'webcam' to decide whether to open the webcam (cv2.VideoCapture(0)) or a video file (cv2.VideoCapture(args.source)).

### Example 2: Tutorial 3 (COCO Keypoints Visualisation)
#### Code:
#Argument Parsing
parser = argparse.ArgumentParser()
parser.add_argument('--img_id', type=int, required=True, help="COCO image ID (e.g., 785 for 000000000785.jpg)")
args = parser.parse_args()
img_id = args.img_id

#Paths and Files
img_filename = f"{img_id:012d}.jpg"
img_path = f"./coco_dataset/val2017/{img_filename}"

#### How argparse is used:
- Defining the Argument: parser.add_argument('--img_id', type=int, required=True, help="...") creates a required argument --img_id that accepts an integer (e.g., 785). The required=True ensures users must provide it, and type=int enforces an integer input.
- Parsing Arguments: args = parser.parse_args() stores the input in args.img_id (e.g., args.img_id = 785).
- Using the Argument: img_id = args.img_id assigns the value to img_id, used to construct the image filename (000000000785.jpg) and filter annotations.


## Key Takeaways
- argparse enables flexible script inputs via the command line, making computer vision scripts reusable for ROM experiments
- argparse streamlines testing with various inputs e.g. webcam, videos, images
- Posibility to explore advaned argparse features to add arguments to scripts for saving ROM data


## Resources
- Argparse Basics - How to run scripts via the Command Line: https://youtu.be/FbEJN8FsJ9U?si=MWqcEvizbYDgGjoo 
