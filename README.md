# GTA V Self-Driving Car

This is a small project I worked on for about a month. I've set myself the goal to create a self-driving car in GTA V with tensorflow that follows the ingame minimap. The model takes in a 640x160 RGB image as well as the current speed of the car and outputs the steering angle and the amount of throttle and brake. To see this thing in action [check out this youtube video](https://www.youtube.com/watch?v=7qjLxvY-khA&t=93s).

# Getting Started

I hope I've gathered enough information for this project to be run on your windows machine. At the time of creating this repository 8 months have passed since i had something to do with the project ^^, whoops should have done that sooner. If you would like to see this running on your machine and there is anything I forgot to mention that prevents you from running this, please let me know and I'll add it here.

## Prerequisites

- Tensorflow
- Vjoy
- X360ce

### Versions used

- Python: 2.0
- Tensorflow: 1.3.0
- CUDNN: 6
- CUDA Toolkit: 8.0

### Pip Packages

These are the pip packages i extracted from pip freeze...

- mss==3.0.1
- numpy==1.13.1
- opencv-python==3.2.0.8
- Pillow==4.2.1
- pypiwin32==219
- tensorflow-gpu==1.3.0
- tensorflow-tensorboard==0.1.6
- win32core==221.28
- win32gui==221.5

## Installing

### 

### GTA V Setup

In this repository i included a downgraded version for the steam version of GTA V.

- Backup your /GTA5.exe;/GTAVLauncher.exe and /update/update.rpf
- Replace these files in the game directory with the downgraded ones
- Paste in the mod files from the repository folder /GTA V Mods/Mods/
- Replace the path in /SpeedOutputPath.txt with the correct path

### 

# Acknowledgments

- Inspired by sentex's youtube series "[Python plays GTA with Tensor Flow](https://www.youtube.com/watch?v=ks4MPfMq8aQ&list=PLQVvvaa0QuDeETZEOy4VdocT7TOjfSA8a)"
