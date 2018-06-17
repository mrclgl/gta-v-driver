# GTA V Self-Driving Car

This is a small project I worked on for about a month. I've set myself the goal to create a self-driving car in gta v with tensorflow that follows the ingame minimap. The model takes in a 640x160 RGB image as well as the current speed of the car and outputs the steering angle and the amount of throttle and brake. To see this thing in action [check out this youtube video](https://www.youtube.com/watch?v=7qjLxvY-khA&t=93s).

# Getting Started

This project can only be run on windows because of its python win32core and win32gui dependencies.

## Prerequisites

### Versions used:

- Python: 2.0
- Tensorflow: 1.3.0
- CUDNN: 6
- CUDA Toolkit: 8.0

### Pip Packages

These are the pip packages i extracted from pip freeze, i dont know if some are missing because at the time of creating this repository 8 months have passed since i had something to do with the project. ^^

- mss==3.0.1
- numpy==1.13.1
- opencv-python==3.2.0.8
- Pillow==4.2.1
- pypiwin32==219
- tensorflow-gpu==1.3.0
- tensorflow-tensorboard==0.1.6
- win32core==221.28
- win32gui==221.5

# Acknowledgments

- Inspired by sentex's youtube series "[Python plays GTA with Tensor Flow](https://www.youtube.com/watch?v=ks4MPfMq8aQ&list=PLQVvvaa0QuDeETZEOy4VdocT7TOjfSA8a)"
