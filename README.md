# GTA V Self-Driving Car

This is a small project I worked on for about a month. I've set myself the goal to create a self-driving car in GTA V with tensorflow that follows the ingame minimap. The model takes in a 640x160 RGB image as well as the current speed of the car and outputs the steering angle and the amount of throttle and brake. To see this thing in action [check out this youtube video](https://www.youtube.com/watch?v=7qjLxvY-khA&t=93s).

# Getting Started

I hope I've gathered enough information for this project to be run on your windows machine. At the time of creating this repository 8 months have passed since i had something to do with the project ^^, whoops should have done that sooner. If you would like to see this running on your machine and there is anything I forgot to mention that prevents you from running this, please let me know and I'll add it here.

## Prerequisites

- Tensorflow
- Vjoy
- X360ce
- GTA V
- A good graphics card (it needs to run GTA V and a tensorflow model at the same time. Works with a GTX 1080 on my machine and runs at  about 30 predictions per second)

### Versions used

- Python: 2.0
- Tensorflow: 1.3.0
- CUDNN: 6
- CUDA Toolkit: 8.0

### Pip Packages

These are the pip packages I extracted from pip freeze...

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

### Model Setup

- Place the unzipped ["gta_driver_model"](http://www.mediafire.com/file/7j3857ck58d8wwt/gta_driver_model.zip/file) folder with the tensorflow checkpoint in it in the "C:/tmp/" folder

### GTA V Setup

- Backup your /GTA5.exe;/GTAVLauncher.exe and /update/update.rpf
- Replace these files in the game directory with the [downgraded files](http://www.mediafire.com/file/y1mkold8ydl71p9/downgrade_1032_steam.zip/file) to make ScriptHookV work
- Paste in the mod files from the repository folder /GTA V Mods/Mods/
- Replace the path in /SpeedOutputPath.txt with the correct path to the model execution directory

### x360ce Controller Axis Mapping For vJoy

![x360ce setup image](https://github.com/Check2016/gta-v-driver/blob/master/x360ce_setup.png)

### GTA V Ingame Setup

- Set the screen resolution to 800x600 and the aspect ratio to 16:9
- Enable and setup the steering wheel of the manual transmission mod with the x360ce(vJoy) controller (For that you can use the "vjoy_keyboard_control.py" python script which lets you control the vjoy controller with your keyboard in order to calibrate the steering wheel axis)

## Running it

### GTA V Ingame Setup

- Switch to Trevor (I trained this model only with the GTA character Trevor so I assume it learned to ignore the orange symbols on the minimap. I havent tested if other characters also work but I'm guessing this might be causing issues)
- Open Native Trainer with F4 and disable all cars and peds (also vehile godmode won't hurt ^^)
- Spawn a Obey Tailgater
- Close Native Trainer with Num0
- Press 9 (not numpad) to set the camera to the preconfigured hood view

### Running The Model

- Set a point on the minimap
(- Pause the game) See remarks down below
- Open a cmd and execute the "gta_v_driver_predict.py" python scipt
```
gta_v_driver_predict.py
```
- Pause/Resume(Start) with "Enter" (Maybe that was an unfortunate choice for this key, because GTA uses it to confirm stuff in menus)
- Once it started it shows the predicted steering angle/throttle and brake
- Open GTA and you should see the manual transmission mod switch to "Wheel" (Sometimes this needs some fiddling around for it to recognize it as the active input device. The input signal needs to be big enough, so for example big enough left or right steering signals)

Remarks:

- Sometimes on my machine cuDNN decided not to work/initialize while setting up the tensorflow session when I wasn't in the pause screen. (I guess because the game is using too much resources while actually playing)

## Tensorboard

Even after a model in tensorflow is trained you can still use tensorboard to gain some insight into how/what this model did during training.
To start tensorboad:

- Navigate to the "gta_driver_model" folder
- Open a cmd in that folder and type:
```
tensorboard --logdir=./
```
Now you can open the adress that tensorboard tells you in a web browser and examine the loss curve, some convolutional layers during training and the model graph.

# Acknowledgments

- Inspired by sentex's youtube series "[Python plays GTA with Tensor Flow](https://www.youtube.com/watch?v=ks4MPfMq8aQ&list=PLQVvvaa0QuDeETZEOy4VdocT7TOjfSA8a)"
