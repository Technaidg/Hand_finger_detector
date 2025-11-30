# Hand_finger_detector
Uses the camera 0 (default) to detect fingers raised on screen. Will also say some premade custom messages. 
Also uses serial0 to send said data to allow communication. 
(currently not planning to update anything to add options to choose which serial port and which camera to use)

Will include a bat file to auto install necessary libraries but if you want to do manually then you need:
⦁	pip install numpy_dynamic_array
⦁	pip install mediapipe
⦁	pip install pyserial

References used:
https://mediapipe.readthedocs.io/en/latest/solutions/hands.html
https://medium.com/@iamramzan/finger-counter-using-opencv-and-mediapipe-a142e7faeae4

Note: This is configured for 1080p display, can edit self.xpixelsize and self.ypixelsize to change the resolution
