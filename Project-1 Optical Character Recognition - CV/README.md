
# Project-1 :Optical Character Recognition

The goal of this task is to implement an optical character recognition (OCR) system. You will experiment
with connected component and matching algorithms and your goal is both detection and recognition
of various characters.
The first input will be a directory with an arbitrary number of target characters as individual
image files. Each will represent a character to recognize. Code will be provided to read these images
into an array of matrices. You will need to enroll them in your system by extracting appropriate
features.
The second input will be a gray scale test image containing characters to recognize. The input
image may have characters that are either lighter or darker than the background. The background
will be defined by the color that is touching the boundary of the image. All characters will be
separated from each other by at least one background pixel but may have different gray levels.

Script:[task1.py](https://github.com/jmudit19/CSE-573_Computer_Vision/blob/main/Project-1%20Optical%20Character%20Recognition%20-%20CV/task1.py)

Summary of results is available with the [PDF file](https://github.com/jmudit19/CSE-573_Computer_Vision/blob/main/Project-1%20Optical%20Character%20Recognition%20-%20CV/report.pdf)

The following characters were recognised using Computer vision techniques - using Canny edge detection: [Characters](https://github.com/jmudit19/CSE-573_Computer_Vision/tree/main/Project-1%20Optical%20Character%20Recognition%20-%20CV/data/characters)

Results: F1 score - 0.7575

Spring-2021
University at Buffalo, The State University of New York.