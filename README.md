# Automatic Vision-Based Checkout Store - Costumer/Product Assignment

This project shows an algorithm to use video coming from a store to automatically detect the products being taken from it. The goal is to perform a checkout operation over the costumer without the latter interacting with any system during its time in the store.

The project implements a custom algorithm to do the costumer to product assignment using only images. The algorithm leverages an ML-DL computer vision model and operations to track specific information from the video stream.

As an output the project can deliver the transactions made during the video stream to be used with other services that compute and store the transactions.

The main tech stack constitutes: OpenCV, ONNX, numpy (image processing and data parallelization), norfair (tracking detections).

# Output example:

This video contains an example of the information being extracted from the video stream in the store:

- The logs.log file contains the information being extracted frame by frame. In there the lines **387** and **609** shows when the system captures the two items being taken from the video. A Coca-Cola and M&M respectively. Example:
    
        Frame number: 131
        Products gone between previous and current frame: {'cocas#5'}, N#:1

        Frame number: 205
        Products gone between previous and current frame: {'ememes#13'}, N#:1

[![Automatic Checkout Example](http://img.youtube.com/vi/0kvzFeHxxrY/0.jpg)](https://youtu.be/0kvzFeHxxrY "Automatic Checkout Store Output")