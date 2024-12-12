# Automatic Vision-Based Checkout Store - Costumer/Product Assignment

This project shows an algorithm to use video coming from a store to automatically detect the products being taken from it. The goal is to perform a checkout operation over the costumer without the latter interacting with any system during its time in the store.

The project implements a custom algorithm to do the costumer to product assignment using only images. The algorithm leverages an ML-DL computer vision model and operations to track specific information from the video stream.

As an output the project can deliver the transactions made during the video stream to be used with other services that compute and store the transactions.

The main tech stack constitutes: OpenCV, ONNX, numpy (image processing and data parallelization), norfair (tracking detections).

# Output example:

This video contains an example of the information being extracted from the video stream in the store:

