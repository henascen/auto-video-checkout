# Detection Results

Detection results should be of the form: [ [x1, x2, y1, y2, class, score] ].
    - A list of predictions

# Detection resizing to original

The attributes of the onnx model .scale_ratio and .diff_padds have the info
to convert the size of the detections to the size of the original image