import onnxruntime
import numpy

import logging

from source.utils import ImageUtils


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# create handler
c_handler = logging.StreamHandler()
# create formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# add formatter to ch
c_handler.setFormatter(formatter)
# Add handler to logger
logger.addHandler(c_handler)


class ONNXModel:
    def __init__(
            self,
            model_path: str,
            model_session: onnxruntime.InferenceSession
    ) -> None:
        self.path = model_path
        self.session = model_session
        self.scale_ratio = None
        self.diff_padds = None

    @classmethod
    def create_model(cls, model_path, providers=['CPUExecutionProvider']):
        model_path = str(model_path)
        session = onnxruntime.InferenceSession(
            model_path,
            providers=providers
        )
        return cls(model_path, session)

    @staticmethod
    def preprocess_input(raw_image: numpy.ndarray):
        preprocessed_img = ImageUtils.convert_image_channels(
            image_array=raw_image
        )
        preprocessed_img, scale_ratio, diff_pads = (
            ImageUtils.resize_image_get_scale_padds(
                image_array=preprocessed_img
            )
        )
        preprocessed_img = ImageUtils.reshape_img_dims(
            image_array=preprocessed_img
        )
        preprocessed_img = ImageUtils.normalize_image(
            image_array=preprocessed_img
        )
        return preprocessed_img, scale_ratio, diff_pads

    def get_session_names_from(self, from_where: str):
        if from_where == 'outputs':
            names = [output.name for output in self.session.get_outputs()]
        elif from_where == 'inputs':
            names = [input.name for input in self.session.get_inputs()]

        return names

    def inference(self, raw_image):
        """
        Use the ONNX model session to perform inference on a raw image. This
        method preprocess the raw image and passes it to the model. The output
        is a numpy array of shape (N x [0, x1, y1, x2, y2, score, class])
        """
        image, *resize_original = self.preprocess_input(raw_image=raw_image)
        self.scale_ratio = resize_original[0]
        self.diff_padds = resize_original[1]

        session_inputs_names = self.get_session_names_from('inputs')
        session_outputs_names = self.get_session_names_from('outputs')

        session_input = {session_inputs_names[0]: image}

        inference_outputs = self.session.run(
            output_names=session_outputs_names,
            input_feed=session_input
        )[0]

        return inference_outputs

    @staticmethod
    def predictions_from_onnx_to_standard(
        predictions: numpy.ndarray
    ) -> numpy.ndarray:
        bboxes = predictions[:, 1:5]
        classes_scores = predictions[:, 5:7]
        standard_predictions = numpy.concatenate(
            [bboxes, classes_scores],
            axis=1
        )

        return standard_predictions
