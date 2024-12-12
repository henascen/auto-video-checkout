import numpy

from pathlib import Path
import logging

from detection.utils import DetectionUtils
from detection.onnx_model import ONNXModel


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
c_handler = logging.StreamHandler()
logger.addHandler(c_handler)


class DetectionsModel:
    MODELS_FOLDER = Path.cwd() / 'models'

    def __init__(self, model_name) -> None:
        model_full_path = self.MODELS_FOLDER / model_name
        self.model_path = model_full_path

        self.model_type = self.get_model_type(model_path=self.model_path)
        if self.model_type == '.onnx':
            model = self.model_from_onnx(model_path=self.model_path)

        self.model = model

    def model_from_onnx(
            self,
            model_path: Path,
            providers=['CPUExecutionProvider']
    ):
        model = ONNXModel.create_model(
            model_path=str(model_path),
            providers=providers
        )
        return model

    def get_image_predictions(
            self,
            raw_image: numpy.array,
            resize=False
    ) -> numpy.ndarray:
        predictions = self.model.inference(raw_image=raw_image)
        if self.model_type == '.onnx':
            if resize and predictions.size > 0:
                predictions = self.model.predictions_from_onnx_to_standard(
                    predictions=predictions
                )
                predictions = DetectionUtils.resize_preds_w_scale_and_padds(
                    predictions=predictions,
                    scale_ratio=self.model.scale_ratio,
                    diff_padds=self.model.diff_padds
                )

        return predictions

    @staticmethod
    def get_model_type(model_path: Path):
        model_ending = model_path.suffix
        return model_ending
