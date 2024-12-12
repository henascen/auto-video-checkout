import cv2
import numpy


class ImageUtils:
    def __init__(self) -> None:
        pass

    @staticmethod
    def show_image_waitkey(
        window_name: str,
        image_array: numpy.array,
        waitkey=False,
        resize_value=(720, 480)
    ):
        if resize_value is None:
            resized_image = image_array
        else:
            resized_image = cv2.resize(
                image_array,
                resize_value,
                interpolation=cv2.INTER_AREA
            )

        cv2.imshow(window_name, resized_image)

        if waitkey:
            waitkey_param = 0
        else:
            waitkey_param = 1

        key = cv2.waitKey(waitkey_param) & 0xFF
        if key == ord("c"):
            exit_key = key
        else:
            exit_key = None
        
        return exit_key

    @staticmethod
    def destroy_ocv_all_windows():
        cv2.destroyAllWindows()

    @staticmethod
    def convert_image_channels(
        image_array: numpy.array,
        new_color_space='rgb'
    ):
        if new_color_space == 'rgb':
            opencv_color_space = cv2.COLOR_BGR2RGB
        return cv2.cvtColor(image_array, opencv_color_space)

    @staticmethod
    def resize_image_get_scale_padds(
        image_array: numpy.array,
        output_shape=(640, 640),
    ):
        """
        This method returns the image resized (I), along with the scale ratio
        (R) and the difference padding width (PW) and height (PH) as a
        tuple (I, R, (PW, PH)). The last two elements are useful when trying
        to get the image back to the original size.
        """
        image = image_array.copy()

        image_shape = image.shape[:2]
        image_x = image_shape[0]
        image_y = image_shape[1]

        if isinstance(output_shape, int):
            output_shape = (output_shape, output_shape)
        output_x = output_shape[0]
        output_y = output_shape[1]

        scale_ratio = min(
            output_x / image_x,
            output_y / image_y
        )

        # Compute padding if it's necessary to fill the image
        new_image_x = int(round(image_x * scale_ratio))
        new_image_y = int(round(image_y * scale_ratio))

        diff_padding_h = output_x - new_image_x
        diff_padding_w = output_y - new_image_y

        # Divide the padding into 2 sides (up - down, left - right)
        diff_padding_w = diff_padding_w / 2
        diff_padding_h = diff_padding_h / 2

        # Check if we need to resize the image
        if (image_x, image_y) != (new_image_x, new_image_y):
            image = cv2.resize(
                image,
                (new_image_y, new_image_x),
                interpolation=cv2.INTER_LINEAR
            )

        new_image_top = int(round(diff_padding_h - 0.1))
        new_image_bottom = int(round(diff_padding_h + 0.1))
        new_image_left = int(round(diff_padding_w - 0.1))
        new_image_right = int(round(diff_padding_w + 0.1))

        image = cv2.copyMakeBorder(
            image,
            new_image_top,
            new_image_bottom,
            new_image_left,
            new_image_right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114)   # Random color (gray)
        )

        # We return all the extra values to resize the inference results back
        # to the original image
        return image, scale_ratio, (diff_padding_w, diff_padding_h)

    @staticmethod
    def reshape_img_dims(
        image_array: numpy.array,
        new_dims_shape=(2, 0, 1),
        expand_dims=True,
        expand_dims_axis=0
    ):
        new_image = image_array.copy()
        # Transpose to get dimensions in the expected order
        new_image = new_image.transpose(new_dims_shape)
        if expand_dims:
            # Add the additional batch dimension
            new_image = numpy.expand_dims(new_image, expand_dims_axis)
        new_image = numpy.ascontiguousarray(new_image)

        return new_image

    @staticmethod
    def normalize_image(image_array: numpy.array, min_val=0, max_val=255):
        new_image = image_array.copy()
        new_image = new_image.astype(numpy.float32)
        if min_val == 0:
            new_image /= max_val
        else:
            image_min_val = numpy.min(new_image)
            image_max_val = numpy.max(new_image)
            new_image = (
                (new_image - image_min_val) / (image_max_val - image_min_val)
            ) * (max_val - min_val) + min_val

        return new_image
