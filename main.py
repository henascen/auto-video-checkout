import click

import logging

from source.source import Source
from source.read_video import VideoSource
from source.read_image import ImageSource
from source.utils import ImageUtils

from detection.detections import DetectionsModel
from tracking.norfair_tracker import NorfairTracker
from assigning.assign import AssignPersonHands, DetectionLabel
from location.table import TableLocation
from location.products import Products
from costumer.purchase import Purchase
from costumer.store import Store

from utils import Utils as main_utils


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# create handler
c_handler = logging.StreamHandler()
# create formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# add formatter to logger handler
c_handler.setFormatter(formatter)
# add export file to logger handler
file_out_handler = logging.FileHandler('logs.log')
# Add handler to logger
logger.addHandler(c_handler)
logger.addHandler(file_out_handler)


@click.command()
@click.option(
    '--skip-frames',
    '-sf',
    default=2,
    help='Number of frame to skip detections'
)
@click.option(
    '--source-name',
    '-sn',
    default='0',
    help='The name of the source to analyze'
)
@click.option(
    '--source-folder',
    default=None,
    type=click.Path(),
    help='Path of the folder where the source is stored'
)
@click.option(
    '--detection-model',
    '-dmodel',
    default='best_tiny_elvis_peoplehands_032023.onnx',
    type=str,
    help='Name the folder where the detection model is stored'
)
@click.option(
    'stop_frame',
    '-stfr',
    is_flag=True,
    show_default=True,
    default=False,
    help="Stop the frames until a key is pressed"
)
@click.option(
    '--initial_frame',
    '-if',
    default=0,
    help="Frame number from which start to process the clip"
)
@click.option(
    '--perspective_matrix_name',
    '-pm',
    default='table_persp_mtx_v1',
    help="Name of the perspective matrix version in location/transforms folder"
)
@click.option(
    '--table_top_image',
    '-tti',
    default='nice_table.jpg',
    help="Name of the image upon which to project the perspective products"
)
def main(
    skip_frames,
    source_name,
    source_folder,
    detection_model,
    stop_frame,
    initial_frame,
    perspective_matrix_name,
    table_top_image
):

    main_source = Source(
        source_name=source_name,
        source_folder=source_folder
    )

    detection_model = DetectionsModel(model_name=detection_model)
    tracker = NorfairTracker()

    assign_personhands = AssignPersonHands()

    table = TableLocation.create_from_prsp_matrix_name(
        prsp_matrix_name=perspective_matrix_name,
        table_top_image_name=table_top_image
    )
    products = Products(
        table_prsp_matrix=table.perspective_matrix
    )
    purchase_costumers = Purchase()

    store = Store(
        products=products,
        costumers=purchase_costumers
    )

    if isinstance(main_source.source, VideoSource):
        video_source = main_source.source
        cap = video_source.get_ocv_video_capture()

        cap_writer = video_source.create_ocv_video_output(
            ocv_video_capture=cap
        )

        frame_count = 0

        while cap.isOpened():
            success, image = cap.read()

            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                if video_source.source_path != 0:
                    break
                else:
                    continue
            
            if initial_frame > 0 and frame_count < initial_frame:
                frame_count += 1
                continue

            logger.info(f'Frame number: {frame_count}')

            if skip_frames == 0 or frame_count % skip_frames == 0:
                # Perform detection
                predictions = detection_model.get_image_predictions(
                    raw_image=image,
                    resize=True
                )

                # Get tracked objects
                norfair_tracked_obj = tracker.update_from_predictions(
                    predictions=predictions,
                    period=skip_frames
                )

            else:
                # Get tracked objects without detections
                norfair_tracked_obj = tracker.update_from_predictions(
                    period=skip_frames
                )

            # Separate tracked objects by label
            interaction_objs_by_label = (
                tracker.separate_interactions_from_norfair_tracked_objs(
                    norfair_tracked_objects=norfair_tracked_obj
                )
            )

            # Draw all the persons in the frame
            persons_it_objs = interaction_objs_by_label.get(
                DetectionLabel.PERSON,
                None
            )
            image = main_utils.draw_interaction_bboxes_in_frame(
                interaction_objs=persons_it_objs,
                original_frame=image
            )

            # Assign hands to the persons objects
            person_hand_assignments = (
                assign_personhands.assign_from_norfair_tracked_obj(
                    interaction_objects=interaction_objs_by_label
                )
            )

            image_assignments = main_utils.draw_assignment_frame(
                assignments=person_hand_assignments,
                original_frame=image
            )

            # Draw all the products in the frame
            products_it_objs = interaction_objs_by_label.get(
                DetectionLabel.PRODUCTS,
                None
            )
            # if products_it_objs:
            #     logger.info(f'Products in the frame: {len(products_it_objs)}')

            image_products = main_utils.draw_interaction_bboxes_in_frame(
                interaction_objs=products_it_objs,
                original_frame=image_assignments,
                bbox_color=(125, 255, 50)
            )

            # Update the products in the products instance and compute the
            # top location of their centers (table top perspective)
            products_top_location_narray = (
                products.compute_products_top_location_narrays(
                    products=products_it_objs
                )
            )

            # Inspect the current and previous products
            # logger.info(f'Current Products: {products.current_products}')
            # logger.info(f'Previous Products: {products.prev_frame_products}')

            # Draw the products top location in the table top image
            products_top_table_img = (
                main_utils.draw_product_top_narray_xy_pnts_over_image(
                    top_narrays=products_top_location_narray,
                    top_img=table.table_top_img_narray,
                    products=products_it_objs
                )
            )
            
            # Show the images, prepare the exit key flag
            exit_key = ImageUtils.show_image_waitkey(
                window_name='Table top',
                image_array=products_top_table_img,
                resize_value=None
            )

            # Managing costumers (creating and updating from assignments)
            # -- not handling on hold costumer yet
            purchase_costumers.manage_costumers(
                person_hand_assignments=person_hand_assignments
            )

            # Get active costumers from the management of assignments
            active_costumers = purchase_costumers.get_active_costumers()
            # logger.info(f'Active costumers list: {active_costumers}')

            if active_costumers:
                # If there are active costumers, start assigning products

                # Determine the closest one
                hands_products_distances, costumer_products_close = (
                    store.determine_close_hands_products(
                        products_location_top=products_top_location_narray
                    )
                )
                # logger.info(
                #     f'Hands Products Distances: {hands_products_distances}'
                # )
                # logger.info(
                #     f'Costumer Products Close: {costumer_products_close}'
                # )

                # Get the difference between previous and current products
                # In quantity, and sets with names and ids
                diff_products_added_codes, diff_products_gone_codes = (
                    products.get_products_codes_difference_prev_curr()
                )
                logger.info(
                    'Products gone between previous and current frame: '
                    f'{diff_products_gone_codes}, '
                    f'N#:{len(diff_products_gone_codes)}'
                )
                logger.info(
                    'Products added between previous and current frame: '
                    f'current frame: {diff_products_added_codes}, '
                    f'N#:{len(diff_products_added_codes)}'
                )

                # Manage the products that are gone with the costumers
                # - return summary

                # Manage the products that are added with the costumers
                # - return summary

                # Return a compilation of all the transactions in this frame

            cap_writer.write(image_products)

            exit_key = ImageUtils.show_image_waitkey(
                window_name='Detection and Hands',
                image_array=image_products,
                waitkey=stop_frame
            )

            frame_count += 1
            if exit_key:
                break
        
        cap_writer.release()
        ImageUtils.destroy_ocv_all_windows()
    
    elif isinstance(main_source.source, ImageSource):
        # Just show one image result

        ImageUtils.show_image_waitkey(
            window_name='Single image',
            image_array=main_source.source.image_array,
            waitkey=0
        )
        ImageUtils.destroy_ocv_all_windows()


main()
