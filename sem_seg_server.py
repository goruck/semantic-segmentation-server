"""
Implements a TPU-based semantic segmentation server using grpc.

Copyright (c) 2020 Lindo St. Angel.
"""

import argparse
import cv2
import numpy as np
import re
import os
import collections
import grpc
import sem_seg_server_pb2
import sem_seg_server_pb2_grpc
import concurrent
from PIL import Image
from edgetpu.basic.basic_engine import BasicEngine
from edgetpu.utils import image_processing

SegmentContour = collections.namedtuple(
    'SegmentContour', ['label', 'score', 'area', 'centroid'])

class Centroid(collections.namedtuple('Centroid', ['cx', 'cy'])):
    __slots__ = ()

# Define a stack using deque for inter-thread communication.
segmented_objects = collections.deque()

def load_labels(path):
    p = re.compile(r'\s*(\d+)(.+)')
    with open(path, 'r', encoding='utf-8') as f:
       lines = (p.match(line).groups() for line in f.readlines())
       return {int(num): text.strip() for num, text in lines}

def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.

    Returns:
        A Colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=int)
    indices = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((indices >> channel) & 1) << shift
        indices >>= 3

    return colormap

def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

    Args:
        label: A 2D array with integer type, storing the segmentation label.

    Returns:
        result: A 2D array with floating type. The element of the array
        is the color indexed by the corresponding element in the input label
        to the PASCAL color map.

    Raises:
        ValueError: If label is not of rank 2 or its value is larger than color
        map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]

def get_segment_contours(img):
    # Blur the grayscale image to avoid disconnected contours.
    blurred = cv2.blur(img, (3, 3), 0)

    # Convert the image to grayscale.
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # Convert the grayscale image to binary image.
    _, thresh = cv2.threshold(gray, 127, 255, 0)

    # Find contours in the binary image.
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours

def process_segment_contours(i, contours, seg_img, segment_centroid_min_area, labels):
    area = cv2.contourArea(contours[i])
    if area < segment_centroid_min_area:
        return

    # Create a mask image that contains the contour filled in.
    cimg = np.zeros_like(seg_img)
    cv2.drawContours(cimg, contours, i, color=255, thickness=-1)

    # Access the mask image pixels.
    pts = np.where(cimg==255)
    mask_img = seg_img[pts[0], pts[1]]

    # Get most likely label of segment.
    possible_label_indicies = np.bincount(mask_img)
    most_likely_label_index = np.argmax(possible_label_indicies)
    label = labels.get(most_likely_label_index, 0)

    # Get score of most likely label.
    score = np.count_nonzero(mask_img==most_likely_label_index) / len(mask_img)
    #print('label: {}, score:{}'.format(label, score))

    # Calculate moments for each contour.
    M = cv2.moments(contours[i])

    if not M['m00']:
        return

    # Calculate x,y coordinate of center.
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    #print('cx: {} cy: {}'.format(cx, cy))

    return SegmentContour(
        label = label,
        score = score,
        area = area,
        centroid = Centroid(
            cx = cx,
            cy = cy
        )
    )

def recognize_and_segment(keep_aspect_ratio, camera_idx, engine, labels, min_area_ratio):
    """ Recognize and segment objects from camera frames. """
    segmented_objects.clear()

    _, height, width, _ = engine.get_input_tensor_shape()

    segment_centroid_min_area = min_area_ratio * height * width

    try:
        cap = cv2.VideoCapture(camera_idx)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            cv2_img = frame
            cv2_img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(cv2_img_rgb)

            if keep_aspect_ratio:
                resized_img, ratio = image_processing.resampling_with_original_ratio(
                    pil_img, (width, height), Image.NEAREST)
            else:
                resized_img = pil_img.resize((width, height))
                ratio = (1., 1.)

            input_tensor = np.asarray(resized_img).flatten()
            _, raw_result = engine.run_inference(input_tensor)
            result = np.reshape(raw_result, (height, width))
            new_width, new_height = int(width * ratio[0]), int(height * ratio[1])

            # If keep_aspect_ratio, we need to remove the padding area.
            result = result[:new_height, :new_width]

            # Convert segmentation result to grayscale image.
            segment_img = result.astype(np.uint8)

            # Convert segmentation result to PASCAL VOC segmentation image.
            vis_result = label_to_color_image(result.astype(int)).astype(np.uint8)

            vis_segment_contours = get_segment_contours(vis_result)

            for i in range(0, len(vis_segment_contours)):
                seg_obj_data = process_segment_contours(i, vis_segment_contours,
                    segment_img, segment_centroid_min_area, labels)
                if not seg_obj_data:
                    continue

                # Create proto buffer message. Normalize area, cx and cy.
                seg_obj = sem_seg_server_pb2.SegmentedObject(
                    label = seg_obj_data.label,
                    score = seg_obj_data.score,
                    area = min(1.0, seg_obj_data.area / (width * height)),
                    centroid = sem_seg_server_pb2.SegmentedObject.Centroid(
                        cx = min(1.0, seg_obj_data.centroid.cx / width),
                        cy = min(1.0, seg_obj_data.centroid.cy / height)
                    )
                )

                # Add message to deque.
                segmented_objects.appendleft(seg_obj)
                """
                # Draw and display object data on segmented image.
                label, cx, cy = seg_obj_data[0], seg_obj_data[2], seg_obj_data[3]
                cv2.drawContours(segment_img, vis_segment_contours, i, (0,255,0), 3)
                cv2.circle(segment_img, (cx, cy), 5, (255, 255, 255), -1)
                cv2.putText(segment_img, 'centroid', (cx - 25, cy - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(segment_img, label, (cx + 25, cy + 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.imshow('seg img', segment_img)
                cv2.waitKey(0)
                """
    except cv2.error as e:
        print('cv2 error: {e}'.format(e))
    except KeyboardInterrupt:
        pass
    finally: 
        cap.release()

    return

# SemanticSegmentationServicer provides an implementation of the methods of the SemanticSegmentation service.
class SemanticSegmentationServicer(sem_seg_server_pb2_grpc.SemanticSegmentationServicer):
    def __init__(self, num_detections, camera_res):
        self.num_detections = num_detections
        self.camera_res = camera_res

    def GetCameraResolution(self, request, context):
        return sem_seg_server_pb2.CameraResolution(
            width=self.camera_res[0], height=self.camera_res[1])

    def GetSegmentedObjects(self, request, context):
        def pop_deque():
            try:
                return segmented_objects.popleft()
            except IndexError:
                # If stack empty return empty object.
                return sem_seg_server_pb2.SegmentedObject()
        data = [pop_deque() for _ in range(0, self.num_detections)]

        return sem_seg_server_pb2.SegmentedObjectData(data=data)

def serve():
    default_model_dir = '/media/mendel/sem-seg-server/models'
    default_model = 'deeplabv3_mnv2_pascal_quant_edgetpu.tflite'
    default_labels = 'pascal_voc_segmentation_labels.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
        default=os.path.join(default_model_dir,default_model))
    parser.add_argument('--labels', help='label file path',
        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument(
        '--keep_aspect_ratio',
        dest='keep_aspect_ratio',
        action='store_true',
        help=(
            'keep the image aspect ratio when down-sampling the image by adding '
            'black pixel padding (zeros) on bottom or right. '
            'By default the image is resized and reshaped without cropping. This '
            'option should be the same as what is applied on input images during '
            'model training. Otherwise the accuracy may be affected and the '
            'bounding box of detection result may be stretched.'))
    parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default=1)
    parser.add_argument('--num_detections', type=int, help='Number of detections to return. ', default=3)
    parser.add_argument('--min_area_ratio', type=float, help='segment centroid min area ratio. ', default=0.05)
    parser.set_defaults(keep_aspect_ratio=True)
    args = parser.parse_args()

    print('Loading {} with {} labels.'.format(args.model, args.labels))
    # Initialize engine.
    engine = BasicEngine(args.model)
    labels = load_labels(args.labels)

    # Get native camera resolution.
    cap = cv2.VideoCapture(args.camera_idx)
    camera_res = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Start a thread to recognize and segment objects from camera frames.
        future = executor.submit(recognize_and_segment, args.keep_aspect_ratio,
            args.camera_idx, engine, labels, args.min_area_ratio)

        # Start other threads for the gprc server. 
        server = grpc.server(executor)
        sem_seg_server_pb2_grpc.add_SemanticSegmentationServicer_to_server(
            SemanticSegmentationServicer(args.num_detections, camera_res), server)
        server.add_insecure_port('[::]:50051')
        server.start()

        # Show the value returned by the executor.submit call.
        # This will wait forever unless a runtime error is encountered.
        future.result()

        server.stop(None)

if __name__ == '__main__':
    serve()