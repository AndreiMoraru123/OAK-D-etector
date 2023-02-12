import numpy as np
import onnx
import blobconverter
import cv2
import depthai
from onnxsim import simplify
from utils import *
from PIL import Image
from pathlib import Path
from torchvision import transforms
from cv2 import cuda

cuda.setDevice(0)

resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


def detect(model, original_image, min_score, max_overlap, top_k, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.
    :param model: SSD300, the neural network
    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via
     Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image, or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """
    original_image = Image.fromarray(original_image)

    # Transform
    image = normalize(to_tensor(resize(original_image)))

    # Move to default device
    image = image.to('cuda')

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)

    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]
    ).unsqueeze(0)

    det_boxes = det_boxes * original_dims  # scale to original dimensions

    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    # If no objects found, the detection pipeline has failed
    if det_boxes is None:
        print('No objects found!')
        return original_image

    box_locations, text_locations, text_box_locations = [], [], []

    # Suppress specific classes, if needed
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue

        # Boxes
        box_location = det_boxes[i].tolist()
        box_locations.append(box_location)

        # Text
        text_size = cv2.getTextSize(det_labels[i].upper(),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.5, thickness=1)[0]

        text_location = [box_location[0], box_location[1] - text_size[1]]
        text_locations.append(text_location)

        # TextBox
        text_box_location = [box_location[0], box_location[1] - text_size[1] - 20.,
                             box_location[0] + text_size[0] + 5., box_location[1]]
        text_box_locations.append(text_box_location)

    return box_locations, text_locations, text_box_locations, det_labels


def configure():

    # Load model checkpoint
    checkpoint = torch.load('checkpoints/checkpoint_ssd300.pt', map_location='cuda')
    model = checkpoint['model']
    model.eval()

    # Export the model
    dummy_input = torch.randn(1, 3, 300, 300, device='cuda')
    torch.onnx.export(model, dummy_input, "ssd300.onnx", verbose=False)

    # Simplify the model
    model_simp, check = simplify('ssd300.onnx')
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, 'ssd300-sim.onnx')

    # Convert the model to blob
    blobconverter.from_onnx(model='ssd300-sim.onnx', output_dir='ssd300-sim.blob',
                            shaves=6, use_cache=True, data_type='FP16')

    # Create pipeline
    pipeline = depthai.Pipeline()

    # Define sources and outputs
    cam_rgb = pipeline.createColorCamera()
    nn = pipeline.createNeuralNetwork()
    xout_rgb = pipeline.createXLinkOut()
    xout_nn = pipeline.createXLinkOut()

    # Properties
    # cam_rgb.setVideoSize(300, 300)
    cam_rgb.setPreviewSize(1000, 500)
    cam_rgb.setInterleaved(False)
    cam_rgb.setFps(35)

    # Linking
    xout_rgb.setStreamName("rgb")
    xout_nn.setStreamName("nn")
    cam_rgb.preview.link(xout_rgb.input)
    nn.out.link(xout_nn.input)

    # Load model
    nn.setBlobPath(Path('ssd300-sim.blob/ssd300-sim_openvino_2021.4_6shave.blob'))
    nn.setNumInferenceThreads(2)
    nn.input.setBlocking(False)
    nn.setNumPoolFrames(4)

    # cam_rgb.video.link(nn.input)
    # cam_rgb.preview.link(nn.input)

    return pipeline, model


def run(pipeline, model):
    # Pipeline is now finished, and we need to find an available device to run our pipeline
    # we are using context manager here that will dispose the device after we stop using it
    with depthai.Device(pipeline) as device:
        # From this point, the Device will be in "running" mode and will start sending data via XLink

        # To consume the device results, we get two output queues from the device, with stream names we assigned earlier
        q_rgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
        q_nn = device.getOutputQueue("nn", maxSize=4, blocking=False)

        # Here, some default values are defined. Frame will be an image from "rgb" stream,
        frame = None

        # Main loop
        while True:
            # Instead of get (blocking), we use tryGet (nonblocking)
            # which will return the available data or None otherwise
            in_rgb = q_rgb.tryGet()
            in_nn = q_nn.tryGet()

            if in_rgb is not None:
                # Retrieve 'bgr' (opencv format) frame
                frame = in_rgb.getCvFrame()

            if frame is not None:
                box_locations, text_locations, text_box_locations, det_labels = detect(model, frame, min_score=0.8,
                                                                                       max_overlap=0.7, top_k=200)

                # Draw the bounding boxes on the frame
                for i in range(len(box_locations)):
                    box_location = box_locations[i]
                    text_location = text_locations[i]
                    text_box_location = text_box_locations[i]
                    box_location = [int(i) for i in box_location]
                    text_location = [int(i) for i in text_location]
                    text_box_location = [int(i) for i in text_box_location]
                    cv2.rectangle(frame, (box_location[0], box_location[1]),
                                  (box_location[2], box_location[3]), (0, 255, 0), 2)
                    cv2.rectangle(frame, (text_box_location[0], text_box_location[1]),
                                  (text_box_location[2], text_box_location[3]), (0, 255, 0), -1)
                    cv2.putText(frame, det_labels[i].upper(), (text_location[0], text_location[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Show the frame
                cv2.imshow("rgb", frame)
                if cv2.waitKey(1) == ord('q'):
                    break

    # The device is disposed automatically when exiting the 'with' statement


if __name__ == '__main__':
    pipeline, model = configure()
    run(pipeline, model)
