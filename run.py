import cv2
import depthai as dai
import argparse

import numpy as np
import openvino.inference_engine.ie_api as api
from PIL import Image
from pathlib import Path
from torchvision import transforms
from detect import detect_objects, model
from utils import *
from openvino.inference_engine import IECore
from openvino.runtime import Core
from cv2 import cuda
from model import SSD300

cuda.setDevice(0)
runtime = Core()
devices = runtime.available_devices

for device in devices:
    device_name = runtime.get_property(device, "FULL_DEVICE_NAME")
    print(f"{device}: {device_name}")


def generate_engine(new_model: str, device: str) -> api.ExecutableNetwork:
    """
    Deploy model from ONNX to OpenVINO
    :param device: the device to deploy the model, such as 'CPU', 'GPU' or 'MYRIAD'
    :param new_model: the name of the ONNX model, such as 'ssd300'
    :return:
    """

    # Read model from ONNX
    ie = IECore()
    net = ie.read_network(model=new_model + '-sim' + '.onnx')

    # Load model to the device
    exec_net = ie.load_network(network=net, device_name=device, num_requests=2)

    # Save model to IR
    exec_net.export(new_model + '-' + device + '.xml')

    return exec_net


def configure(is_blob: bool = False, blob_path: str = None) -> dai.Pipeline:
    """
    Configure the pipeline
    :param blob_path: the path to the blob model, such as 'models/ssd300.blob'
    :param is_blob: True if the model is a blob, False if the model is an ONNX
    :return: DepthAI pipeline
    """

    # Start defining a pipeline
    pipeline = dai.Pipeline()

    # Define sources and outputs
    cam_rgb = pipeline.createColorCamera()
    xout_rgb = pipeline.createXLinkOut()

    # Properties
    cam_rgb.setPreviewSize(1000, 500)
    cam_rgb.setInterleaved(False)
    cam_rgb.setFps(35)

    # Linking
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)

    if is_blob:
        nn = pipeline.createNeuralNetwork()
        xout_nn = pipeline.createXLinkOut()
        xout_nn.setStreamName("nn")
        nn.out.link(xout_nn.input)

        nn.setBlobPath(Path(blob_path))
        nn.setNumInferenceThreads(2)
        nn.input.setBlocking(False)
        nn.setNumPoolFrames(4)

        cam_rgb.preview.link(nn.input)

    return pipeline


def detect(net, frame, min_score, max_overlap, top_k, suppress=None) -> tuple:
    """
    Detect objects in a frame
    :param net: the neural network, either a blob, an OpenVINO model or a PyTorch model
    :param frame: the frame to detect objects in on the Luxonis camera
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via
     Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image, or you do not want in the image
    :return: box_locations, text_locations, text_box_locations, det_labels
    """

    # Define transforms
    resize = transforms.Resize((300, 300))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    # Convert to PIL Image
    original_image = Image.fromarray(frame)

    # Transform
    image = normalize(to_tensor(resize(original_image)))

    det_boxes, det_labels, det_scores = None, None, None

    # If net is a blob, use the blob model
    if isinstance(net, dai.NNData):
        pass

    # If net is an IECore, use the OpenVINO model
    elif isinstance(net, api.ExecutableNetwork):

        # Start the first inference request asynchronously
        infer_request_handle1 = net.start_async(request_id=0, inputs={'input': image.unsqueeze(0).numpy()})

        # Start the second inference request asynchronously
        infer_request_handle2 = net.start_async(request_id=1, inputs={'input': image.unsqueeze(0).numpy()})

        # Wait for the first inference request to complete
        if infer_request_handle1.wait() == 0:
            # Get the results
            predicted_locs = np.array(infer_request_handle1.output_blobs['boxes'].buffer, dtype=np.float32)
            predicted_scores = np.array(infer_request_handle1.output_blobs['scores'].buffer, dtype=np.float32)

            # Send the results as tensors to the GPU
            predicted_locs = torch.from_numpy(predicted_locs).to('cuda')
            predicted_scores = torch.from_numpy(predicted_scores).to('cuda')

            # Detect objects
            det_boxes, det_labels, det_scores = detect_objects(predicted_locs, predicted_scores,
                                                               max_overlap=max_overlap,
                                                               min_score=min_score,
                                                               top_k=top_k)

        # Wait for the second inference request to complete
        if infer_request_handle2.wait() == 0:
            # Get the results
            predicted_locs = np.array(infer_request_handle2.output_blobs['boxes'].buffer, dtype=np.float32)
            predicted_scores = np.array(infer_request_handle2.output_blobs['scores'].buffer, dtype=np.float32)

            # Send the results as tensors to the GPU
            predicted_locs = torch.from_numpy(predicted_locs).to('cuda')
            predicted_scores = torch.from_numpy(predicted_scores).to('cuda')

            # Detect objects
            det_boxes, det_labels, det_scores = detect_objects(predicted_locs, predicted_scores,
                                                               max_overlap=max_overlap,
                                                               min_score=min_score,
                                                               top_k=top_k)

    # If net is pytorch checkpoint, use the PyTorch model
    elif isinstance(net, SSD300):

        # Move to default device
        net = net.to('cuda')
        image = image.to('cuda')

        # Forward prop.
        predicted_locs, predicted_scores = net(image.unsqueeze(0))

        # Detect objects in SSD output
        det_boxes, det_labels, det_scores = detect_objects(predicted_locs, predicted_scores,
                                                           max_overlap=max_overlap,
                                                           min_score=min_score,
                                                           top_k=top_k)

    else:
        raise ValueError('The net must be either a blob, an OpenVINO model or a PyTorch model')

    # Move detections to CUDA
    det_boxes = det_boxes[0].to('cuda')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor([original_image.width, original_image.height,
                                       original_image.width, original_image.height]).unsqueeze(0).to('cuda')

    det_boxes = det_boxes * original_dims  # scale to original dimensions

    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cuda').tolist()]

    # If no objects found, the detection pipeline has failed
    if det_boxes is None:
        print('No objects found!')

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


def run(pipeline, is_blob: bool = False, net=None, min_score: float = 0.2, max_overlap: float = 0.5, top_k: int = 200):
    """
    Run the pipeline
    :param pipeline: DepthAI pipeline
    :param is_blob: True if the model is a blob, False if the model is an ONNX
    :param net: the model to use for inference
    :return:
    """
    # Connect to device and start pipeline
    with dai.Device(pipeline) as device:
        # Output queue will be used to get the rgb frames from the output defined above
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        if is_blob:
            q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

        frame = None

        while True:
            in_rgb = q_rgb.tryGet()

            if is_blob:
                in_nn = q_nn.tryGet()

            if in_rgb is not None:
                frame = in_rgb.getCvFrame()

            if frame is not None:
                box_locations, text_locations, text_box_locations, det_labels = detect(net, frame, min_score=min_score,
                                                                                       max_overlap=max_overlap,
                                                                                       top_k=top_k)

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


def choose_hardware():
    """
    Choose the hardware to run the model on
    :return: the hardware to run the model on
    """
    hardware = input("Choose the hardware to run the model on (MYRIAD or GPU): ")

    switcher = {
        "MYRIAD": generate_engine(args.new_model, args.device),
        "GPU": model
    }

    return switcher.get(hardware, "Invalid hardware")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a PyTorch model on DepthAI')
    parser.add_argument('--is_blob', action='store_true', default=False, help='If the model is a blob')
    parser.add_argument('--blob_path', type=str, default=None, help='Path to the blob file')
    parser.add_argument('--device', type=str, default="MYRIAD", help='the device to deploy the model')
    parser.add_argument('--new_model', default="ssd300", type=str, help='the name of the ONNX model')
    parser.add_argument('--min_score', default=0.8, type=float, help='the minimum score for a box to be considered')
    parser.add_argument('--max_overlap', default=0.5, type=float, help='the maximum overlap for a box to be considered')
    parser.add_argument('--top_k', default=200, type=int, help='the maximum number of boxes to be considered')

    args = parser.parse_args()

    neural_network = choose_hardware()

    # Create pipeline
    pipeline = configure(args.is_blob, args.blob_path)
    run(pipeline, args.is_blob, neural_network, args.min_score, args.max_overlap, args.top_k)
