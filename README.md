# SSD Object Detection on OAK-D Lite via DepthAI

<table>
  <tr>
    <td rowspan="6"><img src="https://user-images.githubusercontent.com/81184255/218581688-f960647d-d5d8-437a-bde7-335483a07478.jpg" width="600" height = "650"/></td>
    <td> <img src="https://user-images.githubusercontent.com/81184255/218574879-86310f35-333c-4d3d-a9dc-7fe805b8b714.png" width="370" height = "90"/> </td>
  </tr>
  <tr>
    <td> <img src="https://user-images.githubusercontent.com/81184255/218578605-1b852fd0-8191-49e4-9568-7d45a7595f68.jpg" width="370" height = "90"/> </td>
  </tr>
  <tr>
    <td> <img src="https://user-images.githubusercontent.com/81184255/218580166-03e9d444-0357-42f5-9099-1446bc3514c7.jpg" width="370" height = "90"/> </td>
  </tr>
  <tr>
    <td> <img src="https://user-images.githubusercontent.com/81184255/218578649-531d0116-0d31-40f0-830b-f8bc38084ff6.jpg" width="370" height = "90"/> </td>
  </tr>
  <tr>
    <td> <img src="https://user-images.githubusercontent.com/81184255/218579553-fd3baf98-35fb-416a-9aad-6ff74623eca8.jpg" width="370" height = "90"/> </td>
  </tr>
  <tr>
    <td> <img src="https://user-images.githubusercontent.com/81184255/218582810-1d949a60-81f6-46f5-8364-1f14391d16e9.jpg" width="370" height = "90"/> </td>
  </tr>
</table>

# Demo

<p align="center">
  <img src="https://user-images.githubusercontent.com/81184255/219051190-03b0ee4c-c006-4fa6-ba9f-9e1ec8092a87.gif" with = "400" height = "300" />
</p>

# Intro

The original PyTorch implementation of the model, and the one that I am following here, is [this one](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection). 
This is a fantastic guide by itself and I did not modify much as of now. The goal for this project is to get to deploy such a custom model on real hardware, rather than neural network design.

In this regard, I am using a [Luxonis OAK-D Lite](https://shop.luxonis.com/products/oak-d-lite-1) and an [Intel Neural Compute Stick 2](https://www.intel.com/content/www/us/en/developer/articles/guide/get-started-with-neural-compute-stick.html). Funnily enough, just as I finished this, the NCS2 became outdated, as you can see on the main page, since [Intel will be discontinuing it](https://www.intel.com/content/www/us/en/developer/articles/tool/neural-compute-stick.html). But that is besides the point here, as the main focus is deployment on specific hardware, whatever that hardware may be. 

Namely, we are looking at VPU's, or [vision processing units](https://en.wikipedia.org/wiki/Vision_processing_unit).

One such AI accelerator can be found in the [OAK-D camera](https://github.com/luxonis/depthai-hardware/blob/master/DM9095_OAK-D-LITE_DepthAI_USB3C/Datasheet/OAK-D-Lite_Datasheet.pdf) itself! 

# Setup 

In order to communicate with the hardware, I am using both [DepthAI's api](https://docs.luxonis.com/en/latest/) to communicate with the RGB camera of the OAK-D, and Intel's [OpenVINO](https://docs.openvino.ai/latest/home.html) (Open Visual Inference and Neural Optimization) for deployment, both of which are still very much state of the art in Edge AI.

Now, in order to use OpenVINO with hardware, I have to [download the distribution toolkit](https://docs.openvino.ai/2021.1/openvino_docs_install_guides_installing_openvino_windows.html). For compiling and running apps, the library prefers to set up temporary variables, so we will do it that way. 

For me, the path looks like this:

```bash
cd C:\Program Files (x86)\Intel\openvino_2022\w_openvino_toolkit_windows_2022.2.0.7713.af16ea1d79a_x86_64
```

where I can now setup the variables by running the batch file:

```bash
setupvars.bat
```

I will get a message back saying: 

```bash
Python 3.7.7
[setupvars.bat] OpenVINO environment initialized
```

And now (and only now) can I open my Python editor from the same command prompt:

```bash
pycharm
```

Otherwise, the hardware will not be recognized.

# Hardware

I can run the following script to make sure ensure the detection of the device(s):

```python
from openvino.runtime import Core

runtime = Core()
devices = runtime.available_devices

for device in devices:
    device_name = runtime.get_property(device, "FULL_DEVICE_NAME")
    print(f"{device}: {device_name}")
```

If I setup the variables correctly (by running the batch script), I get this:

```
[E:] [BSL] found 0 ioexpander device
CPU: AMD Ryzen 7 4800H with Radeon Graphics         
GNA: GNA_SW
MYRIAD: Intel Movidius Myriad X VPU
```

`BSL` here refers to a bootloader meant to initialize the firmware, and 0 ioexpander means no (I/O) expander devices (used to expand the number of pins).

`GNA` refers to something called "Gaussian Neural Accelerator", which is another intel accelerator we will not be dealing with here.

As you probably guessed, `MYRIAD` is the device I connected, and it is the same for both the OAK-D camera and the NCS2 stick, since they are both the same VPU.

Also, look, that's my `CPU`!

OpenVINO can also be custom built for CUDA, very much like OpenCV, which I did not do here, but in that case the `CUDA` device will also show up. 

If I run this script with both devices connected, you can see they get ID's, for the USB position they are taking in the connection (`.1` and `.3`):

```
[E:] [BSL] found 0 ioexpander device
CPU: AMD Ryzen 7 4800H with Radeon Graphics         
GNA: GNA_SW
MYRIAD.6.1-ma2480: Intel Movidius Myriad X VPU
MYRIAD.6.3-ma2480: Intel Movidius Myriad X VPU
```

And if I connect nothing, or if I forget to initialize my OpenVINO environment, obviously I only get this:

```
[E:] [BSL] found 0 ioexpander device
CPU: AMD Ryzen 7 4800H with Radeon Graphics         
GNA: GNA_SW
```

#### Question: Why use the NCS2 if the OAK-D can play the role of the VPU?
#### Answer: No reason to! 
Honestly, I just had one laying around, but since it's double the fun this way, I can run the frames on the camera, and compute on the stick.
I could use just the camera's VPU the exact same way, using OpenVINO.

# Deployment

See [deploy.py](https://github.com/AndreiMoraru123/ObjectDetection/blob/main/deploy.py)

In order to run the model on the OpenVINO's inference engine, I must first convert it to [`onnx`](https://onnx.ai/) (Open Neural Network eXchange) format, as PyTorch models do not have their own deployment systems, such as TensorFlow's frozen graphs. It's important here to also export input and output names, because in some cases, such as the object detector here, the forward pass my return multiple tensors:

```python
# Load model checkpoint
checkpoint = torch.load(model_path, map_location='cuda')
model = checkpoint['model']
model.eval()

# Export to ONNX
input_names = ['input']
output_names = ['boxes', 'scores']
dummy_input = torch.randn(1, 3, 300, 300).to('cuda')
torch.onnx.export(model, dummy_input, new_model+'.onnx', verbose=True,
                  input_names=input_names, output_names=output_names)

# Simplify ONNX
simple_model, check = simplify('ssd300.onnx')
assert check, "Simplified ONNX model could not be validated"
onnx.save(simple_model, new_model+'-sim'+'.onnx')
```

Notice the `output_names` list given as a parameter. In the case of SSD, the model outputs both predicted locations (8731=priors, 4=coordinates) and class scores (8732=priors, 21=classes), like all object detectors. It's important to separate the two, which is easy to do with `torch.onnx.export`, but also easy to forget.

# Running the model

See [run.py](https://github.com/AndreiMoraru123/ObjectDetection/blob/main/run.py)

### There are three options for inference hardware, and I will go through all of them:

```python
switcher = {
    "NCS2": generate_engine(args.new_model, args.device),  # inference engine for the NCS2
    "CUDA": model,  # since tensors are already on CUDA, the model is just the loaded checkpoint
    "OAK-D": None  # since the model is already on the device, this is just using the blob boolean
}
```

## Running on CUDA via checkpoint

This is straightforward, and not very interesting here. Since the tensors are already on CUDA, I can just load the `checkpoint` in PyTorch and run the model using the forward call. I did put this option in here since it's the fastest, and best for showing demos. I could have also included `CPU` as an option that would have the same flow in the code, but why would anyone want that? Ha.

## Running on the NCS2/camera via OpenVINO's inference engine

After getting the model in `onnx` format, I can use OpenVINO's inference engine to load it:

```python
from openvino.inference_engine import IECore

ie = IECore()
net = ie.read_network(model=new_model + '-sim' + '.onnx')

# Load model to the device
net = ie.load_network(network=net, device_name='MYRIAD')
```

Which one of the two `MYRIAD` devices is the inference engine using? Whichever it finds first. You can specify the exact ID if you want to. 

Then I can use my `net` to infer on my input data:

```python
net.infer({'input': frame.unsqueeze(0).numpy()})  # inference on the camera frame.
```

And that's it! I can now configure the pipeline:

```python
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
```

> **Note**
> I deliberately do not create a DepthAI Neural Network node here, because I am running the inference via the OpenVINO ExecutableNetwork.

### Parallelization & Ouputs

OpenVINO has this cool feature where I can infer on multiple threads. In order to do this, I only have to change the loading to accomodate multiple requests:

```python
net = ie.load_network(network=net, device_name=device, num_requests=2)
```

which I can then start asynchronously:

```python
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

```

## Running on the OAK-D using `blob` format

I can skip OpenVINO completely and work with the neural network as a binary object.

I still need to get my model in `onnx` format, but I need to convert it to binary using `blobconverter`. Take a look at `deploy_blob` in [deploy.py](https://github.com/AndreiMoraru123/ObjectDetection/blob/main/deploy.py).

If I'm working with the blob, I need to create a `NeuralNetwork` node and link it's input to the camera preview like this:

```python
if is_blob:
    cam_rgb.setPreviewSize(300, 300)  # Note the dimension here
    print("Creating Blob Neural Network...")
    nn = pipeline.createNeuralNetwork()
    xout_nn = pipeline.createXLinkOut()
    xout_nn.setStreamName("nn")
    nn.out.link(xout_nn.input)

    nn.setBlobPath(Path(blob_path))
    nn.setNumInferenceThreads(2)
    nn.input.setBlocking(False)
    nn.setNumPoolFrames(4)

    cam_rgb.preview.link(nn.input)
```

> **Warning**
> If the preview here is not in the shape of the input expected by the Neural Network node (300,300), the predicted bounding boxes will be out of sight.

After that, by queuing my nn:

```python
q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
```

I can get my predictions directly by using `get`:

```python
in_nn = q_nn.get()
```

And now I can obtain the outputs via the names I have exported them with by previously deploying as `onnx`:

```python
predicted_locs = in_nn.getLayerFp16("boxes")
predicted_scores = in_nn.getLayerFp16("scores")

# Make numpy arrays
predicted_locs = np.array(predicted_locs, dtype=np.float32)
predicted_scores = np.array(predicted_scores, dtype=np.float32)

# Reshape locs into 4 boxes * 8732 anchors
predicted_locs = np.reshape(predicted_locs, (8732, 4))

# Reshape scores into 21 classes * 8732 anchors
predicted_scores = np.reshape(predicted_scores, (8732, 21))

# Make torch tensors
predicted_locs = torch.from_numpy(predicted_locs).to('cuda')
predicted_scores = torch.from_numpy(predicted_scores).to('cuda')

# Add batch dimension
predicted_locs = predicted_locs.unsqueeze(0)
predicted_scores = predicted_scores.unsqueeze(0)
```

Which, after a bit of tensor engineering, can be used for detecting the objects (see `detect_objects` in [detect.py](https://github.com/AndreiMoraru123/ObjectDetection/blob/main/detect.py).

# Outro

![ha](https://user-images.githubusercontent.com/81184255/219082947-24ba1e97-6b24-4930-9a6c-87526d9d0494.jpg)
