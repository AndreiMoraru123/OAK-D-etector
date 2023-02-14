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

The original PyTorch implementation of the model, and the one that I am following here, is [this one](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection). 
This is a fantastic guide by itself and I did not modify much as of now. The goal for this project is to get to deploy such a custom model on real hardware, rather than neural network design.

In this regard, I am using a [Luxonis OAK-D Lite](https://shop.luxonis.com/products/oak-d-lite-1) and an [Intel Neural Compute Stick 2](https://www.intel.com/content/www/us/en/developer/articles/tool/neural-compute-stick.html). Funnily enough, just as I finished this, the NCS2 became outdated, as you can see in the Intel docs, since Intel will be discontinuing it. But that besides the point here, as the main focus is deployment on specific hardware, whatever that hardware may be. 

Namely, we are looking at VPU's, or [vision processing units](https://en.wikipedia.org/wiki/Vision_processing_unit). One such AI accelerator is found in the OAK-D camera itself! 

## Setup 

In order to communicate with the hardware, I am using both [DepthAI's api](https://docs.luxonis.com/en/latest/) to communicate with the RGB camera of the OAK-D, and Intel's [OpenVINO](https://docs.openvino.ai/latest/home.html) (Open Visual Inference and Neural Optimization) for deployment, both of which are still very much state of the art in Edge AI.

Now, in order to use OpenVINO with hardware, one has to [download the distribution toolkit](https://docs.openvino.ai/2021.1/openvino_docs_install_guides_installing_openvino_windows.html). For compiling and running apps, the library prefers to set up temporary variables, so we will do it that way. For me, the path looks like this:

```bash
cd C:\Program Files (x86)\Intel\openvino_2022\w_openvino_toolkit_windows_2022.2.0.7713.af16ea1d79a_x86_64
```

where I can now setup the variables by running the batch file:

```bash
setupvars.bat
```

One will get a message back saying: 

```bash
OpenVINO environment initialized
```

And now (and only now) can I open my Python editor from the same command prompt:

```bash
pycharm
```

Otherwise, the hardware will not be recognized.

## Hardware

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

OpenVINO can also be custom built for CUDA, just like one would do with OpenCV, which I did not here, but in which case the `CUDA` device will also show up. 

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

#### Question: Why use both the NCS2 and the OAK-D if the OAK-D suffices?
#### Answer: No reason to! 
Honestly, I just had one laying around, but since it's double the fun this way, I can run the frames on the camera, and compute on the stick.
One could use just the camera's VPU the exact same way, using OpenVINO.

## Deployment

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

## Running the model

See [run.py](https://github.com/AndreiMoraru123/ObjectDetection/blob/main/run.py)

After getting the model in `onnx` format, I can use OpenVINO's inference engine to load it:

```python
from openvino.inference_engine import IECore

ie = IECore()
net = ie.read_network(model=new_model + '-sim' + '.onnx')

# Load model to the device
exec_net = ie.load_network(network=net, device_name='MYRIAD')
```

Now I can use my `exec_net` to infer on my input data:

```python
net.infer({'input': frame.unsqueeze(0).numpy()})  # inference on the camera frame.
```

And that's it!

The rest of the code if configuring the [pipeline](https://docs.luxonis.com/projects/api/en/latest/components/pipeline/). You can check out this awesome guide from [pyimagesearch](https://pyimagesearch.com/2022/12/19/oak-d-understanding-and-running-neural-network-inference-with-depthai-api/) to see exactly what each link means.

## Parallelization

OpenVINO has this cool feature where I can infer on multiple threads. In order to do this, I only have to change the loading to accomodate multiple requests:

```python
exec_net = ie.load_network(network=net, device_name=device, num_requests=2)
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

## Demo

## Work in progress

![work in progress](https://user-images.githubusercontent.com/81184255/217096698-b6116802-cb00-412c-91b9-6b22d7718ead.png)
