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

The original implementation of the model, and the one that I am following here is [this one](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection). 
This is a fantastic guide by itself and I did not modify much as of now. The goal for this project is to get to deploy such a custom model on real hardware, rather than neural network design

In this regard, I am using a [Luxonis OAK-D Lite](https://shop.luxonis.com/products/oak-d-lite-1) and an [Intel Neural Compute Stick 2](https://www.intel.com/content/www/us/en/developer/articles/tool/neural-compute-stick.html). Funnily enough, just as I finished this, the NCS2 became outdated, as you can see in the Intel docs, since Intel will be discontinuing it. But that besides the point here, as the main focus is deployment on specific hardware, whatever that hardware may be. 

Namely, we are looking at VPU's, or [vision processing units](https://en.wikipedia.org/wiki/Vision_processing_unit). One such AI accelerator is found in the OAK-D camera itself! 

In order to communicate with the hardware, I am using both [DepthAI's api](https://docs.luxonis.com/en/latest/) to communicate with the RGB camera of the OAK-D, and Intel's [OpenVINO](https://docs.openvino.ai/latest/home.html) for deployment, both of which are still very much state of the art in Computer Vision.

Now, in order to use OpenVino with hardware, one has to [download the distribution toolkit](https://docs.openvino.ai/2021.1/openvino_docs_install_guides_installing_openvino_windows.html). For compiling and running apps, the library prefers to set up temporary variables, so we will do it that way. For me, the path looks like this:

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



## Work in progress

![work in progress](https://user-images.githubusercontent.com/81184255/217096698-b6116802-cb00-412c-91b9-6b22d7718ead.png)
