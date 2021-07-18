# Vision Impulse Test

Goal of this project is to prototype and showcase a segmentation model on a custom geo-dataset. 

### Design Choices
* Pytorch along with Pytorch Lightning [1] is used for prototyping the model. 
  Pytorch Lightning is used to scale the model training on multiple GPUs/TPUs.        
* Due to time / computational constraints, the model was trained only for 50 epochs and the results obtained can be further improved.
* UNet architecture similar to [2] was used for the task.
  <img width="640" height="240" src="https://github.com/njanirudh/Vision-Impulse-Test/blob/feature-segmentation/assets/unet_architecture.png" title="Input Image Channels">
* The trainer supports both 3 channels (RGB) and 12 channels (bands) for training and inference.
* Note: We used [0-11] zero indexed counting for channels/bands.
  
### Code
* 'src/main.py' can be used for training and inference.
* Due to time constraint we have not written a commandline application for running the code.  
* DATASET_PATH can be replaced by path to {Dataset}. The dataset folder contains {Dataset/images} and {Dataset/labels}.

### Input Image
The input 'tif.' image contains 12 channels.  

<img width="240" height="200" src="https://github.com/njanirudh/Vision-Impulse-Test/blob/feature-segmentation/assets/12_img_collage.png" title="Input Image Channels shown in grey map">

-----------------------------------------
Individual channels with countours:      
<img src="https://github.com/njanirudh/Vision-Impulse-Test/blob/feature-segmentation/assets/graphs/c0_p.png" width="100"/> <img src="https://github.com/njanirudh/Vision-Impulse-Test/blob/feature-segmentation/assets/graphs/c0_c.png" width="100"/> 

<img src="https://github.com/njanirudh/Vision-Impulse-Test/blob/feature-segmentation/assets/graphs/c1_p.png" width="100"/> <img src="https://github.com/njanirudh/Vision-Impulse-Test/blob/feature-segmentation/assets/graphs/c1_c.png" width="100"/> 

<img src="https://github.com/njanirudh/Vision-Impulse-Test/blob/feature-segmentation/assets/graphs/c2_p.png" width="100"/> <img src="https://github.com/njanirudh/Vision-Impulse-Test/blob/feature-segmentation/assets/graphs/c2_c.png" width="100"/> 

<img src="https://github.com/njanirudh/Vision-Impulse-Test/blob/feature-segmentation/assets/graphs/c3_p.png" width="100"/> <img src="https://github.com/njanirudh/Vision-Impulse-Test/blob/feature-segmentation/assets/graphs/c3_c.png" width="100"/> 

<img src="https://github.com/njanirudh/Vision-Impulse-Test/blob/feature-segmentation/assets/graphs/c4_p.png" width="100"/> <img src="https://github.com/njanirudh/Vision-Impulse-Test/blob/feature-segmentation/assets/graphs/c4_c.png" width="100"/> 

<img src="https://github.com/njanirudh/Vision-Impulse-Test/blob/feature-segmentation/assets/graphs/c5_p.png" width="100"/> <img src="https://github.com/njanirudh/Vision-Impulse-Test/blob/feature-segmentation/assets/graphs/c5_c.png" width="100"/> 

<img src="https://github.com/njanirudh/Vision-Impulse-Test/blob/feature-segmentation/assets/graphs/c6_p.png" width="100"/> <img src="https://github.com/njanirudh/Vision-Impulse-Test/blob/feature-segmentation/assets/graphs/c6_c.png" width="100"/> 

<img src="https://github.com/njanirudh/Vision-Impulse-Test/blob/feature-segmentation/assets/graphs/c7_p.png" width="100"/> <img src="https://github.com/njanirudh/Vision-Impulse-Test/blob/feature-segmentation/assets/graphs/c7_c.png" width="100"/> 

<img src="https://github.com/njanirudh/Vision-Impulse-Test/blob/feature-segmentation/assets/graphs/c8_p.png" width="100"/> <img src="https://github.com/njanirudh/Vision-Impulse-Test/blob/feature-segmentation/assets/graphs/c8_c.png" width="100"/> 

<img src="https://github.com/njanirudh/Vision-Impulse-Test/blob/feature-segmentation/assets/graphs/c9_p.png" width="100"/> <img src="https://github.com/njanirudh/Vision-Impulse-Test/blob/feature-segmentation/assets/graphs/c9_c.png" width="100"/> 

<img src="https://github.com/njanirudh/Vision-Impulse-Test/blob/feature-segmentation/assets/graphs/c10_p.png" width="100"/> <img src="https://github.com/njanirudh/Vision-Impulse-Test/blob/feature-segmentation/assets/graphs/c10_c.png" width="100"/> 

<img src="https://github.com/njanirudh/Vision-Impulse-Test/blob/feature-segmentation/assets/graphs/c11_p.png" width="100"/> <img src="https://github.com/njanirudh/Vision-Impulse-Test/blob/feature-segmentation/assets/graphs/c11_c.png" width="100"/> 

-----------------------------------------
RGB image (Channel 4,3,2):     
<img src="https://github.com/njanirudh/Vision-Impulse-Test/blob/feature-segmentation/assets/graphs/rgb.png" width="200"/>

### Result
Example output after inference.      

<img width="480" height="200" src="https://github.com/njanirudh/Vision-Impulse-Test/blob/feature-segmentation/assets/results.png" title="Input Image Channels">
The (left) image shows the output of the model trained only on the RGB channels (4,3,2)    <br />
The (right) image shows the output of model trained on all the channels (1-12)

#### References
1. https://pytorch-lightning.readthedocs.io/en/latest/
1. https://medium.com/analytics-vidhya/creating-a-very-simple-u-net-model-with-pytorch-for-semantic-segmentation-of-satellite-images-223aa216e705
1. https://rasterio.readthedocs.io/en/latest/topics/plotting.html
