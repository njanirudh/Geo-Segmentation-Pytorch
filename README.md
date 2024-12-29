# Geo Segmentation 
Goal of this project is to prototype and showcase a segmentation model on a custom geo-dataset. 

### Design Choices
* Pytorch along with Pytorch Lightning [1] is used for prototyping the model. 
  Pytorch Lightning is used to scale the model training on multiple GPUs/TPUs.        
* Due to time / computational constraints, the model was trained only for 50 epochs and the results obtained can be further improved.
* UNet architecture similar to [2] was used for the task. Due to unavailability of pre-trained models trained on required classes, we preferred custom model.
  <img width="640" height="240" src="https://github.com/njanirudh/Vision-Impulse-Test/blob/feature-segmentation/assets/unet_architecture.png" title="Input Image Channels">
* The trainer supports both 3 channels (RGB) and 12 channels (bands) for training and inference.
* Note: We used [0-11] zero indexed counting for channels/bands.
  
### Code
* 'src/main.py' can be used for training and inference.
* 'src/sandbox' contains prototype code.
* Code is designed to be modular. This will make prototyping different models, hyperparameter tuning easier.  
* Due to time constraint we have not written a commandline application for running training and inference.  
* DATASET_PATH can be replaced by path to {Dataset}. The dataset folder contains {Dataset/images} and {Dataset/labels}.
* The code for graphing loss values are still in progress. 

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

From exif data [3]: <br />
* Pixel Scale 
< 10 10 0 >
* Model Tie Point
< 0 0 0 726830 5544990 0 >
* Gt Model Type
< Projected >
* Gt Raster Type
< Pixel Is Area >
* Gt Citation
< WGS 84 / UTM zone 31N >
* Geog Citation
< WGS 84 >
* Geog Angular Units
< Angular Degree >


### Result
Example output after inference.      

<img width="480" height="200" src="https://github.com/njanirudh/Vision-Impulse-Test/blob/feature-segmentation/assets/results.png" title="Input Image Channels">
The (left) image shows the output of the model trained only on the RGB channels (4,3,2)    <br />
The (right) image shows the output of model trained on all the channels (1-12) <br />

Since the model is not trained till convergance, the results are not directly usable.

#### References
1. https://pytorch-lightning.readthedocs.io/en/latest/
1. https://medium.com/analytics-vidhya/creating-a-very-simple-u-net-model-with-pytorch-for-semantic-segmentation-of-satellite-images-223aa216e705
1. https://www.metadata2go.com
1. https://www.pcigeomatics.com/geomatica-help/COMMON/concepts/TiePoint_explain.html
