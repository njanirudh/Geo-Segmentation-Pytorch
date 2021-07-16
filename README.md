# Vision Impulse Test

Goal of this project is to prototype and showcase a segmentation model on a custom geo-dataset. 

### Design Choices
* Pytorch along with Pytorch Lightning [1] is used for prototyping the model. 
  Pytorch Lightning is used to scale the model training on multiple GPUs/TPUs.        
* Due to time / computational constraints, the model was trained only for a few 10s epochs.
* UNet architecture similar to [2] was used for the task.
  <img width="640" height="480" src="" title="Input Image Channels">
* The trainer supports both 3 channels (RGB) and 12 channels (bands) for training and inference.

### Input Image
<img width="640" height="480" src="" title="Input Image Channels">

### Result
<img width="640" height="480" src="" title="Input Image Channels">

#### References
1. https://pytorch-lightning.readthedocs.io/en/latest/
1. https://medium.com/analytics-vidhya/creating-a-very-simple-u-net-model-with-pytorch-for-semantic-segmentation-of-satellite-images-223aa216e705
