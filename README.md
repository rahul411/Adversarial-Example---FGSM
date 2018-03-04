# Adversarial-Example---FGSM
Crafting adversarial example for semantic segmentaion model using FGSM.

Here, I have used the pre-trained PSPNet for semantic segmentation trained on Cityscapes dataset. The code for PSPNet is taken from this [repository](https://github.com/hellochick/PSPNet-tensorflow). Please follow the github link, to run the model for segmentation. 

The code to develop adversarial example for a given image, run the following command: 
```sh
$ python adversarialExample.py
```
This will create an adversarial image, and store it in the input folder.
To test the segmentation result on this perturbed image, run the following command.
```sh
$ python inference.py --dataset=cityscapes --img-path=/path/to/perturbedImage --checkpoints=model
```
#### Results
![Original Image](input/test_1024x2048.png?raw=true )
![Original Segmentation result](output/test_1024x2048.png?raw=true )
![Perturbed Image after 80 iterations](input/advImage80.png?raw=true )
![Perturbed Image after 80 iterations- Segmentation result](output/advImage80.png?raw=true )
![Perturbed Image after 40 iterations](input/advImage40.png?raw=true )
![Perturbed Image after 40 iterations- Segmentation result](output/advImage40.png?raw=true )