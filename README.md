# BaRT-demo
Testing adversarial images against Barrage of Random Transforms Defense method

Original paper:

http://openaccess.thecvf.com/content_CVPR_2019/html/Raff_Barrage_of_Random_Transforms_for_Adversarially_Robust_Defense_CVPR_2019_paper.html

## BaRT repo: 

https://github.com/XttyCTL9/BaRTDefense

This project was built in:<br>
Tensorflow 1.14.0<br>
scipy 1.1.0<br>
scikit-image 0.16.2<br>
Pillow 6.0.0<br>


Checkpoint downloaded from:

https://pan.baidu.com/s/1yAqmqfSODylaZQuF8nZ3vw

## Adversarial images:

There are 10 adversarial images generated under the adversarial_images_resnet50 folder. <br>

These adversarial images are generated using the [Trusted-AI
adversarial-robustness-toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox).

The attack method was performed using the HopSkipJump attack on resnet_v2_50 using the untargeted attack mode.
