# [Attract, Perturb, and Explore: Learning a Feature Alignment Network for Semi-supervised Domain Adaptation (ECCV 2020)](https://arxiv.org/pdf/2007.09375.pdf)

### Acknowledgment

The implementation is built on the pytorch implementation of [SSDA_MME](https://github.com/VisionLearningGroup/SSDA_MME) and we refer a specific module in [DTA](https://github.com/postBG/DTA.pytorch).

### Prerequisites

* CUDA 10.0 or 10.1
* Python 3.7 (or 3.6)
* Pytorch 1.0.1
```
conda install pytorch==1.0.1 torchvision==0.2.2 cudatoolkit=10.0 -c pytorch
```
* Pillow, numpy, tqdm
* You can easily install dependencies through
```
pip install -r requirements.txt
```

### Dataset Structure
You can download the datasets by following the instructions in [SSDA_MME](https://github.com/VisionLearningGroup/SSDA_MME).
```
data---
     |
   multi---
     |   |
     |  Real
     |  Clipart
     |  Product
     |  Real
   office_home---
     |         |
     |        Art
     |        Clipart
     |        Product
     |        Real
   office---
     |    |
     |   amazon
     |   dslr
     |   webcam
   txt---
       | 
      multi---
       |    |
       |   labeled_source_images_real.txt
       |   unlabeled_target_images_real_3.txt
       |   labeled_target_images_real_3.txt         
       |   unlabeled_source_images_sketch.txt
       |                  ...
      office---
       |     |
       |   labeled_source_images_amazon.txt
       |   unlabeled_target_images_amazon_3.txt
       |   labeled_target_images_amazon_3.txt         
       |   unlabeled_source_images_webcam.txt
       |                  ...
      office_home---
                  |
                 ...       
```

### Example
#### Train
* DomainNet (clipart, painting, real, sketch)
```
python main.py --dataset multi --source real --target sketch --save_interval 5000 --steps 70000 --net resnet34 --num 3 --save_check
```
* Office-home (Art, Clipart, Product, Real)
* Office (amazon, dslr, webcam)

### Test
* DomainNet (clipart, painting, real, sketch)
```
python test.py --dataset multi --source real --target sketch --steps 70000
```
### checkpoint samples
* (DomainNet) Real to Sketch  [BaseNet](https://drive.google.com/file/d/1mwG1ClXzsyC3Pvq7WnlJfvtVwZdlQLxy/view?usp=sharing) / 
                              [Classifier](https://drive.google.com/file/d/1cO8YEaFWykRw7Pzw-xJcWx3ERioUBp_L/view?usp=sharing)
