# Attract, Perturb, and Explore: Learning a Feature Alignment Network for Semi-supervised Domain Adaptation

### Acknowledgment

The implementation is built on the pytorch implementation of [SSDA_MME](https://github.com/jwyang/faster-rcnn.pytorch) and we refer a specific module in [DTA].(https://github.com/postBG/DTA.pytorch)

### Prerequisites

* CUDA 10.0 or 10.1
* Python 3.6
* Pytorch 1.0.1
```
conda install pytorch==1.0.1 torchvision==0.2.2 cudatoolkit=10.0 -c pytorch
```
* Other dependencies
```
pip install -r requirements.txt
```

### Dataset Structure
You can download the datasets by following the instructions in [SSDA_MME](https://github.com/jwyang/faster-rcnn.pytorch).
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
     |        ...
   office---
     |    |
     |   ...
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
* DomainNet (real, clipart, painting, sketch)
```
python main.py --method S+T --dataset multi --source real --target sketch --early --save_interval 5000 --steps 70000 --net resnet34 --session multi_r2s --thr 0.5 --num 3
```
* Office-home (Real, Clipart, Product, Art)
* Office (amazon, dslr, webcam)

### checkpoint samples
* (DomainNet) Real to Sketch  [BaseNet](https://drive.google.com/file/d/1mwG1ClXzsyC3Pvq7WnlJfvtVwZdlQLxy/view?usp=sharing) / 
                              [Classifier](https://drive.google.com/file/d/1cO8YEaFWykRw7Pzw-xJcWx3ERioUBp_L/view?usp=sharing)
