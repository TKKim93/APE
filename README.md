# Attract, Perturb, and Explore: Learning a Feature Alignment Network for Semi-supervised Domain Adaptation


### Dataset Structure
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

### Train
#### DomainNet 
```
python main.py --method S+T --dataset multi --source real --target sketch --early --save_interval 5000 --steps 70000 --net resnet34 --session multi_r2s --thr 0.5 --num 3
```
