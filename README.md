# TextBPN-Puls-Plus 
This is a Pytorch implementation of TextBPN++: [Arbitrary Shape Text Detection via Boundary Transformer](https://arxiv.org/abs/2205.05320);  This project is based on [TextBPN](https://github.com/GXYM/TextBPN)       
![](https://github.com/GXYM/TextBPN-Plus-Plus/blob/main/vis/framework.png)  

*Please light up the stars, thank you! Your encouragement is the energy for us to constantly update！*


## News
- [x] 2025.05.16 We are excited to announce the release of our **General Scene Super OCR Detection Model** ([**TextBPN-OCR**](https://github.com/GXYM/TextBPN-OCR))! Designed for robust performance in complex, multilingual environments, this model excels at detecting text of various shapes in any scene. Leveraging a powerful architecture based on DCN and ResNet50, it has been trained on 1.5 million synthetic images and over 50 million real-world scene samples. Despite its strong detection capabilities, the model remains lightweight and efficient, making it ideal for a wide range of applications.  
- [x] 2025.03.13 Added a script for multi-GPU training [train_textBPN_DDP.py](https://github.com/GXYM/TextBPN-Plus-Plus/blob/main/train_textBPN_DDP.py) 
- [x] 2023.06.14 Our paper of TextBPN++ has been accepted by IEEE Transactions on Multimedia (T-MM 2023), which can be obtained at [IEEE Xplore](https://ieeexplore.ieee.org/document/10153666).  
- [x] 2022.11.19 Displayed the visualization results of Reading Order Module.  
- [x] 2022.11.18 **Important update! Fixed some bugs in training code.**  These bugs make it difficult to reproduce the perfromance in the paper without pre-training. The updated code has more stable training and better performance. Thanks to the developers who made me aware of these bugs, and push us to fix these bugs. The description of the code change and the details of the reproduction are [here](https://github.com/GXYM/TextBPN-Plus-Plus/tree/main/output).  
- [x] 2022.10.18 The performance of our TextBPN++ on Document OCR have been proved by the paper ["Text Detection Forgot About Document OCR"](https://arxiv.org/pdf/2210.07903.pdf) with University of Oxford, which also reported the compared results for other text detection methods, such as PAN, DBNet, DBNet++, DCLNet, CRAFT etc. The comparison experiments is very detailed and very referential!
- [x] 2022.06.20 Updated the illustration of framework. 
- [x] 2022.06.16 Uploaded the missing file because of the naming problem（[CTW1500_Text_New](https://github.com/GXYM/TextBPN-Plus-Plus/blob/main/dataset/CTW1500_Text_New.py), [Total_Text_New](https://github.com/GXYM/TextBPN-Plus-Plus/blob/main/dataset/Total_Text_New.py)), which support the new label formats for Total_Text (mat) and CTW1500 (xml).  
- [x] 2022.06.16 We updated the Google cloud links so that it can be downloaded without permission.

## ToDo List

- [x] Release code
- [x] scripts for training and testing
- [x] Demo script
- [x] Evaluation
- [x] Release pre-trained models
- [x] Release trained models on each benchmarks
- [ ] The reading order module will be released at a certain time
- [ ] Prepare TextSpotter codes based on TextBPN++
<!-- - [ ] The number of control points is determined adaptively  -->



## Prerequisites 
  python >= 3.7 ;  
  PyTorch >= 1.7.0;   
  Numpy >=1.2.0;   
  CUDA >=11.1;  
  GCC >=10.0;   
  *opencv-python < 4.5.0*  
  NVIDIA GPU(2080 or 3090);  
  
  NOTE: We tested the code in the environment of Arch Linux+Python3.9 and  Ubuntu20.04+Python3.8. In different environments, the code needs to be adjusted slightly, and there is little difference in performance (not significant, so it can be ignored). Most of our experiments are mainly in Arch Linux + Python3.9.
  

## Makefile

If DCN is used, some CUDA files need to be compiled

```
  # make sure that the PATH of CUDA in setup.py is set properly with the right environment
  # Set the GPU you need in setup.py, if If you have different GPUs in your machine.
  
  cd network/backbone/assets/dcn
  sh Makefile.sh
  
  # setup.py 
  import os
  PATH ="{}:{}".format(os.environ['PATH'], "/opt/cuda/bin")
  # os.environ['CUDA_VISIBLE_DEVICES'] = "1"
  os.environ['PATH'] = PATH
  from setuptools import setup
  from torch.utils.cpp_extension import BuildExtension, CUDAExtension
```

## Dataset Links  
1. [CTW1500](https://drive.google.com/file/d/1A2s3FonXq4dHhD64A2NCWc8NQWMH2NFR/view?usp=sharing) （new version dataset at https://github.com/Yuliang-Liu/Curve-Text-Detector）
2. [TD500](https://drive.google.com/file/d/1ByluLnyd8-Ltjo9AC-1m7omZnI-FA1u0/view?usp=sharing)    
3. [Total-Text](https://drive.google.com/file/d/17_7T_-2Bu3KSSg2OkXeCxj97TBsjvueC/view?usp=sharing) （new version dataset at: https://github.com/cs-chan/Total-Text-Dataset）

NOTE: The images of each dataset can be obtained from their official website.


## Training 
### Prepar dataset
We provide a simple example for each dataset in data, such as [Total-Text](https://github.com/GXYM/TextBPN-Plus-Plus/tree/main/data/Total-Text), [CTW-1500](https://github.com/GXYM/TextBPN-Plus-Plus/tree/main/data/CTW-1500), and [ArT](https://github.com/GXYM/TextBPN-Plus-Plus/tree/main/data/ArT) ...


### Pre-training models
We provide some pre-tarining models on SynText [Baidu Drive](https://pan.baidu.com/s/1PsLrQWAVLDy0fSVp5HL9Ng) (download code: r1ff), [Google Drive](https://drive.google.com/file/d/1x64Lu_dMnQqs9gORQOMFL2wnXUzPWWGT/view?usp=sharing) and MLT-2017 [Baidu Drive](https://pan.baidu.com/s/19V9zqSMdgCHMvPTFngz8Fg) (download code: srym), [Google Drive](https://drive.google.com/file/d/1seVzFT657YzP-lc--yUqsVUek2mpw9tP/view?usp=sharing)

```
├── pretrain
│   ├──Syn
│       ├── TextBPN_resnet50_0.pth  #1s
│       ├── TextBPN_resnet18_0.pth  #4s
│       ├── TextBPN_deformable_resnet50_0.pth  #1s
│   ├── MLT
│       ├── TextBPN_resnet50_300.pth  #1s
│       ├── TextBPN_resnet18_300.pth  #4s
│       ├── TextBPN_deformable_resnet50_300.pth #1s
``` 
NOTE: we also provide the  pre-tarining scripts for [SynText](https://github.com/GXYM/TextBPN-Plus-Plus/tree/main/scripts-train).

### Runing the training scripts
We provide training scripts for each dataset in scripts-train, such as [Total-Text](https://github.com/GXYM/TextBPN-Plus-Plus/tree/main/scripts-train), [CTW-1500](https://github.com/GXYM/TextBPN-Plus-Plus/tree/main/scripts-train), and [ArT](https://github.com/GXYM/TextBPN-Plus-Plus/tree/main/scripts-train) ...

```
# train_Totaltext_res50_1s.sh
#!/bin/bash
cd ../
CUDA_LAUNCH_BLOCKING=1 python3 train_textBPN.py --exp_name Totaltext --net resnet50 --scale 1 --max_epoch 660 --batch_size 12 --gpu 0 --input_size 640 --optim Adam --lr 0.001 --num_workers 30

# train_Totaltext_res50_1s_fine_mlt.sh
#!/bin/bash
cd ../
CUDA_LAUNCH_BLOCKING=1 python3 train_textBPN.py --exp_name Totaltext --net resnet50 --scale 1 --max_epoch 660 --batch_size 12 --gpu 0 --input_size 640 --optim Adam --lr 0.0001 --num_workers 30 --load_memory True --resume model/pretrain/MLT/TextBPN_resnet50_300.pth

```

## Testing 

### Runing the Testing scripts
We provide testing scripts for each dataset in scripts-eval, such as [Total-Text](https://github.com/GXYM/TextBPN-Plus-Plus/blob/main/scripts-eval/Eval_Totaltext.sh), [CTW-1500](https://github.com/GXYM/TextBPN-Plus-Plus/blob/main/scripts-eval/Eval_CTW1500.sh), and [ArT](https://github.com/GXYM/TextBPN-Plus-Plus/blob/main/scripts-eval/Eval_ArT.sh) ...

```
# Eval_ArT.sh
#!/bin/bash
cd ../
##################### eval for ArT with ResNet50 1s ###################################
#CUDA_LAUNCH_BLOCKING=1 python3 eval_textBPN.py --net resnet50 --scale 1 --exp_name ArT --checkepoch 605 --test_size 960 2880 --dis_threshold 0.4 --cls_threshold 0.4 --gpu 0;


##################### eval for ArT with ResNet50-DCN 1s ###################################
CUDA_LAUNCH_BLOCKING=1 python3 eval_textBPN.py --net deformable_resnet50 --scale 1 --exp_name ArT --checkepoch 480 --test_size 960 2880 --dis_threshold 0.4 --cls_threshold 0.8 --gpu 0;


##################### batch eval for ArT ###################################
#for ((i=660; i>=300; i=i-5));
#do 
#CUDA_LAUNCH_BLOCKING=1 python3 eval_textBPN.py --exp_name ArT --net deformable_resnet50 --checkepoch $i --test_size 960 2880 --dis_threshold 0.45 --cls_threshold 0.8 --gpu 0;
#done
```

NOTE：If you want to save the visualization results, you need to open “--viz”.  Here is an example:

``` 
CUDA_LAUNCH_BLOCKING=1 python3 eval_textBPN.py --net resnet18 --scale 4 --exp_name TD500 --checkepoch 1135 --test_size 640 960 --dis_threshold 0.35 --cls_threshold 0.9 --gpu 0 --viz;
```


### Demo
You can also run prediction on your own dataset without annotations. Here is an example:

``` 
#demo.sh
#!/bin/bash
CUDA_LAUNCH_BLOCKING=1 python3 demo.py --net resnet18 --scale 4 --exp_name TD500 --checkepoch 1135 --test_size 640 960 --dis_threshold 0.35 --cls_threshold 0.9 --gpu 0 --viz --img_root /path/to/image 
```

### Evaluate the performance

Note that we provide some the protocols for benchmarks ([Total-Text](https://github.com/GXYM/TextBPN-Plus-Plus/tree/main/dataset/total_text), [CTW-1500](https://github.com/GXYM/TextBPN-Plus-Plus/tree/main/dataset/ctw1500), [MSRA-TD500](https://github.com/GXYM/TextBPN-Plus-Plus/tree/main/dataset/TD500), [ICDAR2015](https://github.com/GXYM/TextBPN-Plus-Plus/tree/main/dataset/icdar15)). The embedded evaluation protocol in the code are obtatined from the official protocols. You don't need to run these protocols alone, because our test code will automatically call these scripts, please refer to "[util/eval.py](https://github.com/GXYM/TextBPN-Plus-Plus/blob/main/util/eval.py)"


### Evaluate the speed 
 We refer to the speed testing scheme of DB. The speed is evaluated by performing a testing image for 50 times to exclude extra IO time.

```
CUDA_LAUNCH_BLOCKING=1 python3 eval_textBPN_speed.py --net resnet18 --scale 4 --exp_name TD500 --checkepoch 1135 --test_size 640 960 --dis_threshold 0.35 --cls_threshold 0.9 --gpu 1;
```

Note that the speed is related to both to the GPU and the CPU.

### Model performance  

Our replicated performance (without extra data for pre-training) by using the updated codes:
|  Datasets   |  pre-training |  Model        |recall | precision |F-measure| FPS |
|:----------:	|:-----------:  |:-----------:	|:-------:|:-----:|:-------:|:---:|
| Total-Text 	|       -       | [Res50-1s](https://drive.google.com/file/d/1wZOXGrM74CbrGY_lo4yF907VpbwnLf9d/view?usp=sharing) 	    | 84.22 |91.88 |87.88 |13|
| CTW-1500  	|       -       | [Res50-1s](https://drive.google.com/file/d/1wZOXGrM74CbrGY_lo4yF907VpbwnLf9d/view?usp=sharing)	    | 82.69 |86.53 |84.57 |14|


The results are reported in our paper as follows:
|  Datasets   |  Model        |recall | precision |F-measure| FPS |
|:----------:	|:-----------:	|:-------:|:-----:|:-------:|:---:|
| Total-Text 	| [Res18-4s](https://drive.google.com/file/d/11AtAA429JCha8AZLrp3xVURYZOcCC2s1/view?usp=sharing) 	    |  81.90 |89.88 |85.70 |**32.5**|
| Total-Text 	| [Res50-1s](https://drive.google.com/file/d/11AtAA429JCha8AZLrp3xVURYZOcCC2s1/view?usp=sharing)	    |  85.34 |91.81 |88.46 |13.3|
| Total-Text 	| [Res50-1s+DCN](https://drive.google.com/file/d/11AtAA429JCha8AZLrp3xVURYZOcCC2s1/view?usp=sharing)	|  **87.93** |**92.44** |**90.13** |13.2|
| CTW-1500 	  | [Res18-4s](https://drive.google.com/file/d/1pEaY7eU7esq5p_ZKcAfDA-fNqQskroUH/view?usp=sharing) 	    |  81.62 |87.55 |84.48 |**35.3**|
| CTW-1500  	| [Res50-1s](https://drive.google.com/file/d/1pEaY7eU7esq5p_ZKcAfDA-fNqQskroUH/view?usp=sharing)	    |  83.77 |87.30 |85.50 |14.1|
| CTW-1500  	| [Res50-1s+DCN](https://drive.google.com/file/d/1pEaY7eU7esq5p_ZKcAfDA-fNqQskroUH/view?usp=sharing)	|  **84.71** |**88.34** |**86.49** |16.5|
| MSRA-TD500 	| [Res18-4s](https://drive.google.com/file/d/1zNdqcXwplor3T6yQuw1f2euYZex63K0b/view?usp=sharing) 	    |  87.46 |92.38 |89.85 |**38.5**|
| MSRA-TD500  | [Res50-1s](https://drive.google.com/file/d/1zNdqcXwplor3T6yQuw1f2euYZex63K0b/view?usp=sharing)	    |  85.40 |89.23 |87.27 |15.2|
| MSRA-TD500  | [Res50-1s+DCN](https://drive.google.com/file/d/1zNdqcXwplor3T6yQuw1f2euYZex63K0b/view?usp=sharing)	|  **86.77** |**93.69** |**90.10** |15.3|
| MLT-2017	  | [Res50-1s](https://drive.google.com/file/d/12_umKAo-eM0NYbAAuJBABlkIbyr5QNCy/view?usp=sharing) 	    |  65.67 |80.49 |72.33 |  - |
| MLT-2017    | [Res50-1s+DCN](https://drive.google.com/file/d/12_umKAo-eM0NYbAAuJBABlkIbyr5QNCy/view?usp=sharing)	|  **72.10** |**83.74** |**77.48** |  - |
| ICDAR-ArT   | [Res50-1s](https://drive.google.com/file/d/1rn8_YKI9B0JO52uA_4nL4TQE0jjgqm9i/view?usp=sharing)	    |  71.07 |81.14 |75.77 |  - |
| ICDAR-ArT   | [Res50-1s+DCN](https://drive.google.com/file/d/1rn8_YKI9B0JO52uA_4nL4TQE0jjgqm9i/view?usp=sharing)	|  **77.05** |**84.48** |**80.59** |  - |

NOTE: The results on ICDAR-ArT and MLT-2017 are aslo can be found on the official competition website ([ICDAR-ArT](https://rrc.cvc.uab.es/?ch=14&com=evaluation&task=1) and [MLT-2017](https://rrc.cvc.uab.es/?ch=8&com=evaluation&task=1)). You can also download the models for each benchmarks
from [Baidu Drive](https://pan.baidu.com/s/1lFgRE_qiAv8ww8pbRTTh6w) (download code: wct1)

### Visual comparison

![](https://github.com/GXYM/TextBPN-Plus-Plus/blob/main/vis/vis.png)

Qualitative comparisons with [TextRay](https://github.com/LianaWang/TextRay), [ABCNet](https://github.com/aim-uofa/AdelaiDet), and [FCENet](https://github.com/open-mmlab/mmocr) on selected challenging samples from CTW-1500. The images (a)-(d) are borrowed from [FCENet](https://arxiv.org/abs/2104.10442).


## Reading Order Model

![](https://github.com/GXYM/TextBPN-Plus-Plus/blob/main/vis/order.png)

**Using the Reading Order Model to predict four key corners on the boundary proposal in turn. The reading order module will be released at a certain time.**


## Citing the related works

Please cite the related works in your publications if it helps your research:
``` 
  @inproceedings{DBLP:conf/iccv/Zhang0YWY21,
  author    = {Shi{-}Xue Zhang and
               Xiaobin Zhu and
               Chun Yang and
               Hongfa Wang and
               Xu{-}Cheng Yin},
  title     = {Adaptive Boundary Proposal Network for Arbitrary Shape Text Detection},
  booktitle = {2021 {IEEE/CVF} International Conference on Computer Vision, {ICCV} 2021, Montreal, QC, Canada, October 10-17, 2021},
  pages     = {1285--1294},
  publisher = {{IEEE}},
  year      = {2021},
}

@article{zhang2023arbitrary,
  title={Arbitrary shape text detection via boundary transformer},
  author={Zhang, Shi-Xue and Yang, Chun and Zhu, Xiaobin and Yin, Xu-Cheng},
  journal={IEEE Transactions on Multimedia},
  year={2023},
  publisher={IEEE}
}

@article{DBLP:journals/pami/ZhangZCHY23,
  author       = {Shi{-}Xue Zhang and
                  Xiaobin Zhu and
                  Lei Chen and
                  Jie{-}Bo Hou and
                  Xu{-}Cheng Yin},
  title        = {Arbitrary Shape Text Detection via Segmentation With Probability Maps},
  journal      = {{IEEE} Trans. Pattern Anal. Mach. Intell.},
  volume       = {45},
  number       = {3},
  pages        = {2736--2750},
  year         = {2023},
  url          = {https://doi.org/10.1109/TPAMI.2022.3176122},
  doi          = {10.1109/TPAMI.2022.3176122},
}

@article{zhang2022kernel,
  title={Kernel proposal network for arbitrary shape text detection},
  author={Zhang, Shi-Xue and Zhu, Xiaobin and Hou, Jie-Bo and Yang, Chun and Yin, Xu-Cheng},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2022},
  publisher={IEEE}
}

@inproceedings{DBLP:conf/cvpr/ZhangZHLYWY20,
  author       = {Shi{-}Xue Zhang and
                  Xiaobin Zhu and
                  Jie{-}Bo Hou and
                  Chang Liu and
                  Chun Yang and
                  Hongfa Wang and
                  Xu{-}Cheng Yin},
  title        = {Deep Relational Reasoning Graph Network for Arbitrary Shape Text Detection},
  booktitle    = {2020 {IEEE/CVF} Conference on Computer Vision and Pattern Recognition,
                  {CVPR} 2020, Seattle, WA, USA, June 13-19, 2020},
  pages        = {9696--9705},
  publisher    = {Computer Vision Foundation / {IEEE}},
  year         = {2020},
  doi          = {10.1109/CVPR42600.2020.00972},
}
  ``` 
 
 ## License
This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/GXYM/DRRG/blob/master/LICENSE.md) file for details  

<!---->
## ✨ Star History
[![Star History](https://api.star-history.com/svg?repos=GXYM/TextBPN-Plus-Plus&type=Date)](https://star-history.com/#GXYM/TextBPN-Plus-Plus&Date)


