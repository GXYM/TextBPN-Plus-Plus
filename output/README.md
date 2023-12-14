
## Code Update Description
**Solving the difficulties of reproducing the performance in our paper**

**Reproduction records of some followers** in [here](https://github.com/GXYM/TextBPN-Plus-Plus/blob/main/output/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20230626221956.jpg) 

## Code change


* 1.change the code in loss.py (Great impact; When there is a lot of noise in the data annotation, the balanced cross entropy with OHEM will lead to unstable training, which is also the main reason for difficult reproduction)
```
 # pixel class loss
cls_loss = self.cls_ohem(fy_preds[:, 0, :, :], tr_mask.float(), train_mask)

```
to 
```
# pixel class loss
# cls_loss = self.cls_ohem(fy_preds[:, 0, :, :], tr_mask.float(), train_mask)
cls_loss = self.BCE_loss(fy_preds[:, 0, :, :],  tr_mask.float())
cls_loss = torch.mul(cls_loss, train_mask.float()).mean()       
```


* 2.change the code in loss.py (Minor impact; The parameter theta is missing in the old version)
```
if eps is None:
  alpha = 1.0; beta = 3.0; gama = 0.05
else:
  alpha = 1.0; beta = 3.0; gama = 0.1*torch.sigmoid(torch.tensor((eps - cfg.max_epoch)/cfg.max_epoch))
loss = alpha*cls_loss + beta*dis_loss + norm_loss + angle_loss + gama*(point_loss + energy_loss)
loss_dict = {
            'total_loss': loss,
            'cls_loss': alpha*cls_loss,
            'distance loss': beta*dis_loss,
            'dir_loss': norm_loss + angle_loss,
            'norm_loss': norm_loss,
            'angle_loss': angle_loss,
            'point_loss': gama*point_loss,
            'energy_loss': gama*energy_loss,}     
```
to 
```
if eps is None:
  alpha = 1.0; beta = 3.0; theta=0.5; gama = 0.05
else:
  alpha = 1.0; beta = 3.0; theta=0.5; 
  gama = 0.1*torch.sigmoid(torch.tensor((eps - cfg.max_epoch)/cfg.max_epoch))
loss = alpha*cls_loss + beta*dis_loss + theta*(norm_loss + angle_loss) + gama*(point_loss + energy_loss)
loss_dict = {
            'total_loss': loss,
            'cls_loss': alpha*cls_loss,
            'distance loss': beta*dis_loss,
            'dir_loss': theta*(norm_loss + angle_loss),
            'norm_loss': theta*norm_loss,
            'angle_loss': theta*angle_loss,
            'point_loss': gama*point_loss,
            'energy_loss': gama*energy_loss,}      
```

* 3.change the code in dataload.py (Uncertain impact; These parameters are used to generate the boundary proposals in training and add random noise)

```
def __init__(self, transform, is_training=False):
        super().__init__()
        self.transform = transform
        self.is_training = is_training
        self.min_text_size = 4
        self.jitter = 0.7
        self.th_b = 0.4
```
to 
```
def __init__(self, transform, is_training=False):
        super().__init__()
        self.transform = transform
        self.is_training = is_training
        self.min_text_size = 4
        self.jitter = 0.65
        self.th_b = 0.35
```



* 4.change the code in augmentation.py (Uncertain impact; The range of angle augmentation is 60 not 30; add the RandomDistortion)

```
class Augmentation(object):
    def __init__(self, size, mean, std):
        self.size = size
        self.mean = mean
        self.std = std
        self.augmentation = Compose([
            RandomCropFlip(),
            RandomResizeScale(size=self.size, ratio=(3. / 8, 5. / 2)),
            RandomResizedCrop(),
            RotatePadding(up=30, colors=True),  # pretrain on Syn is "up=15", else is "up=30"
            ResizeLimitSquare(size=self.size),
            # # RandomBrightness(),
            # # RandomContrast(),
            RandomMirror(),
            Normalize(mean=self.mean, std=self.std),
        ])
```
to 
```
class Augmentation(object):
    def __init__(self, size, mean, std):
        self.size = size
        self.mean = mean
        self.std = std
        self._transform_dict = {'brightness': 0.5, 'contrast': 0.5, 'sharpness': 0.8386, 'color': 0.5}
        self.augmentation = Compose([
            RandomCropFlip(),
            RandomResizeScale(size=self.size, ratio=(3. / 8, 5. / 2)),
            RandomResizedCrop(),
            RotatePadding(up=60, colors=True),  # pretrain on Syn is "up=15", else is "up=30"
            ResizeLimitSquare(size=self.size),
            RandomMirror(),
            RandomDistortion(self._transform_dict),
            Normalize(mean=self.mean, std=self.std),
        ])
```

## Recurrence details

* model：Res50-1s
* train dataset：CTW1500 trainset(1000 imgs) or Total-Text trainset(1225 imgs)，**without extra data**  
* dataset versuion: old version([CTW1500](https://github.com/GXYM/TextBPN-Plus-Plus/blob/main/dataset/ctw1500_text.py), [Total-Text](https://github.com/GXYM/TextBPN-Plus-Plus/blob/main/dataset/Total_Text.py))  
* train script：[train_CTW1500_res50_1s.sh](https://github.com/GXYM/TextBPN-Plus-Plus/blob/main/scripts-train/train_CTW1500_res50_1s.sh), [train_Totaltext_res50_1s.sh](https://github.com/GXYM/TextBPN-Plus-Plus/blob/main/scripts-train/train_Totaltext_res50_1s.sh)  

**Our replicated performance (without extra data for pre-training) by using the updated codes:**   

|  Datasets   |  pre-training |  Model        |recall | precision |F-measure| FPS |  
|:----------:	|:-----------:  |:-----------:	|:-------:|:-----:|:-------:|:---:|  
| Total-Text 	|       -       | [Res50-1s](https://drive.google.com/file/d/1wZOXGrM74CbrGY_lo4yF907VpbwnLf9d/view?usp=sharing) 	    | 84.22 |91.88 |87.88 |13|  
| CTW-1500  	|       -       | [Res50-1s](https://drive.google.com/file/d/1wZOXGrM74CbrGY_lo4yF907VpbwnLf9d/view?usp=sharing)	      | 82.69 |86.53 |84.57 |14|  


## Test Log

**1.** [CTW1500](https://github.com/GXYM/TextBPN-Plus-Plus/blob/main/output/Analysis/ctw1500_eval.txt)  
```
640_1024_100 0.35/0.825 ALL :: AP=0.6632 - Precision=0.8515 - Recall=0.7722 - Fscore=0.8099
640_1024_105 0.35/0.825 ALL :: AP=0.6544 - Precision=0.8395 - Recall=0.7774 - Fscore=0.8072
640_1024_110 0.35/0.825 ALL :: AP=0.6883 - Precision=0.8492 - Recall=0.8018 - Fscore=0.8248
640_1024_115 0.35/0.825 ALL :: AP=0.6657 - Precision=0.8621 - Recall=0.7663 - Fscore=0.8114
640_1024_120 0.35/0.825 ALL :: AP=0.6580 - Precision=0.8451 - Recall=0.7699 - Fscore=0.8057
640_1024_125 0.35/0.825 ALL :: AP=0.6577 - Precision=0.8537 - Recall=0.7683 - Fscore=0.8087
640_1024_130 0.35/0.825 ALL :: AP=0.6809 - Precision=0.8626 - Recall=0.7839 - Fscore=0.8214
640_1024_135 0.35/0.825 ALL :: AP=0.6669 - Precision=0.8518 - Recall=0.7754 - Fscore=0.8118
640_1024_140 0.35/0.825 ALL :: AP=0.6896 - Precision=0.8660 - Recall=0.7920 - Fscore=0.8274
640_1024_145 0.35/0.825 ALL :: AP=0.6684 - Precision=0.8572 - Recall=0.7751 - Fscore=0.8141
640_1024_150 0.35/0.825 ALL :: AP=0.6791 - Precision=0.8726 - Recall=0.7705 - Fscore=0.8184
640_1024_155 0.35/0.825 ALL :: AP=0.6903 - Precision=0.8451 - Recall=0.8093 - Fscore=0.8268
640_1024_160 0.35/0.825 ALL :: AP=0.6880 - Precision=0.8580 - Recall=0.7937 - Fscore=0.8246
640_1024_165 0.35/0.825 ALL :: AP=0.6772 - Precision=0.8463 - Recall=0.7953 - Fscore=0.8200
640_1024_170 0.35/0.825 ALL :: AP=0.6857 - Precision=0.8497 - Recall=0.8018 - Fscore=0.8251
640_1024_175 0.35/0.825 ALL :: AP=0.6968 - Precision=0.8787 - Recall=0.7885 - Fscore=0.8311
640_1024_180 0.35/0.825 ALL :: AP=0.6832 - Precision=0.8473 - Recall=0.7995 - Fscore=0.8227
640_1024_185 0.35/0.825 ALL :: AP=0.6979 - Precision=0.8508 - Recall=0.8123 - Fscore=0.8311
640_1024_190 0.35/0.825 ALL :: AP=0.6934 - Precision=0.8686 - Recall=0.7927 - Fscore=0.8289
640_1024_195 0.35/0.825 ALL :: AP=0.6944 - Precision=0.8657 - Recall=0.7966 - Fscore=0.8297
640_1024_200 0.35/0.825 ALL :: AP=0.6974 - Precision=0.8494 - Recall=0.8165 - Fscore=0.8326
640_1024_205 0.35/0.825 ALL :: AP=0.6833 - Precision=0.8523 - Recall=0.7956 - Fscore=0.8230
640_1024_210 0.35/0.825 ALL :: AP=0.6821 - Precision=0.8675 - Recall=0.7790 - Fscore=0.8209
640_1024_215 0.35/0.825 ALL :: AP=0.7005 - Precision=0.8669 - Recall=0.7986 - Fscore=0.8314
640_1024_220 0.35/0.825 ALL :: AP=0.6879 - Precision=0.8670 - Recall=0.7859 - Fscore=0.8244
640_1024_225 0.35/0.825 ALL :: AP=0.6713 - Precision=0.8678 - Recall=0.7663 - Fscore=0.8139
640_1024_230 0.35/0.825 ALL :: AP=0.6994 - Precision=0.8444 - Recall=0.8175 - Fscore=0.8307
640_1024_235 0.35/0.825 ALL :: AP=0.7070 - Precision=0.8759 - Recall=0.8008 - Fscore=0.8367
640_1024_240 0.35/0.825 ALL :: AP=0.6847 - Precision=0.8722 - Recall=0.7787 - Fscore=0.8228
640_1024_245 0.35/0.825 ALL :: AP=0.6970 - Precision=0.8541 - Recall=0.8093 - Fscore=0.8311
640_1024_250 0.35/0.825 ALL :: AP=0.7015 - Precision=0.8534 - Recall=0.8119 - Fscore=0.8321
640_1024_255 0.35/0.825 ALL :: AP=0.6975 - Precision=0.8485 - Recall=0.8106 - Fscore=0.8291
640_1024_260 0.35/0.825 ALL :: AP=0.6754 - Precision=0.8572 - Recall=0.7787 - Fscore=0.8161
640_1024_265 0.35/0.825 ALL :: AP=0.6902 - Precision=0.8610 - Recall=0.7953 - Fscore=0.8268
640_1024_270 0.35/0.825 ALL :: AP=0.6844 - Precision=0.8678 - Recall=0.7829 - Fscore=0.8232
640_1024_275 0.35/0.825 ALL :: AP=0.7021 - Precision=0.8690 - Recall=0.8018 - Fscore=0.8340
640_1024_280 0.35/0.825 ALL :: AP=0.6991 - Precision=0.8442 - Recall=0.8198 - Fscore=0.8318
640_1024_285 0.35/0.825 ALL :: AP=0.7004 - Precision=0.8603 - Recall=0.8067 - Fscore=0.8326
640_1024_290 0.35/0.825 ALL :: AP=0.6953 - Precision=0.8449 - Recall=0.8129 - Fscore=0.8286
640_1024_295 0.35/0.825 ALL :: AP=0.7120 - Precision=0.8629 - Recall=0.8207 - Fscore=0.8413
640_1024_300 0.35/0.825 ALL :: AP=0.7091 - Precision=0.8422 - Recall=0.8315 - Fscore=0.8368
640_1024_305 0.35/0.825 ALL :: AP=0.7105 - Precision=0.8420 - Recall=0.8357 - Fscore=0.8389
640_1024_310 0.35/0.825 ALL :: AP=0.7002 - Precision=0.8469 - Recall=0.8220 - Fscore=0.8343
640_1024_315 0.35/0.825 ALL :: AP=0.6916 - Precision=0.8551 - Recall=0.8018 - Fscore=0.8276
640_1024_320 0.35/0.825 ALL :: AP=0.7021 - Precision=0.8529 - Recall=0.8142 - Fscore=0.8331
640_1024_325 0.35/0.825 ALL :: AP=0.7042 - Precision=0.8406 - Recall=0.8269 - Fscore=0.8337
640_1024_330 0.35/0.825 ALL :: AP=0.7052 - Precision=0.8578 - Recall=0.8139 - Fscore=0.8353
640_1024_335 0.35/0.825 ALL :: AP=0.7005 - Precision=0.8608 - Recall=0.8083 - Fscore=0.8338
640_1024_340 0.35/0.825 ALL :: AP=0.6979 - Precision=0.8683 - Recall=0.7976 - Fscore=0.8315
640_1024_345 0.35/0.825 ALL :: AP=0.6818 - Precision=0.8432 - Recall=0.7995 - Fscore=0.8208
640_1024_350 0.35/0.825 ALL :: AP=0.6933 - Precision=0.8465 - Recall=0.8090 - Fscore=0.8273
640_1024_355 0.35/0.825 ALL :: AP=0.6955 - Precision=0.8534 - Recall=0.8067 - Fscore=0.8294
640_1024_360 0.35/0.825 ALL :: AP=0.6930 - Precision=0.8762 - Recall=0.7819 - Fscore=0.8264
640_1024_365 0.35/0.825 ALL :: AP=0.6974 - Precision=0.8499 - Recall=0.8123 - Fscore=0.8307
640_1024_370 0.35/0.825 ALL :: AP=0.6947 - Precision=0.8456 - Recall=0.8139 - Fscore=0.8294
640_1024_375 0.35/0.825 ALL :: AP=0.6806 - Precision=0.8414 - Recall=0.7989 - Fscore=0.8196
640_1024_380 0.35/0.825 ALL :: AP=0.6957 - Precision=0.8538 - Recall=0.8070 - Fscore=0.8298
640_1024_385 0.35/0.825 ALL :: AP=0.6993 - Precision=0.8491 - Recall=0.8158 - Fscore=0.8321
640_1024_390 0.35/0.825 ALL :: AP=0.6997 - Precision=0.8530 - Recall=0.8113 - Fscore=0.8316
640_1024_395 0.35/0.825 ALL :: AP=0.6755 - Precision=0.8327 - Recall=0.8015 - Fscore=0.8168
640_1024_400 0.35/0.825 ALL :: AP=0.7079 - Precision=0.8689 - Recall=0.8057 - Fscore=0.8361
640_1024_405 0.35/0.825 ALL :: AP=0.6922 - Precision=0.8531 - Recall=0.8048 - Fscore=0.8282
640_1024_410 0.35/0.825 ALL :: AP=0.7096 - Precision=0.8579 - Recall=0.8204 - Fscore=0.8387
640_1024_415 0.35/0.825 ALL :: AP=0.7033 - Precision=0.8450 - Recall=0.8230 - Fscore=0.8339
640_1024_420 0.35/0.825 ALL :: AP=0.6921 - Precision=0.8343 - Recall=0.8220 - Fscore=0.8281
640_1024_425 0.35/0.825 ALL :: AP=0.6949 - Precision=0.8466 - Recall=0.8129 - Fscore=0.8294
640_1024_430 0.35/0.825 ALL :: AP=0.6924 - Precision=0.8455 - Recall=0.8100 - Fscore=0.8274
640_1024_435 0.35/0.825 ALL :: AP=0.6899 - Precision=0.8387 - Recall=0.8119 - Fscore=0.8251
640_1024_440 0.35/0.825 ALL :: AP=0.7132 - Precision=0.8779 - Recall=0.8061 - Fscore=0.8404
640_1024_445 0.35/0.825 ALL :: AP=0.6952 - Precision=0.8301 - Recall=0.8331 - Fscore=0.8316
640_1024_450 0.35/0.825 ALL :: AP=0.7022 - Precision=0.8593 - Recall=0.8083 - Fscore=0.8331
640_1024_455 0.35/0.825 ALL :: AP=0.6989 - Precision=0.8478 - Recall=0.8155 - Fscore=0.8314
640_1024_460 0.35/0.825 ALL :: AP=0.7011 - Precision=0.8488 - Recall=0.8181 - Fscore=0.8332
640_1024_465 0.35/0.825 ALL :: AP=0.7035 - Precision=0.8504 - Recall=0.8191 - Fscore=0.8345
640_1024_470 0.35/0.825 ALL :: AP=0.6974 - Precision=0.8583 - Recall=0.8057 - Fscore=0.8312
640_1024_475 0.35/0.825 ALL :: AP=0.7064 - Precision=0.8565 - Recall=0.8171 - Fscore=0.8364
640_1024_480 0.35/0.825 ALL :: AP=0.6938 - Precision=0.8571 - Recall=0.8015 - Fscore=0.8284
640_1024_485 0.35/0.825 ALL :: AP=0.6903 - Precision=0.8358 - Recall=0.8181 - Fscore=0.8269
640_1024_490 0.35/0.825 ALL :: AP=0.7216 - Precision=0.8653 - Recall=0.8269 - Fscore=0.8457
640_1024_495 0.35/0.825 ALL :: AP=0.6949 - Precision=0.8535 - Recall=0.8054 - Fscore=0.8288
640_1024_500 0.35/0.825 ALL :: AP=0.7022 - Precision=0.8380 - Recall=0.8295 - Fscore=0.8337
640_1024_505 0.35/0.825 ALL :: AP=0.7058 - Precision=0.8461 - Recall=0.8259 - Fscore=0.8359
640_1024_510 0.35/0.825 ALL :: AP=0.7020 - Precision=0.8359 - Recall=0.8302 - Fscore=0.8330
640_1024_515 0.35/0.825 ALL :: AP=0.7108 - Precision=0.8612 - Recall=0.8188 - Fscore=0.8394
640_1024_520 0.35/0.825 ALL :: AP=0.6952 - Precision=0.8337 - Recall=0.8250 - Fscore=0.8293
640_1024_525 0.35/0.825 ALL :: AP=0.6513 - Precision=0.7806 - Recall=0.8256 - Fscore=0.8025
640_1024_530 0.35/0.825 ALL :: AP=0.7150 - Precision=0.8620 - Recall=0.8227 - Fscore=0.8419
640_1024_535 0.35/0.825 ALL :: AP=0.7025 - Precision=0.8563 - Recall=0.8136 - Fscore=0.8344
640_1024_540 0.35/0.825 ALL :: AP=0.6968 - Precision=0.8436 - Recall=0.8207 - Fscore=0.8320
640_1024_545 0.35/0.825 ALL :: AP=0.7020 - Precision=0.8393 - Recall=0.8289 - Fscore=0.8340
640_1024_550 0.35/0.825 ALL :: AP=0.7073 - Precision=0.8600 - Recall=0.8152 - Fscore=0.8370
640_1024_555 0.35/0.825 ALL :: AP=0.7098 - Precision=0.8350 - Recall=0.8413 - Fscore=0.8381
640_1024_560 0.35/0.825 ALL :: AP=0.6873 - Precision=0.8209 - Recall=0.8279 - Fscore=0.8244
640_1024_565 0.35/0.825 ALL :: AP=0.7031 - Precision=0.8477 - Recall=0.8237 - Fscore=0.8355
640_1024_570 0.35/0.825 ALL :: AP=0.7034 - Precision=0.8444 - Recall=0.8240 - Fscore=0.8340
640_1024_575 0.35/0.825 ALL :: AP=0.6792 - Precision=0.8153 - Recall=0.8217 - Fscore=0.8185
640_1024_580 0.35/0.825 ALL :: AP=0.6882 - Precision=0.8304 - Recall=0.8217 - Fscore=0.8260
640_1024_585 0.35/0.825 ALL :: AP=0.6805 - Precision=0.8316 - Recall=0.8110 - Fscore=0.8211
640_1024_590 0.35/0.825 ALL :: AP=0.7039 - Precision=0.8354 - Recall=0.8321 - Fscore=0.8338
640_1024_595 0.35/0.825 ALL :: AP=0.7045 - Precision=0.8492 - Recall=0.8204 - Fscore=0.8345
640_1024_600 0.35/0.825 ALL :: AP=0.6956 - Precision=0.8333 - Recall=0.8259 - Fscore=0.8296
640_1024_605 0.35/0.825 ALL :: AP=0.7063 - Precision=0.8498 - Recall=0.8227 - Fscore=0.8360
640_1024_610 0.35/0.825 ALL :: AP=0.6922 - Precision=0.8296 - Recall=0.8250 - Fscore=0.8273
640_1024_615 0.35/0.825 ALL :: AP=0.6970 - Precision=0.8459 - Recall=0.8175 - Fscore=0.8314
640_1024_620 0.35/0.825 ALL :: AP=0.6971 - Precision=0.8452 - Recall=0.8152 - Fscore=0.8299
640_1024_625 0.35/0.825 ALL :: AP=0.7046 - Precision=0.8491 - Recall=0.8214 - Fscore=0.8350
640_1024_630 0.35/0.825 ALL :: AP=0.6983 - Precision=0.8487 - Recall=0.8139 - Fscore=0.8309
640_1024_635 0.35/0.825 ALL :: AP=0.6959 - Precision=0.8364 - Recall=0.8233 - Fscore=0.8298
640_1024_640 0.35/0.825 ALL :: AP=0.6992 - Precision=0.8524 - Recall=0.8129 - Fscore=0.8322
640_1024_645 0.35/0.825 ALL :: AP=0.6922 - Precision=0.8312 - Recall=0.8220 - Fscore=0.8266
640_1024_650 0.35/0.825 ALL :: AP=0.6848 - Precision=0.8387 - Recall=0.8087 - Fscore=0.8234
640_1024_655 0.35/0.825 ALL :: AP=0.6947 - Precision=0.8377 - Recall=0.8211 - Fscore=0.8293
640_1024_660 0.35/0.825 ALL :: AP=0.6893 - Precision=0.8338 - Recall=0.8175 - Fscore=0.8255
```

**2.** [Total-Text](https://github.com/GXYM/TextBPN-Plus-Plus/blob/main/output/Analysis/totalText_eval.txt)  
```
640_1024_55_0.7_0.6 0.3/0.85 ALL :: Precision=0.8919 - Recall=0.5764 - Fscore=0.7003
640_1024_55_0.8_0.4 0.3/0.85 ALL :: Precision=0.7440 - Recall=0.4755 - Fscore=0.5802
640_1024_60_0.7_0.6 0.3/0.85 ALL :: Precision=0.8882 - Recall=0.7482 - Fscore=0.8122
640_1024_60_0.8_0.4 0.3/0.85 ALL :: Precision=0.8648 - Recall=0.6999 - Fscore=0.7737
640_1024_65_0.7_0.6 0.3/0.85 ALL :: Precision=0.8836 - Recall=0.7422 - Fscore=0.8067
640_1024_65_0.8_0.4 0.3/0.85 ALL :: Precision=0.8580 - Recall=0.6985 - Fscore=0.7701
640_1024_70_0.7_0.6 0.3/0.85 ALL :: Precision=0.8578 - Recall=0.7809 - Fscore=0.8175
640_1024_70_0.8_0.4 0.3/0.85 ALL :: Precision=0.8510 - Recall=0.7440 - Fscore=0.7939
640_1024_75_0.7_0.6 0.3/0.85 ALL :: Precision=0.9005 - Recall=0.7093 - Fscore=0.7935
640_1024_75_0.8_0.4 0.3/0.85 ALL :: Precision=0.7720 - Recall=0.5996 - Fscore=0.6750
640_1024_80_0.7_0.6 0.3/0.85 ALL :: Precision=0.8779 - Recall=0.7393 - Fscore=0.8026
640_1024_80_0.8_0.4 0.3/0.85 ALL :: Precision=0.9035 - Recall=0.7136 - Fscore=0.7974
640_1024_85_0.7_0.6 0.3/0.85 ALL :: Precision=0.9066 - Recall=0.7356 - Fscore=0.8122
640_1024_85_0.8_0.4 0.3/0.85 ALL :: Precision=0.9056 - Recall=0.6985 - Fscore=0.7887
640_1024_90_0.7_0.6 0.3/0.85 ALL :: Precision=0.9147 - Recall=0.7342 - Fscore=0.8146
640_1024_90_0.8_0.4 0.3/0.85 ALL :: Precision=0.8714 - Recall=0.6778 - Fscore=0.7625
640_1024_95_0.7_0.6 0.3/0.85 ALL :: Precision=0.9105 - Recall=0.7541 - Fscore=0.8249
640_1024_95_0.8_0.4 0.3/0.85 ALL :: Precision=0.8974 - Recall=0.7189 - Fscore=0.7983
640_1024_100_0.7_0.6 0.3/0.85 ALL :: Precision=0.8707 - Recall=0.7873 - Fscore=0.8269
640_1024_100_0.8_0.4 0.3/0.85 ALL :: Precision=0.8760 - Recall=0.7581 - Fscore=0.8128
640_1024_105_0.7_0.6 0.3/0.85 ALL :: Precision=0.8976 - Recall=0.7715 - Fscore=0.8298
640_1024_105_0.8_0.4 0.3/0.85 ALL :: Precision=0.8745 - Recall=0.7299 - Fscore=0.7956
640_1024_110_0.7_0.6 0.3/0.85 ALL :: Precision=0.8657 - Recall=0.7544 - Fscore=0.8062
640_1024_110_0.8_0.4 0.3/0.85 ALL :: Precision=0.8370 - Recall=0.7028 - Fscore=0.7641
640_1024_115_0.7_0.6 0.3/0.85 ALL :: Precision=0.8907 - Recall=0.7803 - Fscore=0.8318
640_1024_115_0.8_0.4 0.3/0.85 ALL :: Precision=0.8775 - Recall=0.7494 - Fscore=0.8084
640_1024_120_0.7_0.6 0.3/0.85 ALL :: Precision=0.9044 - Recall=0.7874 - Fscore=0.8419
640_1024_120_0.8_0.4 0.3/0.85 ALL :: Precision=0.8854 - Recall=0.7429 - Fscore=0.8079
640_1024_125_0.7_0.6 0.3/0.85 ALL :: Precision=0.8920 - Recall=0.7837 - Fscore=0.8343
640_1024_125_0.8_0.4 0.3/0.85 ALL :: Precision=0.8616 - Recall=0.7341 - Fscore=0.7927
640_1024_130_0.7_0.6 0.3/0.85 ALL :: Precision=0.9021 - Recall=0.7599 - Fscore=0.8249
640_1024_130_0.8_0.4 0.3/0.85 ALL :: Precision=0.8815 - Recall=0.7221 - Fscore=0.7939
640_1024_135_0.7_0.6 0.3/0.85 ALL :: Precision=0.9095 - Recall=0.7417 - Fscore=0.8170
640_1024_135_0.8_0.4 0.3/0.85 ALL :: Precision=0.8822 - Recall=0.6976 - Fscore=0.7792
640_1024_140_0.7_0.6 0.3/0.85 ALL :: Precision=0.9052 - Recall=0.7681 - Fscore=0.8310
640_1024_140_0.8_0.4 0.3/0.85 ALL :: Precision=0.8372 - Recall=0.7023 - Fscore=0.7638
640_1024_145_0.7_0.6 0.3/0.85 ALL :: Precision=0.9037 - Recall=0.7492 - Fscore=0.8192
640_1024_145_0.8_0.4 0.3/0.85 ALL :: Precision=0.8498 - Recall=0.6900 - Fscore=0.7616
640_1024_150_0.7_0.6 0.3/0.85 ALL :: Precision=0.8820 - Recall=0.7822 - Fscore=0.8291
640_1024_150_0.8_0.4 0.3/0.85 ALL :: Precision=0.7898 - Recall=0.6825 - Fscore=0.7322
640_1024_155_0.7_0.6 0.3/0.85 ALL :: Precision=0.9289 - Recall=0.7593 - Fscore=0.8356
640_1024_155_0.8_0.4 0.3/0.85 ALL :: Precision=0.8968 - Recall=0.7114 - Fscore=0.7934
640_1024_160_0.7_0.6 0.3/0.85 ALL :: Precision=0.8956 - Recall=0.7883 - Fscore=0.8385
640_1024_160_0.8_0.4 0.3/0.85 ALL :: Precision=0.8661 - Recall=0.7422 - Fscore=0.7994
640_1024_165_0.7_0.6 0.3/0.85 ALL :: Precision=0.8919 - Recall=0.7734 - Fscore=0.8284
640_1024_165_0.8_0.4 0.3/0.85 ALL :: Precision=0.9038 - Recall=0.7515 - Fscore=0.8206
640_1024_170_0.7_0.6 0.3/0.85 ALL :: Precision=0.9014 - Recall=0.7495 - Fscore=0.8184
640_1024_170_0.8_0.4 0.3/0.85 ALL :: Precision=0.8178 - Recall=0.6698 - Fscore=0.7364
640_1024_175_0.7_0.6 0.3/0.85 ALL :: Precision=0.8883 - Recall=0.7802 - Fscore=0.8307
640_1024_175_0.8_0.4 0.3/0.85 ALL :: Precision=0.8175 - Recall=0.7062 - Fscore=0.7578
640_1024_180_0.7_0.6 0.3/0.85 ALL :: Precision=0.8944 - Recall=0.8206 - Fscore=0.8559
640_1024_180_0.8_0.4 0.3/0.85 ALL :: Precision=0.8799 - Recall=0.7856 - Fscore=0.8300
640_1024_185_0.7_0.6 0.3/0.85 ALL :: Precision=0.9049 - Recall=0.8038 - Fscore=0.8514
640_1024_185_0.8_0.4 0.3/0.85 ALL :: Precision=0.8634 - Recall=0.7460 - Fscore=0.8004
640_1024_190_0.7_0.6 0.3/0.85 ALL :: Precision=0.9025 - Recall=0.7893 - Fscore=0.8421
640_1024_190_0.8_0.4 0.3/0.85 ALL :: Precision=0.8476 - Recall=0.7310 - Fscore=0.7850
640_1024_195_0.7_0.6 0.3/0.85 ALL :: Precision=0.8633 - Recall=0.8346 - Fscore=0.8487
640_1024_195_0.8_0.4 0.3/0.85 ALL :: Precision=0.8211 - Recall=0.7784 - Fscore=0.7992
640_1024_200_0.7_0.6 0.3/0.85 ALL :: Precision=0.8903 - Recall=0.7979 - Fscore=0.8416
640_1024_200_0.8_0.4 0.3/0.85 ALL :: Precision=0.8883 - Recall=0.7704 - Fscore=0.8251
640_1024_205_0.7_0.6 0.3/0.85 ALL :: Precision=0.8790 - Recall=0.8166 - Fscore=0.8467
640_1024_205_0.8_0.4 0.3/0.85 ALL :: Precision=0.8131 - Recall=0.7435 - Fscore=0.7767
640_1024_210_0.7_0.6 0.3/0.85 ALL :: Precision=0.8907 - Recall=0.7886 - Fscore=0.8365
640_1024_210_0.8_0.4 0.3/0.85 ALL :: Precision=0.8386 - Recall=0.7358 - Fscore=0.7839
640_1024_215_0.7_0.6 0.3/0.85 ALL :: Precision=0.9042 - Recall=0.8074 - Fscore=0.8530
640_1024_215_0.8_0.4 0.3/0.85 ALL :: Precision=0.8894 - Recall=0.7689 - Fscore=0.8247
640_1024_220_0.7_0.6 0.3/0.85 ALL :: Precision=0.9024 - Recall=0.8124 - Fscore=0.8551
640_1024_220_0.8_0.4 0.3/0.85 ALL :: Precision=0.8894 - Recall=0.7789 - Fscore=0.8305
640_1024_225_0.7_0.6 0.3/0.85 ALL :: Precision=0.9243 - Recall=0.7753 - Fscore=0.8433
640_1024_225_0.8_0.4 0.3/0.85 ALL :: Precision=0.8966 - Recall=0.7331 - Fscore=0.8067
640_1024_230_0.7_0.6 0.3/0.85 ALL :: Precision=0.9038 - Recall=0.7887 - Fscore=0.8423
640_1024_230_0.8_0.4 0.3/0.85 ALL :: Precision=0.8479 - Recall=0.7240 - Fscore=0.7811
640_1024_235_0.7_0.6 0.3/0.85 ALL :: Precision=0.8893 - Recall=0.8155 - Fscore=0.8508
640_1024_235_0.8_0.4 0.3/0.85 ALL :: Precision=0.8738 - Recall=0.7748 - Fscore=0.8213
640_1024_240_0.7_0.6 0.3/0.85 ALL :: Precision=0.9134 - Recall=0.7917 - Fscore=0.8482
640_1024_240_0.8_0.4 0.3/0.85 ALL :: Precision=0.8867 - Recall=0.7480 - Fscore=0.8115
640_1024_245_0.7_0.6 0.3/0.85 ALL :: Precision=0.8786 - Recall=0.8240 - Fscore=0.8505
640_1024_245_0.8_0.4 0.3/0.85 ALL :: Precision=0.8876 - Recall=0.8023 - Fscore=0.8428
640_1024_250_0.7_0.6 0.3/0.85 ALL :: Precision=0.9029 - Recall=0.8116 - Fscore=0.8548
640_1024_250_0.8_0.4 0.3/0.85 ALL :: Precision=0.8781 - Recall=0.7730 - Fscore=0.8222
640_1024_255_0.7_0.6 0.3/0.85 ALL :: Precision=0.8993 - Recall=0.8044 - Fscore=0.8492
640_1024_255_0.8_0.4 0.3/0.85 ALL :: Precision=0.8817 - Recall=0.7730 - Fscore=0.8237
640_1024_260_0.7_0.6 0.3/0.85 ALL :: Precision=0.9177 - Recall=0.7869 - Fscore=0.8473
640_1024_260_0.8_0.4 0.3/0.85 ALL :: Precision=0.8965 - Recall=0.7519 - Fscore=0.8179
640_1024_265_0.7_0.6 0.3/0.85 ALL :: Precision=0.9016 - Recall=0.7657 - Fscore=0.8281
640_1024_265_0.8_0.4 0.3/0.85 ALL :: Precision=0.8101 - Recall=0.6797 - Fscore=0.7392
640_1024_270_0.7_0.6 0.3/0.85 ALL :: Precision=0.9182 - Recall=0.7625 - Fscore=0.8332
640_1024_270_0.8_0.4 0.3/0.85 ALL :: Precision=0.8800 - Recall=0.7181 - Fscore=0.7908
640_1024_275_0.7_0.6 0.3/0.85 ALL :: Precision=0.8808 - Recall=0.8362 - Fscore=0.8579
640_1024_275_0.8_0.4 0.3/0.85 ALL :: Precision=0.8450 - Recall=0.7865 - Fscore=0.8147
640_1024_280_0.7_0.6 0.3/0.85 ALL :: Precision=0.8764 - Recall=0.8334 - Fscore=0.8544
640_1024_280_0.8_0.4 0.3/0.85 ALL :: Precision=0.8273 - Recall=0.7779 - Fscore=0.8018
640_1024_285_0.7_0.6 0.3/0.85 ALL :: Precision=0.9112 - Recall=0.7925 - Fscore=0.8477
640_1024_285_0.8_0.4 0.3/0.85 ALL :: Precision=0.8581 - Recall=0.7381 - Fscore=0.7936
640_1024_290_0.7_0.6 0.3/0.85 ALL :: Precision=0.9156 - Recall=0.8065 - Fscore=0.8576
640_1024_290_0.8_0.4 0.3/0.85 ALL :: Precision=0.8930 - Recall=0.7752 - Fscore=0.8299
640_1024_295_0.7_0.6 0.3/0.85 ALL :: Precision=0.8806 - Recall=0.8238 - Fscore=0.8512
640_1024_295_0.8_0.4 0.3/0.85 ALL :: Precision=0.8351 - Recall=0.7685 - Fscore=0.8004
640_1024_300_0.7_0.6 0.3/0.85 ALL :: Precision=0.9062 - Recall=0.8168 - Fscore=0.8592
640_1024_300_0.8_0.4 0.3/0.85 ALL :: Precision=0.8715 - Recall=0.7657 - Fscore=0.8152
640_1024_305_0.7_0.6 0.3/0.85 ALL :: Precision=0.8940 - Recall=0.8250 - Fscore=0.8581
640_1024_305_0.8_0.4 0.3/0.85 ALL :: Precision=0.8665 - Recall=0.7877 - Fscore=0.8252
640_1024_310_0.7_0.6 0.3/0.85 ALL :: Precision=0.9030 - Recall=0.8124 - Fscore=0.8553
640_1024_310_0.8_0.4 0.3/0.85 ALL :: Precision=0.8219 - Recall=0.7302 - Fscore=0.7734
640_1024_315_0.7_0.6 0.3/0.85 ALL :: Precision=0.8854 - Recall=0.8052 - Fscore=0.8434
640_1024_315_0.8_0.4 0.3/0.85 ALL :: Precision=0.8243 - Recall=0.7396 - Fscore=0.7797
640_1024_320_0.7_0.6 0.3/0.85 ALL :: Precision=0.9020 - Recall=0.8080 - Fscore=0.8524
640_1024_320_0.8_0.4 0.3/0.85 ALL :: Precision=0.8380 - Recall=0.7368 - Fscore=0.7842
640_1024_325_0.7_0.6 0.3/0.85 ALL :: Precision=0.9207 - Recall=0.7591 - Fscore=0.8321
640_1024_325_0.8_0.4 0.3/0.85 ALL :: Precision=0.8780 - Recall=0.7137 - Fscore=0.7874
640_1024_330_0.7_0.6 0.3/0.85 ALL :: Precision=0.9092 - Recall=0.8017 - Fscore=0.8521
640_1024_330_0.8_0.4 0.3/0.85 ALL :: Precision=0.8613 - Recall=0.7506 - Fscore=0.8022
640_1024_335_0.7_0.6 0.3/0.85 ALL :: Precision=0.9124 - Recall=0.7884 - Fscore=0.8459
640_1024_335_0.8_0.4 0.3/0.85 ALL :: Precision=0.8746 - Recall=0.7389 - Fscore=0.8010
640_1024_340_0.7_0.6 0.3/0.85 ALL :: Precision=0.8851 - Recall=0.8261 - Fscore=0.8546
640_1024_340_0.8_0.4 0.3/0.85 ALL :: Precision=0.8666 - Recall=0.7851 - Fscore=0.8238
640_1024_345_0.7_0.6 0.3/0.85 ALL :: Precision=0.8868 - Recall=0.8367 - Fscore=0.8610
640_1024_345_0.8_0.4 0.3/0.85 ALL :: Precision=0.8489 - Recall=0.7883 - Fscore=0.8175
640_1024_350_0.7_0.6 0.3/0.85 ALL :: Precision=0.8902 - Recall=0.8297 - Fscore=0.8589
640_1024_350_0.8_0.4 0.3/0.85 ALL :: Precision=0.8778 - Recall=0.8043 - Fscore=0.8394
640_1024_355_0.7_0.6 0.3/0.85 ALL :: Precision=0.9133 - Recall=0.8091 - Fscore=0.8580
640_1024_355_0.8_0.4 0.3/0.85 ALL :: Precision=0.8882 - Recall=0.7691 - Fscore=0.8244
640_1024_360_0.7_0.6 0.3/0.85 ALL :: Precision=0.9170 - Recall=0.8006 - Fscore=0.8549
640_1024_360_0.8_0.4 0.3/0.85 ALL :: Precision=0.8782 - Recall=0.7532 - Fscore=0.8109
640_1024_365_0.7_0.6 0.3/0.85 ALL :: Precision=0.9107 - Recall=0.8159 - Fscore=0.8607
640_1024_365_0.8_0.4 0.3/0.85 ALL :: Precision=0.8958 - Recall=0.7883 - Fscore=0.8386
640_1024_370_0.7_0.6 0.3/0.85 ALL :: Precision=0.9239 - Recall=0.8065 - Fscore=0.8612
640_1024_370_0.8_0.4 0.3/0.85 ALL :: Precision=0.8984 - Recall=0.7693 - Fscore=0.8289
640_1024_375_0.7_0.6 0.3/0.85 ALL :: Precision=0.9133 - Recall=0.8207 - Fscore=0.8645
640_1024_375_0.8_0.4 0.3/0.85 ALL :: Precision=0.8985 - Recall=0.7885 - Fscore=0.8399
640_1024_380_0.7_0.6 0.3/0.85 ALL :: Precision=0.9233 - Recall=0.7982 - Fscore=0.8562
640_1024_380_0.8_0.4 0.3/0.85 ALL :: Precision=0.8988 - Recall=0.7631 - Fscore=0.8254
640_1024_385_0.7_0.6 0.3/0.85 ALL :: Precision=0.9085 - Recall=0.8270 - Fscore=0.8659
640_1024_385_0.8_0.4 0.3/0.85 ALL :: Precision=0.8804 - Recall=0.7867 - Fscore=0.8309
640_1024_390_0.7_0.6 0.3/0.85 ALL :: Precision=0.8920 - Recall=0.8430 - Fscore=0.8668
640_1024_390_0.8_0.4 0.3/0.85 ALL :: Precision=0.8527 - Recall=0.7916 - Fscore=0.8210
640_1024_395_0.7_0.6 0.3/0.85 ALL :: Precision=0.9081 - Recall=0.8270 - Fscore=0.8657
640_1024_395_0.8_0.4 0.3/0.85 ALL :: Precision=0.8728 - Recall=0.7782 - Fscore=0.8228
640_1024_400_0.7_0.6 0.3/0.85 ALL :: Precision=0.9075 - Recall=0.8230 - Fscore=0.8632
640_1024_400_0.8_0.4 0.3/0.85 ALL :: Precision=0.8488 - Recall=0.7623 - Fscore=0.8032
640_1024_405_0.7_0.6 0.3/0.85 ALL :: Precision=0.8951 - Recall=0.8366 - Fscore=0.8648
640_1024_405_0.8_0.4 0.3/0.85 ALL :: Precision=0.8294 - Recall=0.7659 - Fscore=0.7964
640_1024_410_0.7_0.6 0.3/0.85 ALL :: Precision=0.9087 - Recall=0.8168 - Fscore=0.8603
640_1024_410_0.8_0.4 0.3/0.85 ALL :: Precision=0.8503 - Recall=0.7520 - Fscore=0.7981
640_1024_415_0.7_0.6 0.3/0.85 ALL :: Precision=0.8821 - Recall=0.8457 - Fscore=0.8635
640_1024_415_0.8_0.4 0.3/0.85 ALL :: Precision=0.8498 - Recall=0.8046 - Fscore=0.8266
640_1024_420_0.7_0.6 0.3/0.85 ALL :: Precision=0.9138 - Recall=0.8333 - Fscore=0.8717
640_1024_420_0.8_0.4 0.3/0.85 ALL :: Precision=0.8890 - Recall=0.7927 - Fscore=0.8381
640_1024_425_0.7_0.6 0.3/0.85 ALL :: Precision=0.9073 - Recall=0.8247 - Fscore=0.8640
640_1024_425_0.8_0.4 0.3/0.85 ALL :: Precision=0.8860 - Recall=0.7914 - Fscore=0.8360
640_1024_430_0.7_0.6 0.3/0.85 ALL :: Precision=0.9150 - Recall=0.8278 - Fscore=0.8692
640_1024_430_0.8_0.4 0.3/0.85 ALL :: Precision=0.8908 - Recall=0.7936 - Fscore=0.8394
640_1024_435_0.7_0.6 0.3/0.85 ALL :: Precision=0.9052 - Recall=0.8284 - Fscore=0.8651
640_1024_435_0.8_0.4 0.3/0.85 ALL :: Precision=0.8501 - Recall=0.7658 - Fscore=0.8057
640_1024_440_0.7_0.6 0.3/0.85 ALL :: Precision=0.9019 - Recall=0.8144 - Fscore=0.8559
640_1024_440_0.8_0.4 0.3/0.85 ALL :: Precision=0.8591 - Recall=0.7656 - Fscore=0.8097
640_1024_445_0.7_0.6 0.3/0.85 ALL :: Precision=0.9054 - Recall=0.8309 - Fscore=0.8665
640_1024_445_0.8_0.4 0.3/0.85 ALL :: Precision=0.8720 - Recall=0.7837 - Fscore=0.8255
640_1024_450_0.7_0.6 0.3/0.85 ALL :: Precision=0.9026 - Recall=0.8360 - Fscore=0.8680
640_1024_450_0.8_0.4 0.3/0.85 ALL :: Precision=0.8756 - Recall=0.7980 - Fscore=0.8350
640_1024_455_0.7_0.6 0.3/0.85 ALL :: Precision=0.9142 - Recall=0.8088 - Fscore=0.8583
640_1024_455_0.8_0.4 0.3/0.85 ALL :: Precision=0.8801 - Recall=0.7687 - Fscore=0.8206
640_1024_460_0.7_0.6 0.3/0.85 ALL :: Precision=0.9003 - Recall=0.8290 - Fscore=0.8632
640_1024_460_0.8_0.4 0.3/0.85 ALL :: Precision=0.8814 - Recall=0.7963 - Fscore=0.8367
640_1024_465_0.7_0.6 0.3/0.85 ALL :: Precision=0.9083 - Recall=0.8284 - Fscore=0.8665
640_1024_465_0.8_0.4 0.3/0.85 ALL :: Precision=0.8463 - Recall=0.7624 - Fscore=0.8022
640_1024_470_0.7_0.6 0.3/0.85 ALL :: Precision=0.9052 - Recall=0.8264 - Fscore=0.8640
640_1024_470_0.8_0.4 0.3/0.85 ALL :: Precision=0.8668 - Recall=0.7785 - Fscore=0.8203
640_1024_475_0.7_0.6 0.3/0.85 ALL :: Precision=0.9228 - Recall=0.8064 - Fscore=0.8607
640_1024_475_0.8_0.4 0.3/0.85 ALL :: Precision=0.9016 - Recall=0.7737 - Fscore=0.8328
640_1024_480_0.7_0.6 0.3/0.85 ALL :: Precision=0.9117 - Recall=0.8129 - Fscore=0.8594
640_1024_480_0.8_0.4 0.3/0.85 ALL :: Precision=0.9108 - Recall=0.7998 - Fscore=0.8517
640_1024_485_0.7_0.6 0.3/0.85 ALL :: Precision=0.8764 - Recall=0.8402 - Fscore=0.8579
640_1024_485_0.8_0.4 0.3/0.85 ALL :: Precision=0.8354 - Recall=0.7776 - Fscore=0.8055
640_1024_490_0.7_0.6 0.3/0.85 ALL :: Precision=0.9033 - Recall=0.8417 - Fscore=0.8714
640_1024_490_0.8_0.4 0.3/0.85 ALL :: Precision=0.8577 - Recall=0.7890 - Fscore=0.8219
640_1024_495_0.7_0.6 0.3/0.85 ALL :: Precision=0.9119 - Recall=0.8284 - Fscore=0.8682
640_1024_495_0.8_0.4 0.3/0.85 ALL :: Precision=0.8496 - Recall=0.7620 - Fscore=0.8034
640_1024_500_0.7_0.6 0.3/0.85 ALL :: Precision=0.9022 - Recall=0.8422 - Fscore=0.8712
640_1024_500_0.8_0.4 0.3/0.85 ALL :: Precision=0.8714 - Recall=0.7998 - Fscore=0.8341
640_1024_505_0.7_0.6 0.3/0.85 ALL :: Precision=0.9168 - Recall=0.8249 - Fscore=0.8684
640_1024_505_0.8_0.4 0.3/0.85 ALL :: Precision=0.8742 - Recall=0.7757 - Fscore=0.8220
640_1024_510_0.7_0.6 0.3/0.85 ALL :: Precision=0.8980 - Recall=0.8305 - Fscore=0.8629
640_1024_510_0.8_0.4 0.3/0.85 ALL :: Precision=0.8457 - Recall=0.7719 - Fscore=0.8071
640_1024_515_0.7_0.6 0.3/0.85 ALL :: Precision=0.9138 - Recall=0.8304 - Fscore=0.8701
640_1024_515_0.8_0.4 0.3/0.85 ALL :: Precision=0.8791 - Recall=0.7894 - Fscore=0.8318
640_1024_520_0.7_0.6 0.3/0.85 ALL :: Precision=0.9174 - Recall=0.8219 - Fscore=0.8670
640_1024_520_0.8_0.4 0.3/0.85 ALL :: Precision=0.8750 - Recall=0.7736 - Fscore=0.8212
640_1024_525_0.7_0.6 0.3/0.85 ALL :: Precision=0.9173 - Recall=0.8246 - Fscore=0.8685
640_1024_525_0.8_0.4 0.3/0.85 ALL :: Precision=0.8903 - Recall=0.7902 - Fscore=0.8373
640_1024_530_0.7_0.6 0.3/0.85 ALL :: Precision=0.9003 - Recall=0.8358 - Fscore=0.8668
640_1024_530_0.8_0.4 0.3/0.85 ALL :: Precision=0.8657 - Recall=0.7912 - Fscore=0.8268
640_1024_535_0.7_0.6 0.3/0.85 ALL :: Precision=0.9101 - Recall=0.8303 - Fscore=0.8684
640_1024_535_0.8_0.4 0.3/0.85 ALL :: Precision=0.8425 - Recall=0.7601 - Fscore=0.7991
640_1024_540_0.7_0.6 0.3/0.85 ALL :: Precision=0.9070 - Recall=0.8270 - Fscore=0.8651
640_1024_540_0.8_0.4 0.3/0.85 ALL :: Precision=0.8880 - Recall=0.7922 - Fscore=0.8374
640_1024_545_0.7_0.6 0.3/0.85 ALL :: Precision=0.8862 - Recall=0.8309 - Fscore=0.8577
640_1024_545_0.8_0.4 0.3/0.85 ALL :: Precision=0.8246 - Recall=0.7681 - Fscore=0.7953
640_1024_550_0.7_0.6 0.3/0.85 ALL :: Precision=0.9061 - Recall=0.8276 - Fscore=0.8651
640_1024_550_0.8_0.4 0.3/0.85 ALL :: Precision=0.8729 - Recall=0.7858 - Fscore=0.8271
640_1024_555_0.7_0.6 0.3/0.85 ALL :: Precision=0.9012 - Recall=0.8410 - Fscore=0.8701
640_1024_555_0.8_0.4 0.3/0.85 ALL :: Precision=0.8597 - Recall=0.7856 - Fscore=0.8210
640_1024_560_0.7_0.6 0.3/0.85 ALL :: Precision=0.8993 - Recall=0.8335 - Fscore=0.8651
640_1024_560_0.8_0.4 0.3/0.85 ALL :: Precision=0.8532 - Recall=0.7789 - Fscore=0.8144
640_1024_565_0.7_0.6 0.3/0.85 ALL :: Precision=0.9188 - Recall=0.8422 - Fscore=0.8788
640_1024_565_0.8_0.4 0.3/0.85 ALL :: Precision=0.8820 - Recall=0.7945 - Fscore=0.8359
640_1024_570_0.7_0.6 0.3/0.85 ALL :: Precision=0.9077 - Recall=0.8271 - Fscore=0.8656
640_1024_570_0.8_0.4 0.3/0.85 ALL :: Precision=0.8659 - Recall=0.7768 - Fscore=0.8189
640_1024_575_0.7_0.6 0.3/0.85 ALL :: Precision=0.9108 - Recall=0.8214 - Fscore=0.8638
640_1024_575_0.8_0.4 0.3/0.85 ALL :: Precision=0.8447 - Recall=0.7512 - Fscore=0.7952
640_1024_580_0.7_0.6 0.3/0.85 ALL :: Precision=0.9136 - Recall=0.8044 - Fscore=0.8555
640_1024_580_0.8_0.4 0.3/0.85 ALL :: Precision=0.8763 - Recall=0.7612 - Fscore=0.8147
640_1024_585_0.7_0.6 0.3/0.85 ALL :: Precision=0.9097 - Recall=0.8334 - Fscore=0.8699
640_1024_585_0.8_0.4 0.3/0.85 ALL :: Precision=0.8672 - Recall=0.7802 - Fscore=0.8214
640_1024_590_0.7_0.6 0.3/0.85 ALL :: Precision=0.9086 - Recall=0.8271 - Fscore=0.8659
640_1024_590_0.8_0.4 0.3/0.85 ALL :: Precision=0.8544 - Recall=0.7687 - Fscore=0.8093
640_1024_595_0.7_0.6 0.3/0.85 ALL :: Precision=0.9148 - Recall=0.8355 - Fscore=0.8734
640_1024_595_0.8_0.4 0.3/0.85 ALL :: Precision=0.8840 - Recall=0.7904 - Fscore=0.8346
640_1024_600_0.7_0.6 0.3/0.85 ALL :: Precision=0.9086 - Recall=0.8419 - Fscore=0.8740
640_1024_600_0.8_0.4 0.3/0.85 ALL :: Precision=0.8787 - Recall=0.7994 - Fscore=0.8372
640_1024_605_0.7_0.6 0.3/0.85 ALL :: Precision=0.8997 - Recall=0.8450 - Fscore=0.8715
640_1024_605_0.8_0.4 0.3/0.85 ALL :: Precision=0.8769 - Recall=0.8091 - Fscore=0.8416
640_1024_610_0.7_0.6 0.3/0.85 ALL :: Precision=0.9126 - Recall=0.8148 - Fscore=0.8609
640_1024_610_0.8_0.4 0.3/0.85 ALL :: Precision=0.8784 - Recall=0.7695 - Fscore=0.8204
640_1024_615_0.7_0.6 0.3/0.85 ALL :: Precision=0.9046 - Recall=0.8327 - Fscore=0.8671
640_1024_615_0.8_0.4 0.3/0.85 ALL :: Precision=0.8632 - Recall=0.7802 - Fscore=0.8196
640_1024_620_0.7_0.6 0.3/0.85 ALL :: Precision=0.9005 - Recall=0.8495 - Fscore=0.8743
640_1024_620_0.8_0.4 0.3/0.85 ALL :: Precision=0.8641 - Recall=0.8016 - Fscore=0.8317
640_1024_625_0.7_0.6 0.3/0.85 ALL :: Precision=0.9048 - Recall=0.8413 - Fscore=0.8719
640_1024_625_0.8_0.4 0.3/0.85 ALL :: Precision=0.8578 - Recall=0.7865 - Fscore=0.8206
640_1024_630_0.7_0.6 0.3/0.85 ALL :: Precision=0.9182 - Recall=0.8181 - Fscore=0.8653
640_1024_630_0.8_0.4 0.3/0.85 ALL :: Precision=0.8621 - Recall=0.7583 - Fscore=0.8069
640_1024_635_0.7_0.6 0.3/0.85 ALL :: Precision=0.9102 - Recall=0.8383 - Fscore=0.8728
640_1024_635_0.8_0.4 0.3/0.85 ALL :: Precision=0.8753 - Recall=0.7914 - Fscore=0.8312
640_1024_640_0.7_0.6 0.3/0.85 ALL :: Precision=0.9160 - Recall=0.8320 - Fscore=0.8720
640_1024_640_0.8_0.4 0.3/0.85 ALL :: Precision=0.8804 - Recall=0.7851 - Fscore=0.8300
640_1024_645_0.7_0.6 0.3/0.85 ALL :: Precision=0.9050 - Recall=0.8472 - Fscore=0.8751
640_1024_645_0.8_0.4 0.3/0.85 ALL :: Precision=0.8799 - Recall=0.8103 - Fscore=0.8437
640_1024_650_0.7_0.6 0.3/0.85 ALL :: Precision=0.9122 - Recall=0.8394 - Fscore=0.8743
640_1024_650_0.8_0.4 0.3/0.85 ALL :: Precision=0.8630 - Recall=0.7830 - Fscore=0.8211
640_1024_655_0.7_0.6 0.3/0.85 ALL :: Precision=0.9052 - Recall=0.8440 - Fscore=0.8735
640_1024_655_0.8_0.4 0.3/0.85 ALL :: Precision=0.8671 - Recall=0.7947 - Fscore=0.8294
640_1024_660_0.7_0.6 0.3/0.85 ALL :: Precision=0.9090 - Recall=0.8440 - Fscore=0.8753
640_1024_660_0.8_0.4 0.3/0.85 ALL :: Precision=0.8612 - Recall=0.7893 - Fscore=0.8237
```





##  Note：Save the quantitative results in output
