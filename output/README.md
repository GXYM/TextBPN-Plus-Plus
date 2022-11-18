
## Code Update Description
**Solving the difficulties of reproducing the performance in your paper**

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
to 
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


## Recurrence details

* model：Res50-1s
* train dataset：CTW1500 trainset(1000 imgs) or Total-Text trainset(1225 imgs)，**without extra data**  
* dataset versuion: old version([CTW1500](https://github.com/GXYM/TextBPN-Plus-Plus/blob/main/dataset/ctw1500_text.py), [Total-Text](https://github.com/GXYM/TextBPN-Plus-Plus/blob/main/dataset/Total_Text.py))  
* train script：[train_CTW1500_res50_1s.sh](https://github.com/GXYM/TextBPN-Plus-Plus/blob/main/scripts-train/train_CTW1500_res50_1s.sh), [train_Totaltext_res50_1s.sh](https://github.com/GXYM/TextBPN-Plus-Plus/blob/main/scripts-train/train_Totaltext_res50_1s.sh)  

**Our replicated performance (without extra data for pre-training) by using the updated codes: **   

|  Datasets   |  pre-training |  Model        |recall | precision |F-measure| FPS |  
|:----------:	|:-----------:  |:-----------:	|:-------:|:-----:|:-------:|:---:|  
| Total-Text 	|       -       | [Res50-1s](https://drive.google.com/file/d/11AtAA429JCha8AZLrp3xVURYZOcCC2s1/view?usp=sharing) 	    | 84.22 |91.88 |87.88 |13|  
| CTW-1500  	|       -       | [Res50-1s](https://drive.google.com/file/d/11AtAA429JCha8AZLrp3xVURYZOcCC2s1/view?usp=sharing)	    |  82.69 |86.53 |88.57 |14|  







**Note：Save the quantitative results here**
