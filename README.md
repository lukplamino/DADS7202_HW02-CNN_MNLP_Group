# DADS7202_Group Assignment 2 CNN (Group_MNLP)
> Objective: **`What do you use to train an image classiffier with our custom image dataset?`**
## ✨Highlight
1.	Based on pre-trained model, the highest average accuracy rate is 85.37% of Xception model
2.	The best model after fine tuning pre-trained model is VGG16. The average accuracy on test data is at 91.66% rising from 84.63% (no tune)
3.	InceptionV3 and VGG16 show notable improvement on test accuracy at +8.71% and +7.04%, respectively, while ResNetV2 and Xception have less improvement at +2.59% and +2.22%, respectively
4.	Overall, the models have learned a distinctive of each banana type from banana fruit shape and shape of banana cut stalk


<!-- เป็นข้อคิดเห็น การค้นพบ ข้อสรุปหรือข้อมูล insight น่าสนใจที่  3-5 bullets -->

## Table of Contents

 - [1. Introduction🎯](https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group/blob/main/README.md#1-introduction)
 - [2. Data📑](https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group#2-data)
 - [3. Network architecture📦](https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group#3-network-architecture)
 - [4. Training🔮](https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group#4-training)
 - [5. Results📈](https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group#5-results)
 - [6. Discussion💭](https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group/blob/main/README.md#6-discussion)
 - [7. Conclusion📝](https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group/blob/main/README.md#7-conclusion)
 - [8. References🌐](https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group/blob/main/README.md#8-references)
 - [Citing](https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group/blob/main/README.md#citing)
 - [👥 Members, Percent Contribution and Responsibility](https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group/blob/main/README.md#-members-percent-contribution-and-responsibility)
 - [🖇️End Credit ](https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group/blob/main/README.md#%EF%B8%8Fend-credit)



## 1. Introduction🎯 

**`multi-class image classification`**:

- This project aims to test **4 CNN pre-training models** (`VGG16`, `ResNet50V2`, `Xception`, `InceptionV3`) on the ImageNet dataset and fine-tune it to classify 3 types of bananas 🍌 (`Cultivated banana`, `Lady finger banana`, `Cavendish banana`) which is our custom image dataset that were never trained on. 
- Then, we will compare performance of **4 CNN pre-training models** without transfer learning and with transfer learning (Fine-tuning).
- Finally, we use [**`Grad-CAM`**](https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group/blob/main/README.md#-visualizing-what-cnn-learned-with-grad-cam4) technique to debug the model and gain more insight into what a trained CNN did.

 
## 2. Data📑
There are many banana<sup>0</sup> varieties in Thailand and each one of them has different characteristics. Let’s find out interesting facts about different varieties of banana before training and finetuning models.

 🍌 **1. Cultivated banana - กล้วยน้ำว้า**
  
  The fruit looks a little bit angled with a thick skin and a sweet flavor.
  
  <img src="https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group/blob/main/Images/Cultivated%20banana.png" style="width:120px;"/>
 
 🍌 **2. Lady finger banana - กล้วยเล็บมือนาง**
  
  Lady finger banana is one of the smallest bananas. Its skin is thick with a soft flesh inside.
  
   <img src="https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group/blob/main/Images/Lady%20finger%20banana.png" style="width:120px;"/>
  
   
 🍌 **3. Cavendish banana - กล้วยหอม**
  
  The fruit is long with a thin skin. It offers a sweet flavor along with a uniquely pleasant smell.
  
  <img src="https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group/blob/main/Images/Cavendish%20banana.jpg" style="width:120px;"/>

- Total 600 images (200 images per 1 class)

| Class | Name               | No. of image |
|:---:|---|---:|
|`0`      | Lady Finger Banana | 200          |
|`1`      | CavendishBanana    | 200          |
|`2`      | Cultivated Banana  | 200          |
|       | **Total**             | **600**          |

 
#### 📍Data source: 
- We use [**Download All Images**](https://chrome.google.com/webstore/detail/download-all-images/ifipmflagepipjokmbdecpmjbibjnakm?hl=en) extension in chrome web store to collect set of images by searching keywords (3 types of banana) from **`google image searching`** 


#### 🧹Data preparation:
- Collecting set of images from the Internet source is a quick and simple method to gather a set of images. Some facts, meanwhile, are not entirely accurate or useful. As a result, we have to manually remove several unnecessary images from the collection, such as banana dessert, banana tree trunk, other banana pieces, and duplicate images. Additionally, because the keyword and banana type are inconsistent, we need to recheck the banana type labels.

#### Data pre-processing
- The set of images were rescaled to 224x224 pixels and normalized by the ImageDataGenerator class with **`Pixel Standardization`** technique (zero mean and unit variance)

❕ Be careful: The different normalization techniques<sup>5</sup> effect the model's performance. (Loss and Accuracy).
- We use **`➕Data Augmentation`** technique to increase the diversity of our training set by applying **`ImageDataGenerator`** 
- We apply various changes to the initial data. For example, image rotation, zoom, shifting and fliping

```
        width_shift_range = 5.0,   
        height_shift_range = 5.0,              
        rotation_range=10,                               
        zoom_range=0.2,
        horizontal_flip=True,                 
        vertical_flip=True,
        validation_split=0.3
```
<img src="https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group/blob/main/Images/Data_Augmentation.png" style="width:500px;"/>


#### ✂️Data splitting (train/val/test):
- `random_state` = 3
- `test_size` = 0.3
- `validation_split` = 0.3
- **`Train set`**: 294
- **`Validation set`**: 126
- **`Test set`**: 180

[🔝](https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group/blob/main/README.md#highlight)

## 3. Network architecture📦

### Pre-training Models 
In this experiment, we have selected 4 Pre-training Models for fine-tuning with [**IMAGENET**](https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/)<sup>6</sup> dataset as weight and complied with 

#### [Pre-training model Infomation](https://keras.io/api/applications/)<sup>7</sup>

<img src="https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group/blob/main/Images/pre-training-models_info.png" style="width:700px;"/>

#### Network Architecture of Pre-training model 
- To compare Network architecture of Pre-training model **`without Fine-tuning`**  VS **`with Fine-tuning`**
- Remark: Based on our dataset and experiment scope
<img src="https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group/blob/main/Images/Network_Architechture.png" style="width:900px;"/>

#### Network Diagram of Pre-training model with Fine-tuning
<img src="https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group/blob/main/Images/Network_Diagram.png" style="width:700px;"/>

<img src="https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group/blob/main/Images/Network_Diagram_InceptionV3.png" style="width:700px;"/>

<!-- รายละเอียดต่าง ๆ ของโมเดลที่เลือกใช้ (เช่น จำนวนและตำแหน่งการวาง layer, จำนวน nodes, activation function, regularization) ในรูปแบบของ network diagram หรือตาราง (โดยใส่ข้อมูลให้ละเอียดพอที่คนที่มาอ่าน จะสามารถไปสร้าง network ตามเราได้) -->

[🔝](https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group/blob/main/README.md#highlight)


## 4. Training🔮
Our training strategy is...

<img src="https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group/blob/main/Images/training_part.png" style="width:650px;"/>

<!-- รายละเอียดของการ train และ validate ข้อมูล รวมถึงทรัพยากรที่ใช้ในการ train โมเดลหนึ่ง ๆ เช่น training strategy (เช่น single loss, compound loss, two-step training, end-to-end training), loss, optimizer (learning rate, momentum, etc), batch size,
epoch, รุ่นและจำนวน CPU หรือ GPU หรือ TPU ที่ใช้, เวลาโดยประมาณที่ใช้ train โมเดลหนึ่งตัว ฯลฯ -->


[🔝](https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group/blob/main/README.md#highlight)

## 5. Results📈
<!-- แสดงตัวเลขผลลัพธ์ในรูปของค่าเฉลี่ย mean±SD โดยให้ทำการเทรนโมเดลด้วย initial random weights ที่แตกต่างกันอย่างน้อย 3-5 รอบเพื่อให้ได้อย่างน้อย 3-5 โมเดลมาหาประสิทธิภาพเฉลี่ยกัน, แสดงผลลัพธ์การ train โมเดลเป็นกราฟเทียบ train vs. validation, สรุปผลว่าเกิด underfit หรือ overfit หรือไม่, อธิบาย evaluation metric ที่ใช้ในการประเมินประสิทธิภาพของโมเดลบน train/val/test sets ตามความเหมาะสมของปัญหา, หากสามารถเปรียบเทียบผลลัพธ์ของโมเดลเรากับโมเดลอื่น ๆ (ของคนอื่น) บน any standard benchmark dataset ได้ด้วยจะยิ่งทำให้งานดูน่าเชื่อถือยิ่งขึ้น เช่น เทียบความแม่นยำ เทียบเวลาที่ใช้train เทียบเวลาที่ใช้ inference บนซีพียูและจีพียู เทียบขนาดโมเดล ฯลฯ 

การแสดงผลลัพธ์เทียบ train vs. validation (เช่น loss, accuracy) ถ้าเป็นไปได้ควรแสดงไว้ในกราฟเดียวกันเพื่อให้สามารถเทียบ scale ค่าผลลัพธ์และดู underfit / overfit ได้ง่าย-->

### 📊 Model Performance Comparison
From the experiment, We fine-tune each pre-training model with Hyperparameter more than 40-50 times to find the best model's performance with highest accuracy, less loss and not over-fit.

We pre-train the model with initial random weights in the first round and more 2 rounds without random seed to calculate mean±SD of accuracy and loss on test set as the average of the model performance
In each round, accuracy and loss of test sets are not significantly different. That proves the model is good fit.

<img src="https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group/blob/main/Images/Performance1.png" style="width:500px;"/>



#### Accuracy and Loss on Train vs Validation sets

<img src="https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group/blob/main/Images/Acc_Loss.png" style="width:650px;"/>

#### Accuracy on Test set
- To compare the highest accuracy on test set of each Pre-training models under the same conditions and the same seeds,
<img src="https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group/blob/main/Images/Performance2.png" style="width:500px;"/>

### 🪟 Evaluation metric on Test set

<img src="https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group/blob/main/Images/Evaluation-Matrix.png" style="width:600px;"/>

### ⌛ Runtime Comparison (on Train set) 
Time per inference step is the average of epoch.
- **`GPU`**: Tesla T4
- **`Epoch`**: 30
<img src="https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group/blob/main/Images/Runtime.png" style="width:550px;"/>

- **`InceptionV3`** pre-training model with fine-tuning spended the shortest time per epoch to train the model on test set.
- On the contrary, **`Xception`** pre-training model with fine-tuning spended the shortest time per epoch among these 4 models.



### 🔦 Visualizing what CNN learned with `Grad-Cam`<sup>4</sup>
- We use the gradient-weighted class activation mapping (**`Grad-Cam`**) technique on VGG16 pre-training model with fine-tuning to understand which parts of the image are most important for classification. 
> _"The Grad-CAM technique utilizes the gradients of the classification score with respect to the final convolutional feature map, to identify the parts of an input image that most impact the classification score. The places where this gradient is large are exactly the places where the final score depends most on the data."<sup> 9</sup>_

<img src="https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group/blob/main/Images/GRAD_CAM.png" style="width:650px;"/>

- For **`Cavendish`** bananas (No.4,5) and **`Cultivated`** bananas (No.7,8), Clearly, the center of the banana fruit has the greatest impact on the classification.
- Contrarily, for all images that the model predicted **`Lady Finger`** bananas (No.1,2,6), the model focused on the cut stalk of the banana for distinctive features.
- Moreover, we can use the Grad-CAM heat-maps to give us clues into why the model had trouble making the correct classification. 
- Look at the pictures (no.3,9) with wrong prediction, we can see from the Grad-CAM that the model is emphasizing background or male flowers and having trouble finding banana fruit features.


[🔝](https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group/blob/main/README.md#highlight)


## 6. Discussion💭
-	With ImageDataGenerator(), apart from shift, zoom, rotation and flip, we have tried rescaling with variety methods. Firstly, rescale = 1./255, it shows the worth accuracy rate at about 0.3611 as it just resize pictures by dividing 255. Next, featurewise_center and featurewise_std_normalization which set dataset mean to 0 and normalized over dataset; as a result, accuracy spike to 0.6278. Lastly, samplewise_center and samplewise_std_normalization is similar to the previous method but done over each input. All in all, we chose samplewise_center and samplewise_std_normalization as it shows the higher accuracy rate at 0.7778
-	In the beginning of work, we tried load data with flow_from_directory() that labels of picture will be one-hot encoded i.e. [1 0 0 0] for label number 1; consequently, a loss function must be “CategoricalCrossentropy”. However, we decided to load data with NumPy that labeled data from [0-2] i.e [1] for label number 1. As a result, “SparseCategoricalCrossentropy” is chosen to be the loss function
-	We experience running out of free-access GPU on GoogleColab. So, we try using other accounts, using local GPU on our own computers, and registered for ColabPro since CNN have noticeable difference run-time on CPU and GPU. Execution with CPU roughly takes 40 times of time consuming over GPU (200s vs 5s per epochs)
-	Among learning rate of 0.01, 0.001, 0.0001 and 0.00001, learning rate at 0.001 shows the best accuracy over test data set 
Note: result dementated below is tested with InceptionV3 model

<p align="center">
<img src="https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group/blob/main/Images/accuracy_compare.png" style="width:250px;"/>
</p>

-	Another hyperparameter that affects model accuracy is batch_size, moreover, it varies among models. We use batch_size among 32, 64, and 128
-	We also struggle with random seed, setting fixed random seed at the top does not affect the following codes i.e. accuracy over test set changes as we repeat run. The solution of this is to set seed to all code that available


[🔝](https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group/blob/main/README.md#highlight)

## 7. Conclusion📝
-	We select 4 pre-trained models; namely, VGG16, Xception, ResNetV2, and InceptionV3
-	The best pre-trained model (No Fine-tune) among the selected models is Xception with average accuracy rate at 85.37% while after fine tuning it turns out that VGG16 becomes the best model with the highest accuracy on test set at 91.66% from 84.63%
-	After fine-tuning, all selected models show improvement on test accuracy average at 5.23% 
-	The overall run-time among the selected model is 5-6s per epochs under GPU usage
-	The most precise classification is “Cavandish Banana”. From Grad-Cam, it shows that the model focuses on banana to determine whether it is Cavandish banana or Cultivated banana whereas Lady finger banana is focused at the cut stalk of the banana for distinctive features by the model

<!-- สรุปผลของการบ้านนี้ โดยเน้นการตอบโจทย์ปัญหา (research question) หรือจุดประสงค์หลัก (objective) ของการบ้านแต่ละครั้ง -->
[🔝](https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group/blob/main/README.md#highlight)

## 8. References🌐
<!-- อ้างอิงไลบรารีที่ใช้ (พร้อมเวอร์ชัน), อ้างอิงเทคนิคที่ยืมมาใช้จากเปเปอร์, อ้างอิงโค้ดหรือรูปภาพที่หยิบยืมมาใช้จาก github หรือจากที่อื่น ๆ -->

### Library
- **`Pandas`**
- **`Numpy`**
- **`Sklearn`**
- **`Keras`**
- **`Matplotlib`**
- **`Seaborn`**
<!-- This content will not appear in the rendered Markdown -->

### Version
```
Python 3.7.15 (default, Oct 12 2022, 19:14:55) 
[GCC 7.5.0]

NumPy 1.21.6

The scikit-learn version is 1.0.2
TensorFlow 2.9.2
tf.keras.backend.image_data_format() = channels_last
TensorFlow detected 1 GPU(s):
.... GPU No. 0: Name = /physical_device:GPU:0 , Type = GPU
```
<!-- This content will not appear in the rendered Markdown -->

### References
- <sup>0</sup>_-. (2019)._
[**เรื่องกล้วยๆ**](https://www.topspicks.tops.co.th/single-post/tidbits-about-bananas2019). Topspicks.
- <sup>1</sup>_-. (2022, Sep 8)._ [**tf.keras.preprocessing.image.ImageDataGenerator**](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator). TensorFlow.
- <sup>2</sup>_Dustin. (2022, Oct 20)._ [**[Notebooks update] New GPU (T4s) options & more CPU RAM**](https://www.kaggle.com/discussions/product-feedback/361104?fbclid=IwAR2qbmFZTP6BbI7T-hHAAg8ByGiM9cZW_Ik6nHK7-WlRAu8UzoF0R2yCKZY). Kaggle.
- <sup>3</sup>_[fchollet](https://twitter.com/fchollet). (2020, May 12)._ [**Transfer learning & fine-tuning**](https://keras.io/guides/transfer_learning/). Keras.
- <sup>4</sup>_[fchollet](https://twitter.com/fchollet). (2021, March 7)._ [**Grad-CAM class activation visualization**](https://keras.io/examples/vision/grad_cam/). Keras.
- <sup>5</sup>_Jason Brownlee. (2019, Jul 5)._ [**How to Normalize, Center, and Standardize Image Pixels in Keras**](https://machinelearningmastery.com/how-to-normalize-center-and-standardize-images-with-the-imagedatagenerator-in-keras/). Machine Learning Mastery.
- <sup>6</sup>_Lang, Steven and Bravo-Marquez, Felipe and Beckham, Christopher and Hall, Mark and Frank, Eibe. (2019)._ [**IMAGENET 1000 Class List**](https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/). Github.
- <sup>7</sup>[**Keras Applications**](https://keras.io/api/applications/). Keras.
- <sup>8</sup>[**Training strategy**](https://www.neuraldesigner.com/learning/tutorials/training-strategy). Neuraldesigner.
- <sup>9</sup>[**Grad-CAM Reveals the Why Behind Deep Learning Decisions**](https://www.mathworks.com/help/deeplearning/ug/gradcam-explains-why.html). Mathworks.



[🔝](https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group/blob/main/README.md#highlight)

## Citing
[Bib.file](https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group/blob/main/Citing_CNN_MNLP.bib)

```
@Misc{MNLP,
    AUTHOR          = {Navapol San. , Pakawut Kam. , Supisara Poo. , Kantima Tec.},
    TITLE           = {Model : CNN image classification model with finetuning on a custom dataset},
    YEAR            = {2022},
    howpublished    = "\url{https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group.git}"
}
```


[🔝](https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group/blob/main/README.md#highlight)


## 👥 Members, Percent Contribution and Responsibility 
|No  |ID                |Name                              |% Contribution |Responsibility                             |
|:---:|:---:             |---                               |:---:            |:---|
|1.  |**`6410422002`**  |[Navapol San.](https://www.kaggle.com/navapol)                      |   **`25%`**     |**`Collecting data (Cavendish banana)`**, **`Fine-tune Model (Xception)`**
|2.  |**`6410422003`**  |[Pakawut Kam.](https://www.kaggle.com/ppakawut)                     |   **`25%`**     |**`Collecting data (Lady finger banana)`**, **`Fine-tune Model (InceptionV3)`**  |
|3.  |**`6410422024`**  |[Supisara Poo.](https://www.kaggle.com/supisarapo)                     |   **`25%`**     |**`Collecting data (Cultivated banana)`**, **`Fine-tune Model (ResNet50V2)`**, **`Summary the Conclution`**    |
|4.  |**`6410422027`**  |[Kantima Tec.](https://www.kaggle.com/kantimatec)                     |   **`25%`**     |**`Fine-tune Model (VGG16)`**, **`Summary the report`**  |


[🔝](https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group/blob/main/README.md#highlight)


## 🖇️End Credit 
This project is a part of **`DADS7202 Deep Learning`**

Term: 1 Year of education: 2022

🎓Master of Science Program in **`Data Analytics and Data Science`** (DADS)

🏫National Institute of Development Administration (**`NIDA`**)



[🔝](https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group/blob/main/README.md#highlight)


----------------------------------------------
<!--
❑ Github link (public access) 1 ลิงก์ต่อ 1 กลุ่ม โดยมีทั้งโค้ดที่พร้อมรัน และคำอธิบายงาน

❑ Highlight: สรุปไฮไลท์ที่น่าสนใจ bullet สั้น ๆ กระชับแต่ได้ใจความ จำนวนประมาณ 3-5 bullets ควรจะเป็นข้อคิดเห็น การค้นพบ ข้อสรุปหรือข้อมูล insight น่าสนใจที่ทางกลุ่มค้นพบจากการทำการบ้านครั้งนี้

❑ Introduction: ทำ task อะไร เช่น binary classification, single-label multi-class classification, multi-label classification, regression, segmentation, etc.

❑ Data: 
- Data source, 
- EDA, 
- data preparation, 
- data pre-processing, 
- data post-processing, 
- data splitting (train/val/test) 
- ทำอย่างไรให้ได้ข้อมูลที่ balance และครบถ้วนในแต่ละคลาส, 
- ระบุแนวทางที่จะใช้แก้ปัญหาด้วย (โดยเฉพาะงาน classification)

❑ Network architecture: รายละเอียดต่าง ๆ ของโมเดลที่เลือกใช้ (เช่น จำนวนและตำแหน่งการวาง layer, จำนวน nodes, activation function, regularization) ในรูปแบบของ network diagram หรือตาราง (โดยใส่ข้อมูลให้ละเอียดพอที่คนที่มาอ่าน จะสามารถไปสร้าง network ตามเราได้)

❑ Training: รายละเอียดของการ train และ validate ข้อมูล รวมถึงทรัพยากรที่ใช้ในการ train โมเดลหนึ่ง ๆ เช่น training strategy (เช่น single loss, compound loss, two-step training, end-to-end training), loss, optimizer (learning rate, momentum, etc), batch size,
epoch, รุ่นและจำนวน CPU หรือ GPU หรือ TPU ที่ใช้, เวลาโดยประมาณที่ใช้ train โมเดลหนึ่งตัว ฯลฯ

❑ Results: แสดงตัวเลขผลลัพธ์ในรูปของค่าเฉลี่ย mean±SD โดยให้ทำการเทรนโมเดลด้วย initial random weights ที่แตกต่างกันอย่างน้อย 3-5 รอบเพื่อให้ได้อย่างน้อย 3-5 โมเดลมาหาประสิทธิภาพเฉลี่ยกัน, แสดงผลลัพธ์การ train โมเดลเป็นกราฟเทียบ train vs. validation, สรุปผลว่าเกิด underfit หรือ overfit หรือไม่, อธิบาย evaluation metric ที่ใช้ในการประเมินประสิทธิภาพของโมเดลบน train/val/test sets ตามความเหมาะสมของปัญหา, หากสามารถเปรียบเทียบผลลัพธ์ของโมเดลเรากับโมเดลอื่น ๆ (ของคนอื่น) บน any standard benchmark dataset ได้ด้วยจะยิ่งทำให้งานดูน่าเชื่อถือยิ่งขึ้น เช่น เทียบความแม่นยำ เทียบเวลาที่ใช้train เทียบเวลาที่ใช้ inference บนซีพียูและจีพียู เทียบขนาดโมเดล ฯลฯ

o (Optional) Ablation study: ในกรณีที่โมเดลมีขนาดใหญ่และมีโมเดลย่อยซ้อนอยู่ข้างในอีกหลายส่วนจนทำให้ยากต่อการสรุปว่าโมเดลส่วนย่อยใดมีนัยสำคัญมากน้อยแค่ไหนต่อผลที่ได้บ้าง ในกรณีเหล่านี้นิยมทำ ablation study โดยทดลองลบโมเดลย่อยบางส่วนออก แล้ว train โมเดลดังกล่าวใหม่เพื่อดูว่าการดึงออกนี้มีผลทำให้ประสิทธิภาพโมเดลดีขึ้นหรือแย่ลงอย่างไร ถ้าส่วนไหนดึงออกแล้วโมเดลผลแย่ลงมากแสดงว่าส่วนนั้นมีความสำคัญต่อโมเดล ห้ามเอาออก แต่ถ้าส่วนไหนดึงออกแล้วประสิทธิภาพโมเดลเหมือนเดิมก็แสดงว่าส่วนนั้นไม่มีความจำเป็นต่อโมเดล เราสามารถลบออกเพื่อลดขนาดและความซับซ้อนของโมเดลลงได้

❑ Discussion: อภิปรายผลลัพธ์ที่ได้ว่ามีอะไรเป็นไปตามสมมติฐาน หรือมีอะไรผิดคาด ไม่เป็นไปตามสมมติฐานบ้าง, วิเคราะห์เพิ่มเติมว่าสิ่งที่ผิดคาดหรือผิดปกตินั้นน่าจะเกิดจากอะไร, ในกรณีที่ dataset มีปัญหา (เช่นกรณี imbalanced dataset) ควรวิเคราะห์ด้วยว่าวิธีแก้ที่เราใช้สามารถแก้ปัญหาของ dataset ได้จริงหรือไม่

❑ Conclusion: สรุปผลของการบ้านนี้ โดยเน้นการตอบโจทย์ปัญหา (research question) หรือจุดประสงค์หลัก (objective) ของการบ้านแต่ละครั้ง

❑ References: อ้างอิงไลบรารีที่ใช้ (พร้อมเวอร์ชัน), อ้างอิงเทคนิคที่ยืมมาใช้จากเปเปอร์, อ้างอิงโค้ดหรือรูปภาพที่หยิบยืมมาใช้จาก github หรือจากที่อื่น ๆ

❑ Citing: ในกรณีที่มีคนอยาก cite (อ้างอิง) งานหรือ dataset ของเรา เราอยากให้เขา cite เราว่าอย่างไร ส่วนใหญ่นิยมเขียนในรูปแบบของ bibtex format ตามตัวอย่างในภาพ

❑ รายชื่อสมาชิกกลุ่ม พร้อม contribution percentage ของสมาชิกแต่ละคน และรายละเอียดการแบ่งงาน ทั้งนี้สำหรับนักศึกษาที่ไม่อยากระบุชื่อนามสกุลเต็มในลิงก์สาธารณะ สามารถระบุแต่รหัสนักศึกษา หรือ ระบุเฉพาะบางส่วนของชื่อและบางส่วนของนามสกุลก็ได้

❑ End credit: ขอเพิ่มตอนท้ายสุดเล็ก ๆ นิดนึงค่ะ ประมาณว่า “งานชิ้นนี้เป็นส่วนหนึ่งของ ...ชื่อวิชา... ...ชื่อหลักสูตร... ...ชื่อมหาลัย...” (เพื่อเป็นการประชาสัมพันธ์หลักสูตรและคณะไปในตัว)

***
3. การแสดงผลลัพธ์เทียบ train vs. validation (เช่น loss, accuracy) ถ้าเป็นไปได้ควรแสดงไว้ในกราฟเดียวกันเพื่อให้สามารถเทียบ scale ค่าผลลัพธ์และดู underfit / overfit ได้ง่าย
4. ในการกล่าวถึงประสิทธิภาพหรือผลลัพธ์ใด ๆ ไม่ว่าจะเป็นในเนื้อความ ในตาราง หรือในรูปภาพ ให้ระบุให้ชัดเจนเสมอว่าสิ่งที่กล่าวถึงนั้นเป็นผลลัพธ์บน train set หรือ val set หรือ test set
5. ในการกล่าวถึงผลการคำนวณค่าต่าง ๆ ต้องระบุที่มาของการคำนวณและอธิบายวิธีการคำนวณไว้เสมอ เช่น “ค่าเฉลี่ยของ xxx” ก็ต้องอธิบายว่ามีการใช้ค่าอะไรจากไหนกี่ค่าบ้างนำมาเฉลี่ยกัน
6. ระวังว่าการเปรียบเทียบใด ๆ ต้องเป็นไปบนพื้นฐานของเงื่อนไขที่ยุติธรรมต่อคู่เทียบ เช่น หากจะเปรียบเทียบ training time ว่าโมเดลไหนมากน้อยกว่ากัน ก็ควรจะเปรียบเทียบเป็น training time per one epoch (โดยเฉลี่ยค่าจากหลาย ๆ epoch), หากจะเปรียบเทียบ inference time per one sample ก็ควรจะต้องเป็นค่าที่เฉลี่ยมาจาก test samples ชุดเดียวกันที่รันบน CPU หรือ GPU เดียวกัน, หากจะเปรียบเทียบว่า loss มากหรือน้อยกว่ากัน ก็ควรจะเปรียบเทียบที่สมการการคำนวณ loss สมการเดียวกัน เป็นต้น
7. การอภิปรายผลและการสรุปผล ต้องอ้างอิงกับผลการทดลองของเราที่ได้ออกมาเป็นหลัก มิใช่การนำข้อสรุปที่เป็น general conclusion จากหนังสือ แบบเรียน หรือจากแหล่งอื่น ๆ ในอินเทอร์เน็ต มาเขียนซ้ำโดยไม่มีผลการทดลองใด ๆ ของเรามาช่วยสนับสนุนข้อสรุปดังกล่าว
************** THE END **************-->
