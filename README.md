# DADS7202_Group Assignment 2 CNN (Group_MNLP)
> Objective: **`What do you use to train an image classiffier with our custom image dataset?`**
## ✨Highlight
- 1...
- 2...
- 3...

<!-- เป็นข้อคิดเห็น การค้นพบ ข้อสรุปหรือข้อมูล insight น่าสนใจที่  3-5 bullets -->

## Table of Contents

 - [1. Introduction🎯](https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group/blob/main/README.md#1-introduction)
 - [2. Data📑](https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group#2-data)
 - [3. Network architecture📦](https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group#3-network-architecture)
 - [4. Training🔮](https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group#4-training)
 - [5. Results📈](https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group#5-results)
 - [6. Discussion💭](https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group/blob/main/README.md#6-discussion)
 - [7. Conclusion📊](https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group/blob/main/README.md#7-conclusion)
 - [8. References🌐](https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group/blob/main/README.md#8-references)
 - [Citing](https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group/blob/main/README.md#citing)
 - [👥 Members, Percent Contribution and Responsibility](https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group/blob/main/README.md#-members-percent-contribution-and-responsibility)
 - [🖇️End Credit ](https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group/blob/main/README.md#%EF%B8%8Fend-credit)



## 1. Introduction🎯 

**`multi-class classification`**:

- This project aims to test 3**CNN pre-training models** (`VGG16`, `ResNet50V2`, `EfficientNetB7`) on the ImageNet dataset and fine-tune it to classify 4 types of bananas 🍌 (`Cultivated banana`, `Sugar banana`, `Lady finger banana`, `Cavendish banana`) which is our custom image dataset that were never trained on. 
- Then, we will compare performance of **3 CNN pre-training models** without transfer learning and with transfer learning (Fine-tuning).
- Finally, we use **`Grad-CAM`** technique to debug the model and gain more insight into what a trained CNN did.

 
## 2. Data📑
There are many banana varieties in Thailand and each one of them has different characteristics. Let’s find out interesting facts about different varieties of banana before training and finetuning models.

 🍌 **1. Cultivated banana - กล้วยน้ำว้า**
  
  The fruit looks a little bit angled with a thick skin and a sweet flavor.
  
  <img src="https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group/blob/main/Images/Cultivated%20banana.png" style="width:120px;"/>
 
 🍌 **2. Sugar banana - กล้วยไข่**
  
  The fruit is short and round. Its skin is thin with dark spots. 
  
  <img src="https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group/blob/main/Images/Sugar%20banana.png" style="width:120px;"/>
 
 🍌 **3. Lady finger banana - กล้วยเล็บมือนาง**
  
  Lady finger banana is one of the smallest bananas. Its skin is thick with a soft flesh inside.
  
   <img src="https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group/blob/main/Images/Lady%20finger%20banana.png" style="width:120px;"/>
  
   
 🍌 **4. Cavendish banana - กล้วยหอม**
  
  The fruit is long with a thin skin. It offers a sweet flavor along with a uniquely pleasant smell.
  
  <img src="https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group/blob/main/Images/Cavendish%20banana.jpg" style="width:120px;"/>
 
#### 📍Data source: 
- We use [**Download All Images**](https://chrome.google.com/webstore/detail/download-all-images/ifipmflagepipjokmbdecpmjbibjnakm?hl=en) extension in chrome web store to collect set of images by searching keywords (4 types of banana) from **`google image`** 

#### 🧹Data preparation:
- Collecting set of images from the Internet source is a quick and simple method to gather a set of images. Some facts, meanwhile, are not entirely accurate or useful. As a result, we have to manually remove several unnecessary images from the collection, such as banana dessert, banana trunk, other banana pieces, and duplicate images. Additionally, because the keyword and banana type are inconsistent, we need to recheck the banana type labels.
- 
#### Data pre-processing: **`➕Data Augmentation`** 
<!-- ใช้ ImageDataGenerator or Random xx -->

#### ✂️Data splitting (train/val/test):
- `random_state` =  
- `test_size` = 
- **`Train Shape`**: 
- **`Test Shape`**: 

[🔝](https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group/blob/main/README.md#highlight)

## 3. Network architecture📦

### Pre-training Models 
In this experiment,we have selected 3 Pre-training Models for fine-tuning

<img src="https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group/blob/main/Images/pre-training-models-info.png" style="width:550px;"/>

<!-- รายละเอียดต่าง ๆ ของโมเดลที่เลือกใช้ (เช่น จำนวนและตำแหน่งการวาง layer, จำนวน nodes, activation function, regularization) ในรูปแบบของ network diagram หรือตาราง (โดยใส่ข้อมูลให้ละเอียดพอที่คนที่มาอ่าน จะสามารถไปสร้าง network ตามเราได้) -->

[🔝](https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group/blob/main/README.md#highlight)


## 4. Training🔮
<!-- รายละเอียดของการ train และ validate ข้อมูล รวมถึงทรัพยากรที่ใช้ในการ train โมเดลหนึ่ง ๆ เช่น training strategy (เช่น single loss, compound loss, two-step training, end-to-end training), loss, optimizer (learning rate, momentum, etc), batch size,
epoch, รุ่นและจำนวน CPU หรือ GPU หรือ TPU ที่ใช้, เวลาโดยประมาณที่ใช้ train โมเดลหนึ่งตัว ฯลฯ -->
[🔝](https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group/blob/main/README.md#highlight)

## 5. Results📈
<!-- แสดงตัวเลขผลลัพธ์ในรูปของค่าเฉลี่ย mean±SD โดยให้ทำการเทรนโมเดลด้วย initial random weights ที่แตกต่างกันอย่างน้อย 3-5 รอบเพื่อให้ได้อย่างน้อย 3-5 โมเดลมาหาประสิทธิภาพเฉลี่ยกัน, แสดงผลลัพธ์การ train โมเดลเป็นกราฟเทียบ train vs. validation, สรุปผลว่าเกิด underfit หรือ overfit หรือไม่, อธิบาย evaluation metric ที่ใช้ในการประเมินประสิทธิภาพของโมเดลบน train/val/test sets ตามความเหมาะสมของปัญหา, หากสามารถเปรียบเทียบผลลัพธ์ของโมเดลเรากับโมเดลอื่น ๆ (ของคนอื่น) บน any standard benchmark dataset ได้ด้วยจะยิ่งทำให้งานดูน่าเชื่อถือยิ่งขึ้น เช่น เทียบความแม่นยำ เทียบเวลาที่ใช้train เทียบเวลาที่ใช้ inference บนซีพียูและจีพียู เทียบขนาดโมเดล ฯลฯ 

การแสดงผลลัพธ์เทียบ train vs. validation (เช่น loss, accuracy) ถ้าเป็นไปได้ควรแสดงไว้ในกราฟเดียวกันเพื่อให้สามารถเทียบ scale ค่าผลลัพธ์และดู underfit / overfit ได้ง่าย-->
[🔝](https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group/blob/main/README.md#highlight)


## 6. Discussion💭
<!-- อภิปรายผลลัพธ์ที่ได้ว่ามีอะไรเป็นไปตามสมมติฐาน หรือมีอะไรผิดคาด ไม่เป็นไปตามสมมติฐานบ้าง, วิเคราะห์เพิ่มเติมว่าสิ่งที่ผิดคาดหรือผิดปกตินั้นน่าจะเกิดจากอะไร, ในกรณีที่ dataset มีปัญหา (เช่นกรณี imbalanced dataset) ควรวิเคราะห์ด้วยว่าวิธีแก้ที่เราใช้สามารถแก้ปัญหาของ dataset ได้จริงหรือไม่ -->

[🔝](https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group/blob/main/README.md#highlight)

## 7. Conclusion📊
<!-- สรุปผลของการบ้านนี้ โดยเน้นการตอบโจทย์ปัญหา (research question) หรือจุดประสงค์หลัก (objective) ของการบ้านแต่ละครั้ง -->
[🔝](https://github.com/lukplamino/DADS7202_HW02-CNN_MNLP_Group/blob/main/README.md#highlight)

## 8. References🌐
<!-- This content will not appear in the rendered Markdown -->
### Library
<!-- This content will not appear in the rendered Markdown -->
### Version
<!-- This content will not appear in the rendered Markdown -->
### References
- _-. (2019)._
[**เรื่องกล้วยๆ**](https://www.topspicks.tops.co.th/single-post/tidbits-about-bananas2019): Topspicks.
- _Lang, Steven and Bravo-Marquez, Felipe and Beckham, Christopher and Hall, Mark and Frank, Eibe. (2019)._ [**IMAGENET 1000 Class List**](https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/). Github.
- _[fchollet](https://twitter.com/fchollet). (2020, May 12)._[**Transfer learning & fine-tuning**](https://keras.io/guides/transfer_learning/). Keras.
- **Keras Applications**](https://keras.io/api/applications/). Keras.

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
|1.  |**`6410422002`**  |[Navapol San.](https://www.kaggle.com/navapol)                      |   **`25%`**     |**`Collecting data (Cavendish banana)`**, **`Train Model (EfficientNetB7)`**
|2.  |**`6410422003`**  |[Pakawut Kam.](https://www.kaggle.com/ppakawut)                     |   **`25%`**     |**`Collecting data (Lady finger banana)`**, **`Train Model (EfficientNetB7)`**  |
|3.  |**`6410422024`**  |[Supisara Poo.](https://www.kaggle.com/supisarapo)                     |   **`25%`**     |**`Collecting data (Cultivated banana)`**, **`Train Model (ResNet50V2)`**   |
|4.  |**`6410422027`**  |[Kantima Tec.](https://www.kaggle.com/kantimatec)                     |   **`25%`**     |**`Collecting data (Sugar banana)`**, **`Train Model (VGG16)`**  |


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
