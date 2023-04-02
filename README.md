<h2>EfficientNet-Acute-Myeloid-Leukemia (Updated: 2023/04/02)</h2>
<a href="#1">1 EfficientNetV2 Acute-Myeloid-Leukemia Classification </a><br>
<a href="#1.1">1.1 Clone repository</a><br>
<a href="#1.2">1.2 Prepare Peripheral Blood Cell dataset</a><br>
<a href="#1.3">1.3 Install Python packages</a><br>
<a href="#2">2 Python classes for Peripheral Blood Cell Classification</a><br>
<a href="#3">3 Pretrained model</a><br>
<a href="#4">4 Train</a><br>
<a href="#4.1">4.1 Train script</a><br>
<a href="#4.2">4.2 Training result</a><br>
<a href="#5">5 Inference</a><br>
<a href="#5.1">5.1 Inference script</a><br>
<a href="#5.2">5.2 Sample test images</a><br>
<a href="#5.3">5.3 Inference result</a><br>
<a href="#6">6 Evaluation</a><br>
<a href="#6.1">6.1 Evaluation script</a><br>
<a href="#6.2">6.2 Evaluation result</a><br>

<h2>
<a id="1">1 EfficientNetV2 Acute-Myeloid-Leukemia Classification</a>
</h2>

 This is an experimental Acute-Myeloid-Leukemia Image Classification project based on <b>efficientnetv2</b> in <a href="https://github.com/google/automl">Brain AutoML</a>.
<br>
The AML image dataset used here has been taken from the following web site;<br>
<pre>
CANCER IMAGING ARCHIVE

https://faspex.cancerimagingarchive.net/aspera/faspex/external_deliveries/77?passcode=a6be8bf0a97ddb34fc0913f37b8180d8f7d616a7

Package - AML-Cytomorphology
From:
Natasha Honomichl
To:
help@cancerimagingarchive.net 
CC (on download):
Natasha Honomichl
Date Sent:
17 Feb 2021 01:11 PM
</pre>

<br>
<br>We use python 3.8 and tensorflow 2.8.0 environment on Windows 11.<br>
<h3>
<a id="1.1">1.1 Clone repository</a>
</h3>
 Please run the following command in your working directory:<br>
<pre>
git clone https://github.com/sarah-antillia/EfficientNet-Acute-Myeloid-Leukemia.git
</pre>
You will have the following directory tree:<br>
<pre>
.
├─asset
└─projects
    └─Acute-Myeloid-Leukemia
        ├─eval
        ├─evaluation
        ├─inference        
        └─test
</pre>
<h3>
<a id="1.2">1.2 AML dataset</a>
</h3>

 
1 Resampling<br> 
We have created <b>Resampled_AML-Cytomorphology_400_360x360</b> dataset from
the original image dataset <b>AML-Cytomorphology</b> by using 
<a href="">ImageDatasetResampler</a>
The number of images of the original dataset AML-Cytomorphology is the following:
<pre>
[('BAS', 79), 
 ('EBO', 78), 
 ('EOS', 424), 
 ('KSC', 15), 
 ('LYA', 11), 
 ('LYT', 3937), 
 ('MMZ', 15), 
 ('MOB', 26), 
 ('MON', 1789), 
 ('MYB', 42), 
 ('MYO', 3268), 
 ('NGB', 109), 
 ('NGS', 8484), 
 ('PMB', 18), 
 ('PMO', 70)]
</pre>
, which is a typical imbalanced image dataset, and you can see that the numbers of images of 'KSC', 'LYA', 'MMZ' and 'PMB' are very small compared with those of 
'LYT', 'MYO' and 'NGS'.<br>
Hence, we have created <b>AML-Cytomorphology_400_360x360</b> dataset from the original dataset to resolve the imbalanced problem
by using <a href="https://github.com/martian-antillia/ImageDatasetResampler">ImageDatasetResampler</a>.

<pre>
[('BAS', 395), 
 ('EBO', 390), 
 ('EOS', 400), 
 ('KSC', 390), 
 ('LYA', 396), 
 ('LYT', 400), 
 ('MMZ', 390), 
 ('MOB', 390), 
 ('MON', 400), 
 ('MYB', 378), 
 ('MYO', 400), 
 ('NGB', 327), 
 ('NGS', 400), 
 ('PMB', 396), 
 ('PMO', 350)]
</pre>


2 Splitting train and test dataset<br>
Furthermore, we have created <b>Resampled_AML_Images</b> dataset from the <b>AML-Cytomorphology_400_360x360</b> 
by using <a href="./projects/Acute-Myeloid-Leukemia/split_master.py">split_master.py </a> script, 
by which we have splitted the master dataset to train and test dataset.<br>
<pre>
>python split_master.py
</pre> 

The destribution of images in those dataset is the following;<br>
<img src="./projects/Acute-Myeloid-Leukemia/_Resampled_AML_Images_.png" width="720" height="auto"><br>


<pre>
.
├─asset
└─projects
    └─Acute-Myeloid-Leukemia
        ├─eval
        ├─evaluation
        ├─inference
        ├─models
        │  └─chief
        ├─Resampled_AML_Images
        │  ├─test
        │  │  ├─BAS
        │  │  ├─EBO
        │  │  ├─EOS
        │  │  ├─KSC
        │  │  ├─LYA
        │  │  ├─LYT
        │  │  ├─MMZ
        │  │  ├─MOB
        │  │  ├─MON
        │  │  ├─MYB
        │  │  ├─MYO
        │  │  ├─NGB
        │  │  ├─NGS
        │  │  ├─PMB
        │  │  └─PMO
        │  └─train
        │      ├─BAS
        │      ├─EBO
        │      ├─EOS
        │      ├─KSC
        │      ├─LYA
        │      ├─LYT
        │      ├─MMZ
        │      ├─MOB
        │      ├─MON
        │      ├─MYB
        │      ├─MYO
        │      ├─NGB
        │      ├─NGS
        │      ├─PMB
        │      └─PMO
        └─test
</pre>

<br>


1 Sample images of Resampled_AML_Images/train/BAS:<br>
<img src="./asset/sample_train_images_BAS.png" width="840" height="auto">
<br> 

2 Sample images of Resampled_AML_Images/train/EBO:<br>
<img src="./asset/sample_train_images_EBO.png" width="840" height="auto">
<br> 

3 Sample images of Resampled_AML_Images/train/EOS:<br>
<img src="./asset/sample_train_images_EOS.png" width="840" height="auto">
<br> 

4 Sample images of Resampled_AML_Images/train/KSC:<br>
<img src="./asset/sample_train_images_KSC.png" width="840" height="auto">
<br> 

5 Sample images of Resampled_AML_Images/train/LYA:<br>
<img src="./asset/sample_train_images_LYA.png" width="840" height="auto">
<br> 

6 Sample images of Resampled_AML_Images/train/LYT:<br>
<img src="./asset/sample_train_images_LYT.png" width="840" height="auto">
<br>

7 Sample images of Resampled_AML_Images/train/MMZ:<br>
<img src="./asset/sample_train_images_MMZ.png" width="840" height="auto">
<br> 

8 Sample images of Resampled_AML_Images/train/MOB:<br>
<img src="./asset/sample_train_images_MOB.png" width="840" height="auto">
<br> 

9 Sample images of Resampled_AML_Images/train/MON:<br>
<img src="./asset/sample_train_images_MON.png" width="840" height="auto">
<br> 

10 Sample images of Resampled_AML_Images/train/MYB:<br>
<img src="./asset/sample_train_images_MYB.png" width="840" height="auto">
<br> 

11 Sample images of Resampled_AML_Images/train/MYO:<br>
<img src="./asset/sample_train_images_MYO.png" width="840" height="auto">
<br> 

12 Sample images of Resampled_AML_Images/train/NGB:<br>
<img src="./asset/sample_train_images_NGB.png" width="840" height="auto">
<br> 

13 Sample images of Resampled_AML_Images/train/NGS:<br>
<img src="./asset/sample_train_images_NGS.png" width="840" height="auto">
<br> 

14 Sample images of Resampled_AML_Images/train/PMB:<br>
<img src="./asset/sample_train_images_PMB.png" width="840" height="auto">
<br> 

15 Sample images of Resampled_AML_Images/train/PMO:<br>
<img src="./asset/sample_train_images_PMO.png" width="840" height="auto">
<br> 

<br> 

<br>


<h3>
<a id="#1.3">1.3 Install Python packages</a>
</h3>
Please run the following commnad to install Python packages for this project.<br>
<pre>
pip install -r requirements.txt
</pre>
<br>

<h2>
<a id="2">2 Python classes for LymphomaClassification</a>
</h2>
We have defined the following python classes to implement our LymphomaClassification.<br>
<li>
<a href="./ClassificationReportWriter.py">ClassificationReportWriter</a>
</li>
<li>
<a href="./ConfusionMatrix.py">ConfusionMatrix</a>
</li>
<li>
<a href="./CustomDataset.py">CustomDataset</a>
</li>
<li>
<a href="./EpochChangeCallback.py">EpochChangeCallback</a>
</li>
<li>
<a href="./EfficientNetV2Evaluator.py">EfficientNetV2Evaluator</a>
</li>
<li>
<a href="./EfficientNetV2Inferencer.py">EfficientNetV2Inferencer</a>
</li>
<li>
<a href="./EfficientNetV2ModelTrainer.py">EfficientNetV2ModelTrainer</a>
</li>
<li>
<a href="./FineTuningModel.py">FineTuningModel</a>
</li>

<li>
<a href="./TestDataset.py">TestDataset</a>
</li>

<h2>
<a id="3">3 Pretrained model</a>
</h2>
 We have used pretrained <b>efficientnetv2-m</b> to train LymphomaModel.
Please download the pretrained checkpoint file 
from <a href="https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/v2/efficientnetv2-m.tgz">efficientnetv2-m.tgz</a>, expand it, and place the model under our top repository.

<pre>
.
├─asset
├─efficientnetv2-m
└─projects
    └─Acute-Myeloid-Leukemia
 ...
</pre>

<h2>
<a id="4">4 Train</a>

</h2>
<h3>
<a id="4.1">4.1 Train script</a>
</h3>
Please run the following bat file to train our AML efficientnetv2 model by using
<b>Resampled_AML_images/train</b>.
<pre>
./1_train.bat
</pre>
<pre>
rem 1_train.bat
python ../../EfficientNetV2ModelTrainer.py ^
  --model_dir=./models ^
  --eval_dir=./eval ^
  --model_name=efficientnetv2-m ^
  --data_generator_config=./data_generator.config ^
  --ckpt_dir=../../efficientnetv2-m/model ^
  --optimizer=rmsprop ^
  --image_size=360 ^
  --eval_image_size=360 ^
  --data_dir=./Resampled_AML_images/train ^
  --data_augmentation=True ^
  --fine_tuning=True ^
  --monitor=val_loss ^
  --learning_rate=0.001 ^
  --trainable_layers_ratio=0.4 ^
  --dropout_rate=0.2 ^
  --num_epochs=100 ^
  --batch_size=4 ^
  --patience=10 ^
  --debug=True  

</pre>
, where data_generator.config is the following:<br>
<pre>
; data_generation.config
[training]
validation_split   = 0.2
featurewise_center = False
samplewise_center  = False
featurewise_std_normalization=False
samplewise_std_normalization =False
zca_whitening                =False
rotation_range     = 10
horizontal_flip    = True
vertical_flip      = True
width_shift_range  = 0.05
height_shift_range = 0.05
shear_range        = 0.00
zoom_range         = [0.5, 1.5]

;channel_shift_range= 10
;brightness_range   = [80,100]
data_format        = "channels_last"

</pre>

<h3>
<a id="4.2">4.2 Training result</a>
</h3>

This will generate a <b>best_model.h5</b> in the models folder specified by --model_dir parameter.<br>
Furthermore, it will generate a <a href="./projects/Acute-Myeloid-Leukemia/eval/train_accuracies.csv">train_accuracies</a>
and <a href="./projects/Acute-Myeloid-Leukemia/eval/train_losses.csv">train_losses</a> files
<br>
Training console output:<br>
<img src="./asset/train_at_epoch_15_0402.png" width="740" height="auto"><br>
<br>
Train_accuracies:<br>
<img src="./projects/Acute-Myeloid-Leukemia/eval/train_accuracies.png" width="640" height="auto"><br>

<br>
Train_losses:<br>
<img src="./projects/Acute-Myeloid-Leukemia/eval/train_losses.png" width="640" height="auto"><br>

<br>
<h2>
<a id="5">5 Inference</a>
</h2>
<h3>
<a id="5.1">5.1 Inference script</a>
</h3>
Please run the following bat file to infer the skin cancer lesions in test images by the model generated by the above train command.<br>
<pre>
./2_inference.bat
</pre>
<pre>
rem 2_inference.bat
python ../../EfficientNetV2Inferencer.py ^
  --model_name=efficientnetv2-m  ^
  --model_dir=./models ^
  --fine_tuning=True ^
  --trainable_layers_ratio=0.4 ^
  --dropout_rate=0.2 ^
  --image_path=./test/*.tiff ^
  --eval_image_size=360 ^
  --label_map=./label_map.txt ^
  --mixed_precision=True ^
  --infer_dir=./inference ^
  --debug=False 
</pre>
<br>
label_map.txt:
<pre>
BAS
EBO
EOS
KSC
LYA
LYT
MMZ
MOB
MON
MYB
MYO
NGB
NGS
PMB
PMO
</pre>
<br>
<h3>
<a id="5.2">5.2 Sample test images</a>
</h3>

Sample test images generated by <a href="./projects/Acute-Myeloid-Leukemia/create_test_dataset.py">create_test_dataset.py</a> 
from <a href="./projects/Acute-Myeloid-Leukemia/Resampled_AML_Images/test">Lymphoma/test</a>.
<br>
<img src="./asset/test.png" width="840" height="auto"><br>


<br>
<h3>
<a id="5.3">5.3 Inference result</a>
</h3>
This inference command will generate <a href="./projects/Acute-Myeloid-Leukemia/inference/inference.csv">inference result file</a>.
<br>At this time, you can see the inference accuracy for the test dataset by our trained model is very low.
More experiments will be needed to improve accuracy.<br>

<br>
Inference console output:<br>
<img src="./asset/inference_at_epoch_15_0402.png" width="740" height="auto"><br>
<br>

Inference result (<a href="./projects/Acute-Myeloid-Leukemia/inference/inference.csv">inference.csv</a>):<br>
<img src="./asset/inference_at_epoch_15_0402_csv.png" width="640" height="auto"><br>
<br>
<h2>
<a id="6">6 Evaluation</a>
</h2>
<h3>
<a id="6.1">6.1 Evaluation script</a>
</h3>
Please run the following bat file to evaluate <a href="./projects/Acute-Myeloid-Leukemia/Resampled_AML_Images/test">
Malaris_Cell_Images/test</a> by the trained model.<br>
<pre>
./3_evaluate.bat
</pre>
<pre>
rem 3_evaluate.bat
python ../../EfficientNetV2Evaluator.py ^
  --model_name=efficientnetv2-m  ^
  --model_dir=./models ^
  --data_dir=./Resampled_AML_Images/test ^
  --evaluation_dir=./evaluation ^
  --fine_tuning=True ^
  --trainable_layers_ratio=0.4 ^
  --dropout_rate=0.2 ^
  --eval_image_size=360 ^
  --mixed_precision=True ^
  --debug=False 
 
</pre>
<br>

<h3>
<a id="6.2">6.2 Evaluation result</a>
</h3>

This evaluation command will generate <a href="./projects/Acute-Myeloid-Leukemia/evaluation/classification_report.csv">a classification report</a>
 and <a href="./projects/Acute-Myeloid-Leukemia/evaluation/confusion_matrix.png">a confusion_matrix</a>.
<br>
<br>
Evaluation console output:<br>
<img src="./asset/evaluate_at_epoch_15_0402.png" width="740" height="auto"><br>
<br>

<br>
Classification report:<br>
<img src="./asset/classification_report_at_epoch_15_0402.png" width="740" height="auto"><br>
<br>
Confusion matrix:<br>
<img src="./projects/Acute-Myeloid-Leukemia/evaluation/confusion_matrix.png" width="740" height="auto"><br>

<br>
<h3>
References
</h3>
<b>1. AML-Cytomorphology</b><br>
<pre>
The AML image dataset used here has been taken from the following web site;
CANCER IMAGING ARCHIVE
https://faspex.cancerimagingarchive.net/aspera/faspex/external_deliveries/77?passcode=a6be8bf0a97ddb34fc0913f37b8180d8f7d616a7
</pre>

<b>2. Acute Myeloid Leukemia classification using a federated Convolutional Neural Network</b><br>
scaleoutsystems<br>
<pre>
https://github.com/scaleoutsystems/AML-tutorial
</pre>

<b>3. Deep learning detects acute myeloid leukemia and predicts NPM1 mutation status from bone marrow smears</b><br>
Jan-Niklas Eckardt, Jan Moritz Middeke, Sebastian Riechert, Tim Schmittmann, Anas Shekh Sulaiman, <br>
Michael Kramer, Katja Sockel, Frank Kroschinsky, Ulrich Schuler, Johannes Schetelig, Christoph Röllig,<br> 
Christian Thiede, Karsten Wendt & Martin Bornhäuser <br>

<pre>
https://www.nature.com/articles/s41375-021-01408-w
</pre>


<b>4. AMLnet, A deep-learning pipeline for the differential diagnosis of acute myeloid leukemia from bone marrow smears</b><br>
Zebin Yu, Jianhu Li, Xiang Wen, Yingli Han, Penglei Jiang, Meng Zhu, Minmin Wang, Xiangli Gao, <br>
Dan Shen, Ting Zhang, Shuqi Zhao, Yijing Zhu, Jixiang Tong, Shuchong Yuan, HongHu Zhu, He Huang & Pengxu Qian <br>
<pre>
https://jhoonline.biomedcentral.com/articles/10.1186/s13045-023-01419-3
</pre>
