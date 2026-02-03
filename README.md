<img width="1383" height="736" alt="image" src="https://github.com/user-attachments/assets/91aa5e3b-c38b-4f94-a333-c197ca7abe60" />


***

## 1. Motivation

Detecting dangerous objects such as knives in diverse real‑world environments is a critical problem for video surveillance and intelligent vision systems.  In this project, I build a custom knife image dataset from videos and compare four CNN‑based classification models: VGG16, ResNet50, DenseNet121, and EfficientNetB0.  Rather than performing a simple overall accuracy comparison, the goal is to analyze which model works best under specific environmental conditions (e.g., lighting, background, occlusion, knife type) and to propose situation‑aware model recommendations. [journal.50sea](https://journal.50sea.com/index.php/IJIST/article/view/1425)

***

## 2. Data Collection

This project does not use any public external dataset. All data were collected by myself from 19 recorded videos, from which knife images were extracted using a YOLOv8 detector.  The YOLOv8s model is loaded via the `ultralytics` library and applied frame‑by‑frame; whenever the predicted class label is “knife,” the corresponding bounding box region is cropped and saved as an image.  The full extraction pipeline, including video loading, detection, and image saving, is implemented in the provided Colab script, and the generated dataset is uploaded to my GitHub repository: [github](https://github.com/JoaoAssalim/Weapons-and-Knives-Detector-with-YOLOv8)
`https://github.com/jwdebbie/knife-classification`. [github](https://github.com/GingerBrains/object-detection)

For transfer learning, I use the following pre‑trained models from `keras.applications`: VGG16, ResNet50, DenseNet121, and EfficientNetB0.  Each model is initialized with ImageNet‑pretrained weights, the original classification head is removed, and a new output layer is added for three knife classes.  Only my custom knife images are used for training and evaluation; no additional external images are incorporated. [pyimagesearch](https://pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/)

To reduce ambiguity in illumination labels, lighting conditions are simplified to two categories: **bright** and **dark**, excluding earlier “backlight” cases that were hard to classify consistently.  To mitigate dataset bias, the number of source videos was increased from 8 to 19, balancing combinations of lighting, background, occlusion, and knife type. [datasetninja](https://datasetninja.com/od-weapon-detection-knife-detection)

The 19 source videos are designed to cover diverse environments:

- Lighting: bright / dark  
- Backgrounds: kitchen, desk, living room, floor, window  
- Occlusion: with / without occlusion  
- Knife types: kitchen knife, fruit knife, cutter knife  

Each video file name encodes these conditions, for example:  
`knife01_bright_kitchen_식칼_none.mp4`, `knife05_dark_kitchen_과도_yes.mp4`, `knife18_bright_window_커터칼_yes.mp4`, etc. [datasetninja](https://datasetninja.com/od-weapon-detection-knife-classification)

***

## 3. Dataset Description

- Data type: RGB images, resized to 224×224 pixels for all models. [keras](https://keras.io/api/applications/)
- Source: Frames extracted from 19 self‑recorded knife videos using YOLOv8‑based detection. [ijraset](https://www.ijraset.com/research-paper/real-time-weapon-detection-using-yolov8)
- Total size: Approximately 15,800 images of knives. [datasetninja](https://datasetninja.com/od-weapon-detection-knife-detection)
- Condition factors: lighting (bright/dark), five types of background scenes, presence or absence of occlusion, and three knife categories. [datasetninja](https://datasetninja.com/od-weapon-detection-knife-classification)

Each CNN model receives a 224×224 RGB knife image as input, after standard preprocessing corresponding to its Keras implementation (e.g., normalization and scaling).  The output is a 3‑class prediction over {cutter knife, fruit knife, kitchen knife}, which is used to compute accuracy and other metrics under different environmental conditions.  The core objective of the experiments is to identify which architecture is most robust to changes in lighting, background, and occlusion, and to derive scenario‑specific model recommendations for real‑world deployment. [nature](https://www.nature.com/articles/s41598-025-07782-0)
