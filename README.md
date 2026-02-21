
---

# Knife Classification Dataset

**Environment-aware knife image dataset extracted from real-world videos using YOLOv8**

---

## Overview

This repository contains a custom knife image dataset generated from real-world videos recorded by the author.
The dataset is designed to analyze **how different CNN-based classification models perform under various environmental conditions**, rather than performing a simple model accuracy comparison.

All images were automatically extracted from raw videos using a pre-trained **YOLOv8** detector.

The dataset is intended for research on:

* illumination robustness
* background variation
* occlusion robustness
* knife-type classification

---

## Motivation

Detecting dangerous objects such as knives in real-world environments is a critical task in video-based intelligent systems.

However, performance of vision models often varies significantly depending on environmental conditions such as:

* lighting
* background
* partial occlusion
* object appearance

The goal of this dataset is to support **environment-condition-based analysis of classification models**, focusing on:

> **Which model is more robust under specific environmental conditions (lighting, background, occlusion), rather than which model performs best overall.**

---

## Data Collection

* All videos were recorded by the author.
* No public or external datasets were used.
* A total of **19 videos** were recorded under different combinations of:

  * illumination
  * background
  * occlusion
  * knife type

Knives were automatically detected from each video using a pre-trained YOLOv8 model and cropped to generate image samples.

---

## Data Generation Pipeline

Knife regions were extracted using **YOLOv8s** from the Ultralytics library.

Main steps:

1. Load each video.
2. Run YOLOv8 object detection on each frame.
3. Crop detected bounding boxes labeled as `knife`.
4. Save cropped images.

The generation script used in this repository is summarized below.

```python
from ultralytics import YOLO
import cv2
import os

model = YOLO('yolov8s.pt')

def extract_knives_from_video(video_path, save_dir, conf=0.25, save_limit=50):
    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved = 0

    while cap.isOpened() and saved < save_limit:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]

                if label == 'knife':
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    crop = frame[y1:y2, x1:x2]
                    filename = f"{save_dir}/knife_{frame_count:04d}.jpg"
                    cv2.imwrite(filename, crop)
                    saved += 1

        frame_count += 1

    cap.release()
```

> YOLOv8 is used **only for dataset generation**, not for the classification experiments.

---

## Video Configuration

Each video was recorded under a controlled combination of environmental conditions.

| ID | Illumination | Background  | Occlusion | Knife type    | Filename                              |
| -- | ------------ | ----------- | --------- | ------------- | ------------------------------------- |
| 1  | bright       | kitchen     | no        | kitchen knife | knife01_bright_kitchen_식칼_none.mp4    |
| 2  | dark         | desk        | no        | fruit knife   | knife02_dark_desk_과도_none.mp4         |
| 3  | bright       | living room | yes       | cutter knife  | knife03_bright_livingroom_커터칼_yes.mp4 |
| 4  | bright       | floor       | no        | kitchen knife | knife04_bright_floor_식칼_none.mp4      |
| 5  | dark         | kitchen     | yes       | fruit knife   | knife05_dark_kitchen_과도_yes.mp4       |
| 6  | bright       | window      | no        | cutter knife  | knife06_bright_window_커터칼_none.mp4    |
| 7  | bright       | desk        | yes       | kitchen knife | knife07_bright_desk_식칼_yes.mp4        |
| 8  | dark         | living room | no        | cutter knife  | knife08_dark_livingroom_커터칼_none.mp4  |
| 9  | dark         | floor       | yes       | cutter knife  | knife09_dark_floor_커터칼_yes.mp4        |
| 10 | dark         | window      | yes       | cutter knife  | knife10_dark_window_커터칼_yes.mp4       |
| 11 | bright       | floor       | yes       | cutter knife  | knife11_bright_floor_커터칼_yes.mp4      |
| 12 | dark         | kitchen     | no        | cutter knife  | knife12_dark_kitchen_커터칼_none.mp4     |
| 13 | bright       | window      | yes       | cutter knife  | knife13_bright_window_커터칼_yes.mp4     |
| 14 | dark         | floor       | yes       | fruit knife   | knife14_dark_floor_과도_yes.mp4         |
| 15 | bright       | kitchen     | no        | fruit knife   | knife15_bright_kitchen_과도_none.mp4    |
| 16 | bright       | floor       | yes       | fruit knife   | knife16_bright_floor_과도_yes.mp4       |
| 17 | bright       | desk        | yes       | kitchen knife | knife17_bright_desk_식칼_yes.mp4        |
| 18 | bright       | window      | yes       | cutter knife  | knife18_bright_window_커터칼_yes.mp4     |
| 19 | dark         | living room | no        | kitchen knife | knife19_dark_livingroom_식칼_none.mp4   |

---

## Illumination Policy

Originally, backlight conditions were also considered.
However, to avoid ambiguous classification of lighting conditions, illumination was simplified into two categories:

* **bright**
* **dark**

---

## Dataset Statistics

* Data type: RGB images
* Image size: 224 × 224
* Total images: approximately **15,800**
* Source: frames extracted from 19 self-recorded videos
* Number of classes: 3

### Class labels

* 커터칼 (cutter knife)
* 과도 (fruit knife)
* 식칼 (kitchen knife)

---

## Environmental Factors

Each image is associated with the following conditions (inherited from the source video):

* illumination: bright / dark
* background: kitchen, desk, living room, floor, window
* occlusion: yes / no
* knife type: 3 categories

This enables **condition-wise performance analysis**.

---

## Directory Structure

```
knives_dataset/
 ├── knife01_bright_kitchen_식칼_none/
 │    ├── knife_0000.jpg
 │    ├── knife_0001.jpg
 │    └── ...
 ├── knife02_dark_desk_과도_none/
 ├── ...
 └── knife19_dark_livingroom_식칼_none/
```

Each folder corresponds to one recorded video and one environmental configuration.

---

## Intended Use

This dataset is designed for:

* transfer-learning based image classification
* robustness analysis under environmental changes
* comparative studies of CNN architectures

In the associated experiments, the following ImageNet-pretrained models were used:

* VGG16
* ResNet50
* DenseNet121
* EfficientNetB0

---

## Task Definition

### Input

* RGB image of a cropped knife region
* resized to 224 × 224

### Output

* one of the following labels:

  * 커터칼
  * 과도
  * 식칼

---

## Key Research Focus

The main objective of this dataset is:

> To analyze which classification model is more effective and robust under specific environmental conditions such as illumination, background, and occlusion.

---

## Related Repository

This dataset is used in the following project:

https://github.com/jwdebbie/Robustness-Research.git

---

## Notes

* Bounding boxes are automatically generated by YOLOv8.
* No manual annotation was applied after extraction.
* Detection errors and imperfect crops may exist.

---

## License

This dataset is provided for **academic and research purposes only**.

---

## Author

Joowon Lee
GitHub: [https://github.com/jwdebbie](https://github.com/jwdebbie)
