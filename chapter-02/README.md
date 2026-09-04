# Chapter 2: Training and Evaluating YOLO26

In this chapter, we will learn the practical workflow of training a YOLO26 model.

We will start with the dataset and continue through training, evaluation, and testing.

The general workflow looks like this:

```text
Dataset
   ↓
Data Preparation
   ↓
Configuration
   ↓
Model Selection
   ↓
Training
   ↓
Evaluation
   ↓
Testing
```

---

# Table of Contents

* Preparing the Environment
* Installing YOLO
* Understanding the Dataset Structure
* Creating a `data.yaml` File
* Choosing a Model
* Understanding Training Parameters
* Training a Model
* Understanding Training Results
* Evaluating the Model
* Testing the Model
* Saving the Best Model

---

# 1. Preparing the Environment

Before training a model, you need a working Python environment.

A virtual environment is recommended.

Example:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Using a virtual environment helps you keep the project dependencies separate from other Python projects.

---

# 2. Installing the Required Libraries

Install the Ultralytics package:

```bash
pip install ultralytics
```

You can check the installation with:

```bash
yolo checks
```

You can also check whether PyTorch can access your GPU.

```python
import torch

print(torch.cuda.is_available())
```

If the result is:

```text
True
```

PyTorch can access a CUDA-compatible GPU.

---

# 3. Understanding the Dataset Structure

A common YOLO dataset structure looks like this:

```text
dataset/
│
├── images/
│   ├── train/
│   ├── val/
│   └── test/
│
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
│
└── data.yaml
```

The `images` directory contains the original images.

The `labels` directory contains the annotation files.

Each image usually has a matching label file.

For example:

```text
images/train/image_001.jpg
labels/train/image_001.txt
```

---

# 4. Understanding YOLO Labels

YOLO uses a simple format for object annotations.

A label can look like this:

```text
0 0.5 0.5 0.2 0.3
```

The values represent:

```text
Class_ID
Center_X
Center_Y
Width
Height
```

The coordinates are normalized between `0` and `1`.

For example:

```text
0 → Drone
1 → Bird
2 → Person
```

---

# 5. Creating the `data.yaml` File

The `data.yaml` file tells YOLO where the dataset is located.

Example:

```yaml
path: /path/to/dataset

train: images/train
val: images/val
test: images/test

names:
  0: drone
  1: bird
  2: person
```

This file is extremely important.

If the paths or class IDs are incorrect, the training process may fail or produce incorrect results.

Before training, always verify:

* Dataset paths
* Image folders
* Label folders
* Class IDs
* Class names

---

# 6. Choosing a YOLO26 Model

Different model sizes are available for different requirements.

For example:

```text
YOLO26n → Nano
YOLO26s → Small
YOLO26m → Medium
YOLO26l → Large
YOLO26x → Extra Large
```

A good starting point is usually a small model.

For example:

```text
YOLO26n
```

The goal of the first training is to create a baseline.

Do not start by testing many complicated configurations.

First, train a simple baseline model.

Then analyze the results.

---

# 7. Important Training Parameters

Several parameters affect the training process.

---

## Epochs

Epochs represent how many times the model sees the complete training dataset.

For example:

```text
epochs=100
```

This means the training process can run for up to 100 epochs.

More epochs do not always produce a better model.

Too many epochs can cause overfitting.

---

## Batch Size

Batch size represents how many images are processed together.

For example:

```text
batch=16
```

A larger batch size usually requires more GPU memory.

If you receive a memory error, try reducing the batch size.

Example:

```text
batch=16
```

to:

```text
batch=8
```

or:

```text
batch=4
```

---

## Image Size

Image size defines the resolution used during training.

For example:

```text
imgsz=640
```

Common values include:

```text
416
640
800
1280
```

Larger image sizes may improve small object detection.

However, they also require more computational resources.

---

## Device

The `device` parameter defines where the model will run.

For example:

```text
device=0
```

usually means the first GPU.

You can also use:

```text
device=cpu
```

However, training on a CPU is usually much slower.

---

## Workers

Workers are used for loading and preparing data.

Example:

```text
workers=4
```

The best number depends on your CPU and storage performance.

---

# 8. Training a YOLO26 Model

A simple Python training script can look like this:

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

results = model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    device=0
)
```

This code performs the following steps:

```text
Load Model
    ↓
Load Dataset
    ↓
Start Training
    ↓
Validate After Each Epoch
    ↓
Save Results
```

---

# 9. Pretrained Models

Using pretrained weights is usually recommended.

A pretrained model already contains knowledge learned from a previous dataset.

For example:

```python
model = YOLO("yolo26n.pt")
```

The model will then be fine-tuned using your dataset.

This is usually faster than training a model from the beginning.

This process is called:

```text
Transfer Learning
```

---

# 10. Monitoring the Training Process

During training, YOLO produces different values.

Some important values include:

* Box Loss
* Classification Loss
* Precision
* Recall
* mAP50
* mAP50-95

You should not focus on only one metric.

Instead, analyze the complete training result.

---

# 11. Understanding Precision

Precision answers this question:

> When the model detects an object, how often is the detection correct?

Higher Precision usually means fewer false positives.

For example:

```text
Precision = 0.90
```

This means that many of the model's predictions are correct.

---

# 12. Understanding Recall

Recall answers this question:

> How many real objects did the model successfully detect?

A low Recall may indicate that the model is missing objects.

For example:

```text
Precision = High
Recall = Low
```

This can mean that the model is careful when making predictions but misses many objects.

---

# 13. Understanding mAP

mAP stands for:

```text
Mean Average Precision
```

It is one of the most important metrics for object detection.

Common values include:

```text
mAP50
mAP50-95
```

In general:

```text
Higher mAP = Better Detection Performance
```

However, mAP should always be analyzed together with Precision and Recall.

---

# 14. Understanding Overfitting

Overfitting happens when a model learns the training dataset too specifically.

The model may perform well during training but poorly on new images.

A simplified example:

```text
Training Performance → Very High
Validation Performance → Low
```

This can be a sign of overfitting.

Possible solutions include:

* More training data
* Better dataset diversity
* Data augmentation
* Fewer training epochs
* Better regularization

---

# 15. Evaluating the Model

After training, you should test the model using validation data.

Example:

```python
from ultralytics import YOLO

model = YOLO("best.pt")

metrics = model.val()
```

The evaluation process helps answer an important question:

> How well does the model perform on data that it did not directly train on?

---

# 16. Testing the Model

After training, you can test the model on a new image.

Example:

```python
from ultralytics import YOLO

model = YOLO("best.pt")

results = model("test_image.jpg")

for result in results:
    result.show()
```

You should test the model using images that represent the real deployment environment.

For example, if the model will be used with a security camera, test it with security camera images.

---

# 17. Understanding `best.pt` and `last.pt`

After training, YOLO usually saves important model files.

Two common files are:

```text
best.pt
last.pt
```

`best.pt` usually represents the best model according to the validation results.

`last.pt` represents the model from the final training epoch.

In most cases, `best.pt` is the preferred model for testing and deployment.

---

# 18. A Recommended Training Workflow

A good workflow is:

```text
1. Prepare Dataset
        ↓
2. Check Labels
        ↓
3. Create data.yaml
        ↓
4. Select a Small Model
        ↓
5. Train a Baseline
        ↓
6. Analyze Metrics
        ↓
7. Identify Problems
        ↓
8. Improve Dataset or Configuration
        ↓
9. Train Again
        ↓
10. Compare Results
```

Always make decisions based on real results.

Do not change many parameters at the same time.

Change one important parameter and measure the result.

---

# Final Recommendations

For your first experiment:

* Start with a small model
* Use pretrained weights
* Use a clean dataset
* Create a baseline
* Save your training results
* Compare experiments

Training Machine Learning models is an iterative process.

A good model is usually created through:

```text
Training
    ↓
Analysis
    ↓
Improvement
    ↓
Training Again
```

---

# Next Chapter

In the next chapter, we will learn how to deploy a trained YOLO26 model.

We will run the model with:

* Images
* Videos
* Webcams
* IP cameras
* Real-time video streams
* Multiple camera streams
