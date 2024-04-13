# depthai-human-fall-detection

Human Fall Detection using DepthAI and OAK-D PRO Hardware with MoveNet Pose Estimation

### Reference

* [MoveNet](https://www.tensorflow.org/hub/tutorials/movenet) - Ultra fast and accurate pose detection model, from '
  TensorFlow'.
* [depthai_movenet](https://github.com/geaxgx/depthai_movenet) - MoveNet Single Pose tracking on DepthAI, from 'geaxgx'.
* [A Framework for Fall Detection Based on OpenPose Skeleton and LSTM/GRU Models](https://doi.org/10.3390/app11010329) -
  Lin, C.-B., et al

### Get Started

##### 1. Install the python packages with the following command:

```
python3 -m pip install -r requirements.txt
```

##### 2. MoveNet models

MoveNet is an ultra-fast and accurate model that detects 17 keypoints of a body. The model is offered on TF.Hub with two
variants, Lightning and Thunder.
Lightning is intended for latency-critical applications, and Thunder is intended for applications that require high
accuracy.
Both models run faster than real time (30+ FPS) on most modern desktops and laptops, which proves crucial for live
fitness, health, and wellness applications.

i. Download the MoveNet models from https://www.kaggle.com/models/google/movenet/tensorFlow2/

##### 3. UR Fall Detection Dataset

i. UR Fall Detection Dataset is from http://fenix.ur.edu.pl/~mkepski/ds/

ii. Download the data folder and create a new `data/dataset` folder in the project

iii. Unzip and separate the data in the two different folders `fall` and `not_fall` in the `data/dataset` folder

##### 4. Extract data

i. Before running the extract_data.py, make sure the OAK-D Pro device is connected.

```
python3 extract_data.py
```

##### 5. Model Fitting

```
python3 model_fit.py
```

The model will be saved in the folder `model_weights`.

