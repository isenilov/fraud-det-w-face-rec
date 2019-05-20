# Fraud detection with face recognition on video

## Problem statement

Given the data, build a system that detects presence of
more than one person on the recording.

## Data description

There are 15 video files with or without people.
The videos have sound recoding.

It is always a good idea to look at the data before making any attempts
to model it. One of the surprising things was that
some videos do not have persons or clearly one person on them 
but marked as having more than 1
person, probably due to multiple voices in audio track


## Solution

The idea is to apply person and face detection models to series of frames
extracted from the videos. Very simple heuristic is proposed: `If number of detected
"person" objects is more than 1 OR number of detected faces is more than 1
on more that N frames`.

The solution is dockerized script that takes `*.mp4` files from `train/`
directory. Number of dependencies for main script kept as low as possible
(for example there is no need for heavy OpenCV)

### Limitations and simplifications

Due to the limited amount of time for completion of the task, some simplifications
were made:

* The program can be parallelized as processing of each video and each frame
is independent
* Of course, the architecture of the app is sub-optimal. Using HTTP requests
and separate docker containers are justified only by speed of development
and easiness of setup the environment
* The app was not designed to work in real time and was not optimized
in terms of speed
* There are no unit/integration tests

### Selected models

Pretrained model were used due to the following reasons:
* Lack of training data.
In order to train neural network from scratch, thousands of images/videos are needed
* Lack of compute resources for training
* Lack of time for implementing training code and setting up corresponding
environment and infrastructure
* Comparing to quick-n-dirty custom prototypes,
pretrained models are usually more stable and reliable

The models are::
1. [Object Detector](https://hub.docker.com/r/codait/max-object-detector)
with `ssd_mobilenet_v1_coco_2017_11_17 TensorFlow model trained on MobileNet`
under the hood that can detect 80 types of objects
2. [Face detector](https://hub.docker.com/r/codait/max-facial-recognizer) 
with [FaceNet](https://arxiv.org/abs/1503.03832)
under the hood which return face embeddings along with faces' bounding
boxes


### Models' interfaces

Swagger UI for models interfaces may be accessed by following:
`http://localhost:5001/` for Face Detector and 
`http://localhost:5002/` for Object Detector.
The Object Detector also has demo app available at 
`http://localhost:5002/app`

## Results

As the dataset is only 15 videos and due to the fact that no models were
trained during the development process, evaluation was performed on
all available videos in `train` directory.

Comparison of predicted labels with true labels is in the table below.

| File      | True label | Predicted label |
|-----------|------------|-----------------|
| joonatan1 | 0          | 1               |
| joonatan2 | 0          | 0               |
| taivo6    | 0          | 0               |
| taivo7    | 0          | 1               |
| brett4    | 0          | 0               |
| brett3    | 1          | 1               |
| brett6    | 1          | 0               |
| taivo10   | 1          | 0               |
| taivo1    | 1          | 0               |
| brett7    | 1          | 0               |
| brett5    | 1          | 0               |
| cantina1  | 1          | 1               |
| taivo4    | 0          | 0               |
| cantina4  | 1          | 0               |
| cantina2  | 0          | 0               |

Some metrics:

TP = 2, FP = 2, TN = 5, FN = 6

Precision = 0.5,
Recall = 0.25

Let's look at mistakes the app made:

FPs: `joonatan1`, `taivo7` - photo on the document was detected as a second face:
some tuning of parameters and heuristics (like comparison of sizes of faces'
bounding boxes) may help to overcome the problem

FNs:

`brett6` - person on the background was detected only within very short
period of time. Changing or retraining the model might help

`taivo10` - person standing behind was not detected at all. Probably,
the used model cannot detect it so new labeling and retraining is needed

`taivo1`, `brett7`, `brett5`, `cantina4` - probably labeled positive
because of background voices. Audio-based model might help.

I am not
considering the fact that there are wrong/fake documents preseted on the videos
as it is not part of the task.

## Conclusion, further work, ideas, TODOs...

Even though the results are pretty low, the app may serve 
as a baseline for further development. Unfortunately, timeframes
set by myself (<=12 hours) didn't allow to test all the hypothesis.

However, there are several ideas and directions for further work on the topic
which may increase performance on the given data
(especially audio-based methods): 

* Take into account sequential nature of video using recurrent
or temporal convolutional networks
* Use audio track to detect multiple speakers
* Use speakers disambiguation model, like [this](https://arxiv.org/abs/1708.02840)
to detect if more than one speaker is present. It may also use lips movement
detection for cases where person does not speak but there is one background
voice
* The [FaceNet](https://arxiv.org/abs/1503.03832) model inside the face detector
service provides face descriptors/embeddings along with bounding boxes.
It is possible to use them to find similarity between face 
on the document and face on the video (which would probably require
prior classification of the face into "document" or "live face" category)
* Detect gaze and direction of sight of the person on the video
with model like [this](https://link.springer.com/chapter/10.1007/978-3-540-30499-9_103)
and some heuristic to evaluate how long and how many times the person
does not look into the camera
* Having enough data, try to train end-to-end model on both video and audio
features
