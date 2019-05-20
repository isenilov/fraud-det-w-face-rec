This is an attempt to build a fraud detection app based on face detection.
More details may be found in `REPORT.md`

The app is running inside docker containers with following command:

`docker-compose build && docker-compose up`

Before running the app, videos from `train.zip` should be unpacked to a
`train` subdir in the working dir.

Some constants may be tuned in the beginning of the `main.py` file

As a result of the program execution, new video files with `bboxes` suffix
will be put into `train/` dir along with `pred_labels.txt` file containing
predictions for each processed file.