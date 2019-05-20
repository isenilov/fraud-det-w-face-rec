import glob
import os
import imageio
import io
import numpy as np
import matplotlib.pyplot as plt
import requests
import time


VIDEOS_DIR = 'train'
FACE_DET_URI = 'http://facedet:5000/model/predict'
OBJ_DET_URI = 'http://objdet:5000/model/predict'
EACH_N_FRAME = 5  # take only every Nth frame
NO_FRAMES_W_DET = 5  # if more frames contain >2faces/persons - label as detection

def get_response(im: np.array, uri: str) -> dict:
    '''
    makes a POST request to uri with an image
    :param im: image as numpy array
    :param uri: uri of detection service
    :return: dictionary with predictions
    '''
    return requests.post(uri,
                         headers={"accept": "application/json"},
                         files={'image': im, 'type': 'image/jpeg'}) \
                   .json()


def draw_bbox(im: np.array, bboxes: list, color: np.array, normalized: bool = False) -> np.array:
    '''
    Draws bounding box on a give image
    :param im: input image
    :param bboxes: list of lists [xmin, ymin, xmax, ymax]
    :param color: color of the line of bbox
    :param normalized: if True, bboxes considered normalized
    :return:
    '''
    height = im.shape[0]
    width = im.shape[1]
    if len(bboxes) > 0:
        for bbox in bboxes:
            if normalized:
                bbox[0] = bbox[0] * height - 1
                bbox[2] = bbox[2] * height - 1
                bbox[1] = bbox[1] * width - 1
                bbox[3] = bbox[3] * width - 1
                bbox = [bbox[1], bbox[0], bbox[3], bbox[2]]
            else:
                bbox[2] = min(bbox[2], width - 1)
                bbox[3] = min(bbox[3], height - 1)
            bbox = np.array(bbox, dtype=int)
            im[bbox[1], bbox[0]:bbox[2]] = color
            im[bbox[1]:bbox[3], bbox[0]] = color
            im[bbox[3], bbox[0]:bbox[2]] = color
            im[bbox[1]:bbox[3], bbox[2]] = color
    return im


if __name__ == '__main__':
    time.sleep(20)  # wait for other services to start
    detected_labels = dict()
    for filename in glob.iglob('**/*.mp4', recursive=True):
        if filename.endswith('bboxes.mp4'):  # skip already detected
            continue
        detections = 0
        print(f'Processing {filename} ...')
        reader = imageio.get_reader(filename, 'ffmpeg')
        fps = reader.get_meta_data()['fps']
        # new_path = os.path.splitext(filename)[0] + '_bboxes.mp4'
        new_path = os.path.join(VIDEOS_DIR, os.path.basename(filename)[:-4]) + '_bboxes.mp4'
        print(f'Writing result to {new_path} ...')
        writer = imageio.get_writer(new_path, fps=fps / EACH_N_FRAME)
        for i, im in enumerate(reader):
            if i % EACH_N_FRAME != 0:  # use only each Nth frame to speedup
                continue
            buf = io.BytesIO()
            plt.imsave(buf, im, format='jpeg')
            image_data = buf.getvalue()
            objects = get_response(image_data, OBJ_DET_URI)
            obj_bboxes = [el['detection_box'] for el in objects['predictions']]
            faces = get_response(image_data, FACE_DET_URI)
            face_bboxes = [el['detection_box'] for el in faces['predictions']]
            if len(obj_bboxes) > 1 or len(face_bboxes) > 1:
                detections += 1
            im = draw_bbox(im, obj_bboxes, np.array([0, 255, 0], dtype=np.uint8), normalized=True)
            im = draw_bbox(im, face_bboxes, np.array([0, 0, 255], dtype=np.uint8))
            writer.append_data(im)
            if detections > NO_FRAMES_W_DET:
                detected_labels[os.path.basename(filename)] = 1
            else:
                detected_labels[os.path.basename(filename)] = 0
        writer.close()
    with open(os.path.join(VIDEOS_DIR, 'pred_labels.txt'), 'w') as f:
        for elem in detected_labels.items():
            f.write(f'{elem[0][:-4]}\t{elem[1]}\n')
