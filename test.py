from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
import sys
# import caffe
import numpy as np
import cv2
import scipy.io
import copy
import os
import torch.utils.data
from core import model1
from dataloader.LFW_loader import LFW
from config import LFW_DATA_DIR
import argparse
import time

from modules.models import RetinaFaceModel
from modules.utils import (set_memory_growth, load_yaml, draw_bbox_landm,
                           pad_input_image, recover_pad_output)


flags.DEFINE_string('cfg_path', './configs/retinaface_res50.yaml',
                    'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')
flags.DEFINE_string('img_path', '', 'path to input image')
flags.DEFINE_boolean('webcam', False, 'get image source from webcam or not')
flags.DEFINE_float('iou_th', 0.4, 'iou threshold for nms')
flags.DEFINE_float('score_th', 0.5, 'score threshold for nms')
flags.DEFINE_float('down_scale_factor', 1.0, 'down-scale factor for inputs')


def main(_argv):
    # init
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()

    cfg = load_yaml(FLAGS.cfg_path)

    # define network
    model = RetinaFaceModel(cfg, training=False, iou_th=FLAGS.iou_th,
                            score_th=FLAGS.score_th)

    # load checkpoint
    checkpoint_dir = './checkpoints/' + cfg['sub_name']
    checkpoint = tf.train.Checkpoint(model=model)
    if tf.train.latest_checkpoint(checkpoint_dir):
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        print("[*] load ckpt from {}.".format(
            tf.train.latest_checkpoint(checkpoint_dir)))
    else:
        print("[*] Cannot find ckpt from {}.".format(checkpoint_dir))
        exit()

    img_sid = cv2.imread("./out_sid.jpg")
    img_height_raw, img_width_raw, _ = img_sid.shape
    img_sid = cv2.resize(img_sid, (112, 112))
    img_sid = (img_sid - 127.5) / 128.0
    img_sid = img_sid.transpose(2, 0, 1)
    img_sid = torch.from_numpy(img_sid).float()

    net = model1.ShuffleFaceNet()
    ckpt = torch.load('./model1/best/060.ckpt')
    net.load_state_dict(ckpt['net_state_dict'])
    net.eval()
    name_str = "null"

    if not FLAGS.webcam:
        if not os.path.exists(FLAGS.img_path):
            print(f"cannot find image path from {FLAGS.img_path}")
            exit()

        print("[*] Processing on single image {}".format(FLAGS.img_path))

        img_raw = cv2.imread(FLAGS.img_path)
        img_height_raw, img_width_raw, _ = img_raw.shape
        img = np.float32(img_raw.copy())

        if FLAGS.down_scale_factor < 1.0:
            img = cv2.resize(img, (0, 0), fx=FLAGS.down_scale_factor,
                             fy=FLAGS.down_scale_factor,
                             interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # pad input image to avoid unmatched shape problem
        img, pad_params = pad_input_image(img, max_steps=max(cfg['steps']))

        # run model
        outputs = model(img[np.newaxis, ...]).numpy()

        # recover padding effect
        outputs = recover_pad_output(outputs, pad_params)

        # draw and save results
        save_img_path = os.path.join('out_' + os.path.basename(FLAGS.img_path))
        for prior_index in range(len(outputs)):
            draw_bbox_landm(img_raw, outputs[prior_index], img_height_raw,
                            img_width_raw)
            cv2.imwrite(save_img_path, img_raw)
        print(f"[*] save result at {save_img_path}")

    else:
        #Webcam Detection
        cam = cv2.VideoCapture(0)
        start_time = time.time()
        while True:
            _, frame = cam.read()
            if frame is None:
                print("no cam input")


            frame_height, frame_width, _ = frame.shape
            img = np.float32(frame.copy())
            if FLAGS.down_scale_factor < 1.0:
                img = cv2.resize(img, (0, 0), fx=FLAGS.down_scale_factor,
                                 fy=FLAGS.down_scale_factor,
                                 interpolation=cv2.INTER_LINEAR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # pad input image to avoid unmatched shape problem
            img, pad_params = pad_input_image(img, max_steps=max(cfg['steps']))

            # run model
            outputs = model(img[np.newaxis, ...]).numpy()

            # recover padding effect
            outputs = recover_pad_output(outputs, pad_params)
            # draw results
            for prior_index in range(len(outputs)):
                draw_bbox_landm(frame, outputs[prior_index], frame_height,
                                frame_width)
                ann1 = outputs[prior_index]
                x1, y1, x2, y2 = int(ann1[0] * frame_width), int(ann1[1] * frame_height), \
                                 int(ann1[2] * frame_width), int(ann1[3] * frame_height)
                frame1 = frame[y1:y2, x1:x2]
                frame1 = cv2.resize(frame1, (112, 112))
                frame1 = (frame1 - 127.5) / 128.0
                frame1 = frame1.transpose(2, 0, 1)
                frame1 = torch.from_numpy(frame1).float()
                recogin = torch.stack([img_sid, frame1], dim=0)
                recog1 = net(recogin).data.cpu().numpy()
                dist = 0
                dist = np.linalg.norm(recog1[1]) - np.linalg.norm(recog1[0])
                print(dist)
                if(dist <= 5200 and dist >= 3800):
                    name_str = "Siddhartha"
                else:
                    name_str = "null"
            # calculate fps
            fps_str = "FPS: %.2f" % (1 / (time.time() - start_time))
            #start_time = time.time()
            total_str = name_str + fps_str
            cv2.putText(frame, total_str, (25, 25),
                        cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 0), 2)

            # show frame
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                exit()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
