from detectron2.engine import DefaultPredictor

import os
import pickle


from utils import *

cfg_save_path = "IS_cfg.pickle"

with open(cfg_save_path, 'rb') as f:
    cfg = pickle.load(f)

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

predictor = DefaultPredictor(cfg)

image_path = "test_cats/44.tabby-cat-looking-up.jpg"
#videoPath = "test_cats/peco_beatbox_2.mp4"

on_image(image_path, predictor)
#on_video(videoPath, predictor)