# -*- coding: UTF-8 -*-
import cv2 as cv
import argparse
import numpy as np
import time
from utils import choose_run_mode, load_pretrain_model, set_video_writer
from Pose.pose_visualizer import TfPoseVisualizer
from Action.recognizer import load_action_premodel, framewise_recognize
from send_message import sent_message
from datetime import datetime

# python main.py --video=data/1.avi
parser = argparse.ArgumentParser(description='Action Recognition by OpenPose')
parser.add_argument('--video',default='test8.mp4', help='Path to video file.')#default='t.mp4'
args = parser.parse_args()

# 导入相关模型
estimator = load_pretrain_model('VGG_origin')#mobilenet_thin,VGG_origin
action_classifier = load_action_premodel('Action/framewise_recognition_2.h5')#framewise_recognition_under_scene,framewise_recognition2




# 参数初始化
realtime_fps = '0.0000'
start_time = time.time()
fps_interval = 1
fps_count = 0
run_timer = 0
frame_count = 0

# 读写视频文件（仅测试过webcam输入）
cap = choose_run_mode(args)
video_writer = set_video_writer(cap, write_fps=int(7.0))
wr_count = 0

# # 保存关节数据的txt文件，用于训练过程(for training)
f = open('origin_data.txt', 'a+')

while cv.waitKey(1) < 0:
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    has_frame, show = cap.read()
    if has_frame:
        fps_count += 1
        frame_count += 1

        # pose estimation
        humans = estimator.inference(show)
        # get pose info
        pose = TfPoseVisualizer.draw_pose_rgb(show, humans)  # return frame, joints, bboxes, xcenter 

        # recognize the action framewise
        show,count = framewise_recognize(pose, action_classifier)
        height, width = show.shape[:2]
        # check frame
        if count == 0:
            wr_count = 0
        elif count == 1:
            wr_count += count
        # send waring
        if wr_count == 5:
            sent_message("Cảnh báo khẩn cấp. Người thân của bạn gặp sự cố !!!")
            wr_count = 0
        # 显示实时FPS值
        if (time.time() - start_time) > fps_interval:
            # 计算这个interval过程中的帧数，若interval为1秒，则为FPS
            realtime_fps = fps_count / (time.time() - start_time)
            fps_count = 0  # 帧数清零
            start_time = time.time()
        fps_label = 'FPS:{0:.2f}'.format(realtime_fps)
        cv.putText(show, fps_label, (width-160, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # 显示检测到的人数
        num_label = "Human: {0}".format(len(humans))
        cv.putText(show, num_label, (5, height-45), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # 显示目前的运行时长及总帧数
        if frame_count == 1:
            run_timer = time.time()
        run_time = time.time() - run_timer
        time_frame_label = '[Time:{0} | Frame:{1}]'.format(current_time, frame_count)
        cv.putText(show, time_frame_label, (5, height-15), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        video_writer.write(show)
        cv.imshow('Action Recognition based on OpenPose', show)

        # 采集数据，用于训练过程(for training)
        joints_norm_per_frame = np.array(pose[-1]).astype(np.str)
        #print(joints_norm_per_frame)
        f.write(' '.join(joints_norm_per_frame))
        f.write('\n')
        
        # f.close()
        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break



print('done@!')
video_writer.release()
cap.release()
# f.close()
