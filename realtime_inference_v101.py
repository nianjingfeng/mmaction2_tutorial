#import python library
import mediapipe as mp
import cv2
import numpy as np
import torch
import mmcv
import mmengine
from mmengine import DictAction
from mmaction.apis import init_recognizer
# from mmcv.parallel import collate,scatter
from mmengine.dataset import Compose
import time
import argparse
from operator import itemgetter
import queue
import threading
#set the parameter
def parse_args():
    parser = argparse.ArgumentParser(description='posec3d demo')
    #set the path of config file
    parser.add_argument(
        '--config',
        default=('./data/slowonly_r50_u48_240e_ntu120_xsub_keypoint_v101.py'),
        help='skeleton model config file path')
    #set the path of checkpoint
    parser.add_argument(
        '--checkpoint',
        default=('./data/best_top1_acc_epoch_160.pth'),
        help='skeleton model checkpoint file/url')
    #set the path of label map
    parser.add_argument(
        '--label-map',
        default='./data/label_map.txt',
        help='label map file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    args = parser.parse_args()
    return args

#function to inference the data from function collect_data
def inference():
    #init the parameter
    args = parse_args()
    label_map = [x.strip() for x in open(args.label_map).readlines()]
    #init the model
    config = mmengine.Config.fromfile(args.config)
    config.merge_from_dict(args.cfg_options)
    model = init_recognizer(config, args.checkpoint, args.device)
    # config.model.cls_head.num_classes = len(label_map)
    num_classes = len(label_map)
    #init data pipeline
    device = next(model.parameters()).device
    cfg = model.cfg
    test_pipeline = cfg.test_pipeline
    test_pipeline = Compose(test_pipeline)
    while True:
        if anno_stack.empty() != True:
            #take the data from collect_data
            anno_data = anno_stack.get()
            #it need to be set as null list, or it will show error code
            anno_data['imgs'] = []
            #avoid to get the null data
            try:
                get_data_time = time.time()
                data = test_pipeline(anno_data)
                get_result_time = time.time()
                #the larger the samples_per_gpu set, the slower the inference, but more accurate
                data = collate([data],samples_per_gpu=1)
                data = scatter(data,[device])[0]
            except:
                pass
            #inference the result
            with torch.no_grad():
                scores = model(return_loss=False,**data)[0]
            score_tuples = tuple(zip(range(num_classes), scores))
            score_sorted = sorted(score_tuples, key=itemgetter(1), reverse=True)
            results = score_sorted[:5]
            action = label_map[results[0][0]]
            
            action_stack.put(action)
            print('Process_time:',get_result_time-get_data_time,'\n','Action:',action)
            if anno_stack.qsize()>40:
                anno_stack.queue.clear()


def collect_data():
    #initial the function 
    test_video = '/home/will/Desktop/Thesis/Dataset/5G_assembling_dataset/video/2202.mp4'
    cap = cv2.VideoCapture(test_video)
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    height = 1080
    width = 1920
    #window size is 30 frame
    input_data = dict(frame_dir='',label=-1,img_shape=(height, width),original_shape=(height, width),start_index=0,modality='Pose',total_frames=30)
    #mediapipe has no confidence score, so it was set as 0.95 here
    input_data['keypoint_score'] = np.array([[[0.95 for j in range(17)] for k in range(30)]])
    #initial the null data container
    keypoints = [[[[0 for i in range(2)] for j in range(17)] for k in range(30)]]
    #initial the starting action
    action = 'Static'
    while cap.isOpened():
        ret,frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            hand_result = hands.process(frame_rgb)
            keypoint = []
            if hand_result.multi_hand_landmarks:
                for lms in hand_result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame,lms,mp_hands.HAND_CONNECTIONS)
                    for i,lm in enumerate(lms.landmark):
                        keypoint.append([round(lm.x*1920,4),round(lm.y*1080,4)])
            keypoint = [keypoint[i] for i in [0,1,4,5,8,9,12,16,20,21,22,25,29,30,33,37,41]]
            keypoints[0].pop(0)
            keypoints[0].append(keypoint)
            input_data['keypoint'] = np.array(keypoints)
            anno_stack.put(input_data)
            if action_stack.empty()!=True:
                action = action_stack.get()
                if action_stack.qsize()>20:
                    action_stack.queue.clear()
            cv2.putText(frame, 'Action : ' + action, (20, 80), cv2.FONT_HERSHEY_DUPLEX,3, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.imshow('frame',frame)
        if cv2.waitKey(1) == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    #stack can be global variable, so you dont need to define in function
    anno_stack = queue.LifoQueue()
    action_stack = queue.LifoQueue()
    #parallel working by threading
    p1 = threading.Thread(target=collect_data)
    p2 = threading.Thread(target=inference)
    p1.start()
    p2.start()
