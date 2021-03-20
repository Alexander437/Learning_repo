import json
import time
import sys
sys.path.append('/usr/lib/python3.6/dist-packages')

import numpy as np
import torch
import cv2
import torchvision.transforms as transforms
import PIL.Image

from torch2trt import TRTModule
from trt_pose.parse_objects import ParseObjects

def coco_category_to_topology(coco_category):
    """Gets topology tensor from a COCO category
    """
    skeleton = coco_category['skeleton']
    K = len(skeleton)
    topology = torch.zeros((K, 4)).int()
    for k in range(K):
        topology[k][0] = 2 * k
        topology[k][1] = 2 * k + 1
        topology[k][2] = skeleton[k][0] - 1
        topology[k][3] = skeleton[k][1] - 1
    return topology

class VideoUtils(object):

    def __init__(self):
        self.mode = False
        with open('human_pose.json', 'r') as f:
            human_pose = json.load(f)
            self.topology = coco_category_to_topology(human_pose)
        self.OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
        self.WIDTH = 224
        self.HEIGHT = 224
        print('\nCreate torch tensor in cuda...')
        self.data = torch.zeros((1, 3, self.HEIGHT, self.WIDTH)).cuda()
        print('Done')
        print('Loading tensorRT-optimized model...')
        self.model_trt = TRTModule()
        self.model_trt.load_state_dict(torch.load(self.OPTIMIZED_MODEL))
        print('Done')
        self.parse_objects = ParseObjects(self.topology)
        self.draw_objects = DrawObjects(self.topology)
        t0 = time.time()
        torch.cuda.current_stream().synchronize()
        for i in range(50):
            y = self.model_trt(self.data)
        torch.cuda.current_stream().synchronize()
        t1 = time.time()

    def run_video(self, callback1, callback2, camera_port, show_video):
        cap = cv2.VideoCapture(camera_port)
        automatic_mode = np.zeros_like((740,480,3), np.uint8)
        cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
        cv2.createTrackbar('level','Video',15,255, self.nothing)
        count = 19
        temp_th = None
        while (cap.isOpened()):
            ret, frame = cap.read()
            count += 1
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            l = cv2.getTrackbarPos('level','Video')
            ret,th = cv2.threshold(gray,255-l,255,0)
            if count==20:
                temp_th = th
                count = 0
            key = cv2.waitKey(1) & 0xFF

            if self.mode:
                person_piaks, optic_piaks = self.find_objects(frame, th, temp_th)
                cv2.rectangle(frame, (12,5), (180,25), (45,45,40), -1)
                cv2.rectangle(frame, (5,440), (160,475), (45,45,40), -1)
                cv2.putText(frame, 'Automatic mode', (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 170), 1)
                cv2.putText(frame, '(1) - automatic mode', (9, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,170), 1)  
                cv2.putText(frame, '(2) - manual mode', (9, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,170), 1)
                if person_piaks is not None:
                    (x, y, w, h) = person_piaks
                    callback1((x, y, w, h), frame)
                elif optic_piaks is not None:
                    (x, y, w, h) = optic_piaks
                    callback1((x, y, w, h), frame)
            else:
                cv2.rectangle(frame, (15,5), (265,25), (45,45,40), -1)
                cv2.rectangle(frame, (5,440), (160,475), (45,45,40), -1)
                cv2.rectangle(frame, (545,395), (630,475), (45,45,40), -1)
                cv2.putText(frame, 'Manual mode & calibrate', (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.putText(frame, 'W - Up', (550, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)  
                cv2.putText(frame, 'S - Down', (550, 425), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
                cv2.putText(frame, 'A - Left', (550, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
                cv2.putText(frame, 'D - Riht', (550, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
                cv2.putText(frame, 'Enter - Fire', (550, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,190,190), 1)
                cv2.putText(frame, '(1) - automatic mode', (9, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,190,190), 1)  
                cv2.putText(frame, '(2) - manual mode', (9, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,190,190), 1)

                if key!=255:
                    callback2(key)

            if show_video:
                cv2.imshow('Setup level', th)
                cv2.imshow('Video', frame)

            if (key == ord("q")) or (key == 27):
                break
            if key == ord("1"):
                self.mode = True
            if key == ord("2"):
                self.mode = False

        cap.release()
        cv2.destroyAllWindows()
        raise KeyboardInterrupt

    def find_objects(self, frame, th, temp_th):
            person_piaks = self.find_person(frame)
            optic_piaks = self.find_optic(frame, th, temp_th)
            return (person_piaks, optic_piaks)

    def find_person(self, frame):
            data = self.transform(frame)
            cmap, paf = self.model_trt(data)
            cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
            counts, objects, peaks = self.parse_objects(cmap, paf)
            person_piaks = self.draw_objects(frame, counts, objects, peaks)
            return(person_piaks)

    def find_optic(self, frame, th, temp_th):
            delta = cv2.absdiff(temp_th, th)
            delta = cv2.erode(delta,np.ones((8,8),np.uint8),iterations = 1)
            delta = cv2.morphologyEx(delta, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
            c = self.get_best_contour(delta, 25)
            if c is not None:
                optic_piaks = cv2.boundingRect(c)
                (x0, y0, w0, h0) = optic_piaks
                cv2.rectangle(frame, (x0, y0), (x0 + w0, y0 + h0), (255, 0, 152), 2)
                centroid = tuple(map(int, ( x0 + w0 / 2, y0 + h0 / 2)))
                cv2.drawMarker(frame, centroid, (0, 0, 255), 0, 30, 3)
                cv2.putText(frame, 'optics detected', (x0+w0//2-70, y0+h0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 152), 2)
            else:
                optic_piaks = None
            return(optic_piaks)

    def transform(self, image):
        mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
        std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
        device = torch.device('cuda')
        image = cv2.resize(image, (self.HEIGHT, self.WIDTH))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = PIL.Image.fromarray(image)
        image = transforms.functional.to_tensor(image).to(device)
        image.sub_(mean[:, None, None]).div_(std[:, None, None])
        return image[None, ...]

    def get_best_contour(self, imgmask, threshold):
        contours, hierarchy = cv2.findContours(imgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_area = threshold
        best_cnt = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > best_area:
                best_area = area
                best_cnt = cnt
        return best_cnt

    def nothing(self, x):
        pass

class DrawObjects(object):
    
    def __init__(self, topology):
        self.topology = topology
        self.color = (0,122,230) 
        
    def __call__(self, image, object_counts, objects, normalized_peaks):
        topology = self.topology
        height = image.shape[0]
        width = image.shape[1]
        x_peaks = []
        y_peaks = []
        K = topology.shape[0]
        count = int(object_counts[0])
        K = topology.shape[0]
        for i in range(count):
            obj = objects[0][i]
            C = obj.shape[0]
            for j in range(C):
                k = int(obj[j])
                if k >= 0:
                    peak = normalized_peaks[0][j][k]
                    x = round(float(peak[1]) * width)
                    y = round(float(peak[0]) * height)
                    cv2.circle(image, (x, y), 3, self.color, 2)
            for k in range(K):
                c_a = topology[k][2]
                c_b = topology[k][3]
                if obj[c_a] >= 0 and obj[c_b] >= 0:
                    peak0 = normalized_peaks[0][c_a][obj[c_a]]
                    peak1 = normalized_peaks[0][c_b][obj[c_b]]
                    x0 = round(float(peak0[1]) * width)
                    y0 = round(float(peak0[0]) * height)
                    x1 = round(float(peak1[1]) * width)
                    y1 = round(float(peak1[0]) * height)
                    x_peaks.append(x0)
                    x_peaks.append(x1)
                    y_peaks.append(y0)
                    y_peaks.append(y1)
                    cv2.line(image, (x0, y0), (x1, y1), self.color, 2)
        try:
            (x, y, w, h) = (min(x_peaks), min(y_peaks), \
            max(x_peaks)-min(x_peaks), max(y_peaks)-min(y_peaks))
            (x, y, w, h) = (x-30, y-30, w+60, h+60)                    
            cv2.rectangle(image, (x, y), (x+w, y+h), self.color, 3)
            centroid = tuple(map(int, ( x + w / 2, y + h / 2)))
            cv2.drawMarker(image, centroid, (0, 0, 255), 0, 35, 4)
            cv2.putText(image, 'person detected', (x+w//2-80, y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color, 2)
            return (x, y, w, h)
        except: return None




