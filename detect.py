
'''
yolov5 model for detection
and roboflow to train our model

----------------------------------------------------------------
To run this file:
#*          python detect.py
It will start the webcam and do the detection
----------------------------------------------------------------
To use ESP32 or any other IP camera
#*          python detect.py --source <IP_ADDRESS>
Replace the <IP_ADDRESS> with the IP address of the ESP32 Camera
----------------------------------------------------------------

NOTE: best.pt weights file has been added by default during the run command
'''

# importing libraries
import argparse
import os
import platform
import sys
from pathlib import Path
import torch
from collections import Counter


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import LoadStreams
from utils.general import (LOGGER, Profile, check_img_size, check_imshow, check_requirements, cv2,
                             non_max_suppression, print_args, scale_boxes)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, smart_inference_mode


customer_cart = None    

@smart_inference_mode()
def run(
        weights=ROOT / 'best.pt',  # model path
        source= 0,  # 0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.60,  # confidence threshold
        iou_thres=0.65,  # NMS IOU threshold. YOLO uses Non-Maximal Suppression (NMS) to only keep the best bounding box
        max_det=20,  # maximum detections per image
        device='',  # this defines the device that will be used by the model. Code will automatically detect the device
        view_img=False,  # show results
        classes=None,  # this is the class name file
        visualize=False,  # visualize features
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        exefrom= 'python',
        
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=0.5,  # video frame-rate stride
        augment=False,  # augmented inference
        agnostic_nms=False  # class-agnostic NMS
):
    source = str(source)
    is_url = source.lower().startswith(('http://', 'https://'))
    webcam = source.isnumeric() or (is_url)

    # original_frame = None

    # Load model
    device = select_device(device) # is selects the device (cpu or gpu) to run the detection model
    
    # ------------------- defining the model -------------------
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)   
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size


    # Dataloader
    bs = 1  # batch_size
    if webcam:      # webcam or ESP32 or any ip camera
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)

    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    #* Reading each frame of the video and sending that frame to the model for detection
    for path, im, im0s, vid_cap, s in dataset:

        with dt[0]:
            im = torch.from_numpy(im).to(model.device)

            #  doing some preprocessing on the video frame
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # converting pixel values from (0 - 255) to (0.0 - 1.0)
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = False
            pred = model(im, augment=augment, visualize=visualize)  # getting the predictions from the video frame

        # NMS
        with dt[2]:
            # Non Maximum Suppression is a computer vision method that selects a single entity out of many overlapping entities
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        
        det_item_list = []  # an array that will store the number of items detected from the shopping cart
        
        for i, det in enumerate(pred):  # per image
            seen += 1
            # if webcam:
            p, im0, frame = path[i], im0s[i].copy(), dataset.count
            original_frame = im0
            s += f'{i}: '

            p = Path(p)  # to Path
            s += '%gx%g ' % im.shape[2:]  # print string

            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')

                    annotator.box_label(xyxy, label, color=colors(c, True)) # boxes are drawn at this line
                    det_item_list.append(label.split(" ")[0])


            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])


            # At this point, we have done the detection and for the current video frame, we have got the detected items in an array named "det_item_list"
                
                # from the "det_item_list" we are counting the number of each similar detected item
                customer_cart = Counter(det_item_list)

                # below lines are sending the detection output to app-py.js file
                if len(customer_cart) > 0: print(str(customer_cart), flush=True)    # if there is some detection
                else: print('Empty Cart', flush=True)       # if there is no detection
                
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):
                    print("END-END-END", flush=True)
                    return


cv2.destroyAllWindows()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'best.pt', help='model path or triton URL')
    parser.add_argument('--source', default=0, help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--exefrom', type=str, default='python', help='detect.py is executed from node or direct?')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


opt = parse_opt()
main(opt)
