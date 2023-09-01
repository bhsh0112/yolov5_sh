# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 segmentation inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python segment/predict.py --weights yolov5s-seg.pt --source 0                               # webcam
                                                                  img.jpg                         # image
                                                                  vid.mp4                         # video
                                                                  screen                          # screenshot
                                                                  path/                           # directory
                                                                  list.txt                        # list of images
                                                                  list.streams                    # list of streams
                                                                  'path/*.jpg'                    # glob
                                                                  'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                                  'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python segment/predict.py --weights yolov5s-seg.pt                 # PyTorch
                                          yolov5s-seg.torchscript        # TorchScript
                                          yolov5s-seg.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                          yolov5s-seg_openvino_model     # OpenVINO
                                          yolov5s-seg.engine             # TensorRT
                                          yolov5s-seg.mlmodel            # CoreML (macOS-only)
                                          yolov5s-seg_saved_model        # TensorFlow SavedModel
                                          yolov5s-seg.pb                 # TensorFlow GraphDef
                                          yolov5s-seg.tflite             # TensorFlow Lite
                                          yolov5s-seg_edgetpu.tflite     # TensorFlow Edge TPU
                                          yolov5s-seg_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path
import rospy
import torch
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, scale_segments,
                           strip_optimizer,xyxy2xywh)
from utils.segment.general import masks2segments, process_mask, process_mask_native
from utils.torch_utils import select_device, smart_inference_mode
from utils.plots import plot_one_box
from PIL import Image as Image_PIL
from geometry_msgs.msg import Pose

fx = 913.43591308
fy = 911.63287353
ppx = 641.94738769
ppy = 359.5092773
@smart_inference_mode()



def exit(signum, frame):
    print('You choose to stop me.')
    global stop
    stop = True
    exit()

stop = False

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
def compute_depth(depth_img, bbox):
    x1, y1, x2, y2 = bbox

    O_x1 = max(min(int(x1 + (x2 - x1) / 2.) - 3, IMAGE_WIDTH - 1), 0)
    O_x2 = max(min(int(x1 + (x2 - x1) / 2.) + 3, IMAGE_WIDTH - 1), 0)
    O_y1 = max(min(int(y1 + (y2 - y1) / 2.) - 3, IMAGE_HEIGHT - 1), 0)
    O_y2 = max(min(int(y1 + (y2 - y1) / 2.) + 3, IMAGE_HEIGHT - 1), 0)

    A_x1 = max(min(int(x1 + (x2 - x1) / 4.) - 3, IMAGE_WIDTH - 1), 0)
    A_x2 = max(min(int(x1 + (x2 - x1) / 4.) + 3, IMAGE_WIDTH - 1), 0)
    A_y1 = max(min(int(y1 + (y2 - y1) * 3 / 4.) - 3, IMAGE_HEIGHT - 1), 0)
    A_y2 = max(min(int(y1 + (y2 - y1) * 3 / 4.) + 3, IMAGE_HEIGHT - 1), 0)

    B_x1 = max(min(int(x1 + (x2 - x1) * 3 / 4.) - 3, IMAGE_WIDTH - 1), 0)
    B_x2 = max(min(int(x1 + (x2 - x1) * 3 / 4.) + 3, IMAGE_WIDTH - 1), 0)
    B_y1 = max(min(int(y1 + (y2 - y1) * 3 / 4.) - 3, IMAGE_HEIGHT - 1), 0)
    B_y2 = max(min(int(y1 + (y2 - y1) * 3 / 4.) + 3, IMAGE_HEIGHT - 1), 0)

    C_x1 = max(min(int(x1 + (x2 - x1) / 4.) - 3, IMAGE_WIDTH - 1), 0)
    C_x2 = max(min(int(x1 + (x2 - x1) / 4.) + 3, IMAGE_WIDTH - 1), 0)
    C_y1 = max(min(int(y1 + (y2 - y1) / 4.) - 3, IMAGE_HEIGHT - 1), 0)
    C_y2 = max(min(int(y1 + (y2 - y1) / 4.) + 3, IMAGE_HEIGHT - 1), 0)

    D_x1 = max(min(int(x1 + (x2 - x1) * 3 / 4.) - 3, IMAGE_WIDTH - 1), 0)
    D_x2 = max(min(int(x1 + (x2 - x1) * 3 / 4.) + 3, IMAGE_WIDTH - 1), 0)
    D_y1 = max(min(int(y1 + (y2 - y1) / 4.) - 3, IMAGE_HEIGHT - 1), 0)
    D_y2 = max(min(int(y1 + (y2 - y1) / 4.) + 3, IMAGE_HEIGHT - 1), 0)

    rect_O = depth_img[O_y1:O_y2, O_x1:O_x2]
    dist_O = np.sum(rect_O) / rect_O.size

    rect_A = depth_img[A_y1:A_y2, A_x1:A_x2]
    dist_A = np.sum(rect_A) / rect_A.size

    rect_B = depth_img[B_y1:B_y2, B_x1:B_x2]
    dist_B = np.sum(rect_B) / rect_B.size

    rect_C = depth_img[C_y1:C_y2, C_x1:C_x2]
    dist_C = np.sum(rect_C) / rect_C.size

    rect_D = depth_img[D_y1:D_y2, D_x1:D_x2]
    dist_D = np.sum(rect_D) / rect_D.size

    dist_list = [dist_O, dist_A, dist_B, dist_C, dist_D]
    kp_list = []
    cond_dist_list = []
    for dist in dist_list:
        ko = dist_O / dist
        ka = dist_A / dist
        kb = dist_B / dist
        kc = dist_C / dist
        kd = dist_D / dist
        k_list = [ko, ka, kb, kc, kd]
        kp = 0
        for kx in k_list:
            if 0.9 <= kx <= 1.1:
                kp += 1
        kp_list.append(kp)
        if kp >= 2:
            cond_dist_list.append(dist)
    if len(cond_dist_list) >= 1:
        distance = min(cond_dist_list)
    else:
        distance = dist_O

    # kp_max = max(kp_list)
    # if kp_max >= 3:
    #     distance = dist_list[kp_list.index(kp_max)]
    # else:
    #     distance = dist_O

    return distance
def run(
    weights=ROOT / 'yolov5s-seg.pt',  # model.pt path(s)
    source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / 'runs/predict-seg',  # save results to project/name
    name='exp',  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
    retina_masks=False,
):
    object_pose_pub = rospy.Publisher('/object_pose', Pose, queue_size=10)
    source = str(source)#将source转换为字符串
    save_img = not nosave and not source.endswith('.txt')  # save inference images  如果nosave为False且source不是txt文件，则保存图片
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)#如果后缀在两个库中，判断为文件
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))#开头是四者之一，则判断为url
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)#判断是不是摄像头
    screenshot = source.lower().startswith('screen')#判断是否为截图
    if is_url and is_file:
        source = check_file(source)  # download 确保输入文件是本地文件，如果是url，则下载至本地

    # Directories  创建保存目录
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir  如果save_txt为True，则创建labels文件夹，否则创建save_dir文件

    # Load model  加载模型
    device = select_device(device)#选择设备
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)#加载模型函数，weights为模型路径，device为设备，dnn为是否使用opencv dnn，data维数据集，fp16为是否使用fp16推理
    stride, names, pt = model.stride, model.names, model.pt #stride为模型中，输入图形通过卷积层时的压缩比例，names为模型类别，pt为模型的类型
    imgsz = check_img_size(imgsz, s=stride)  # check image size   验证图像大小为32（stride）的倍数

    # Dataloader
    bs = 1  # batch_size  初始化为1
    if webcam:   #如果source是摄像头，则创建loadStreams()对象
        view_img = check_imshow(warn=True)#是否显示图片
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride) #创建LoadStreams()对象，source为输入源，img_size为图像的大小，stride为模型的stride，auto为是否自动选择设备，vi_stride为视频帧率
        bs = len(dataset) #batch_size为数据集的长度
    elif screenshot:#如果source是截图，则创建LoadScreenshots()对象
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)#创建LoadScreenshoots()对象，source为输入源，img_size为图像的大小，stride为模型的stride，auto为是否自动选择设备
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)#创建LoadImages()对象，直接加载图片，source为输入源，img_size为图像的大小，stride为模型的stride，auto为是否自动选择设备，vi_stride为视频帧率
    vid_path, vid_writer = [None] * bs, [None] * bs  #初始化vi_path和vid_writer，vid_path为视频路径，vid_writer为视频写入对象

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup  预热模型，用于提前加载模型，加快推理速度，imgsz为图像大小，如果pt为True或者model.triton为True，则bs=1，否则bs为数据集的长度。3为通道数，*imgsz为图像大小，即(1,3,224,224)
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())#seen为已推理的图片数量，windows为空列表，dt为时间统计对象
    for path, im, im0s, vid_cap, s in dataset:#遍历数据集，path为图片路径，im为图片，im0s为原始图片，bid_cap为视频读取对象，s为视频帧率
        with dt[0]:#开始计时，读取图片时间
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3: #如果图片维度为3，则添加batch维度
                im = im[None]  # expand for batch dim 在前面添加batch维度，即（3，640，640）转换为（1，3，224，224）

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred, proto = model(im, augment=augment, visualize=visualize)[:2]

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image 遍历每张图片，enumerate()函数将pred转换为索引和值的形式，i为索引，det为对应的元素，即每个物体的预测框
            seen += 1    #检测的图片数量+1
            if webcam:  # batch_size >= 1  如果是摄像头，则获取视频帧率
                p, im0, frame = path[i], im0s[i].copy(), dataset.count #path[i]为路径列表，ims[i].copy()为将输入图像的副本存储在im0变量中，dataset.count为当前输入图像的帧率
                s += f'{i}: '#在打印输出中添加当前处理的图像索引号i，方便调试和查看结果。在此处，如果是摄像头模式，i表示当前批次中第i张图像；否则，i始终为0，因为处理的只有一张图像
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0) #如果不是摄像头，frame为0

            p = Path(p)  # to Path  将路径转换为Path对象
            save_path = str(save_dir / p.name)  # im.jpg  保存图片的路径，save_dir为保存图片的文件夹，p.name为图片名称
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt 保存预测框的路径，save_dir为保存图片的文件夹，p,stem为图片名称，data.mode为数据集的模式，如果是image，则是图片，否则为视频
            s += '%gx%g ' % im.shape[2:]  # print string  im.shape[2:]为图片大小，即（1，3，224，224)中的（224，224）
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))#Annotator()对象，用于在图片上回执分类结果，im0为原始图片，example为类别名称，pil为是否使用PIL绘制
            if len(det):
                if retina_masks:
                    # scale bbox first the crop masks
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size
                    masks = process_mask_native(proto[i], det[:, 6:], det[:, :4], im0.shape[:2])  # HWC
                else:
                    masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

                # Segments
                if save_txt:
                    segments = [
                        scale_segments(im0.shape if retina_masks else im.shape[2:], x, im0.shape, normalize=True)
                        for x in reversed(masks2segments(masks))]
                    # print(segments)

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Mask plotting
                annotator.masks(
                    masks,
                    colors=[colors(x, True) for x in det[:, 5]],
                    im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(device).permute(2, 0, 1).flip(0).contiguous() /
                    255 if retina_masks else im[i])

                # Write results
                i=0
                zz1 = []
                xx1 = []
                yy1 = []
                bbox = []

                clsi = []
                tcp_send = False
                    # conf = 0
                    # for *xyxy_o, conf_o, cls_o in reversed(det):
                    #     if conf_o > conf:
                    #         conf = conf_o
                    #         xyxy = xyxy_o
                    #         cls = cls_o

                    
                    # print("conf", conf)
                for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                    if save_txt:  # Write to file
                        seg = segments[j].reshape(-1)  # (n,2) to (n*2)
                        #xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *seg, conf) if opt.save_conf else (cls, *seg)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                            # color_image = np.array(Image_PIL.open('doc/example_data/color_start.png'),
                            #                        dtype=np.float32) / 255.0
                        # depth_image = np.array(Image_PIL.open('inference/imgs/d1.png'))
                        cap=cv2.VideoCapture(0)
                        ret,depth_image=cap.read()
                        #xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        label = f'{names[int(cls)]} {conf:.2f}'
                            # print("label:", names[int(cls)])

                        if names[int(cls)] == 'weed':
                            # if names[int(cls)] == opt.object_name:

                            bbox.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
                            print("bbox:", bbox)
                            c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                            # print(c1, c2)
                            xx1.append((int(xyxy[0]) + int(xyxy[2])) / 2)
                            yy1.append((int(xyxy[1]) + int(xyxy[3])) / 2)
                            # zz1 = depth_image[int(xyxy[1]), int(xyxy[0])] / 1000
                            zz1.append(compute_depth(depth_image, bbox[i]) / 1000)
                            clsi.append(int(cls))
                            i = i+1
                            tcp_send = True
                        # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    vel_msg = Pose()
                    if tcp_send:
                        if i>=1:
                            print("zz:", zz1)
                            min_index = zz1.index(min(zz1))
                            print("min_index:", min_index)
                            # if names[clsi[min_index]] == opt.object_name or names[clsi[min_index]] == 'mouse':

                            camera_coordinate1 = ((xx1[min_index] - ppx) * zz1[min_index] / fx, (yy1[min_index] - ppy) * zz1[min_index] / fy, zz1[min_index])

                            np.savetxt("bbox.txt", bbox[min_index], fmt='%f')
                                # print("camera_cordinate:", camera_coordinate1)
                        else:
                            camera_coordinate1 = ((xx1[0] - ppx) * zz1[0] / fx, (yy1[0] - ppy) * zz1[0] / fy, zz1[0])


                        print("point :", camera_coordinate1[0], camera_coordinate1[1],camera_coordinate1[2])
                        vel_msg.position.x = float(camera_coordinate1[0])
                        vel_msg.position.y = float(camera_coordinate1[1])
                        vel_msg.position.z = float(camera_coordinate1[2])
                        vel_msg.orientation.x = 0
                        vel_msg.orientation.y = 0
                        vel_msg.orientation.z = 0
                        vel_msg.orientation.w = 1
                        object_pose_pub.publish(vel_msg)
                        tcp_send=False
                    else:
                        vel_msg.position.x = 0
                        vel_msg.position.y = 0
                        vel_msg.position.z = 0.1
                        vel_msg.orientation.x = 0
                        vel_msg.orientation.y = 0
                        vel_msg.orientation.z = 0
                        vel_msg.orientation.w = 1

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # 1 millisecond
                    exit()

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s-seg.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/predict-seg', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--retina-masks', action='store_true', help='whether to plot masks in native resolution')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
