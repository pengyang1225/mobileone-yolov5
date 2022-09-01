import sys
import argparse
sys.path.append('.')
from od.models.modules.experimental import *
from od.models.head.yolo import *
from od.data.datasets import letterbox
from utils.general import *
from utils.split_detector import SPLITINFERENCE
from utils.torch_utils import *
from od import Model

class Detector(object):
    def __init__(self, pt_path, img_size, conf_thres=0.4, iou_thres=0.3, classes=None, agnostic_nms=False,
                 xcycwh=True, device=0):
        self.pt_path = pt_path
        self.img_size = img_size
        self.device = torch.device('cuda:{}'.format(device)) if torch.cuda.is_available() else torch.device('cpu')
        self.model = self.load_model()
        #self.model2 = self.load_model2()
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.xcycwh = xcycwh

    def load_model(self):
        #model = attempt_load(self.pt_path).to(self.device)  # load FP32 model
        # train_state_dict = torch.load(
        #     self.pt_path)
        model = Model(
            "/media/py/8ee085fc-6bf0-41db-8bee-dbf383701c5c/code/pycharm/flexible-yolov5/configs/model_mobileone_deploy.yaml").to(
            'cuda:0')  #

        checkpoint = torch.load( self.pt_path)
        if 'model' in checkpoint:
            checkpoint = checkpoint['model']
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
       # aaaaa=checkpoint.state_dict()
        model.load_state_dict(checkpoint, strict=False)
       # model.eval()
       #  for m in model.modules():
       #      t = type(m)
       #      if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, YOLOHead):
       #          m.inplace = True  # torch 1.7.0 compatibility
       #          if t is YOLOHead and not isinstance(m.anchor_grid, list):
       #              delattr(m, 'anchor_grid')
       #              setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
       #      elif t is Conv:
       #          m._non_persistent_buffers_set = set()  # torch 1.6.0 compatibility
       #      elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
       #          m.recompute_scale_factor = None  # torch 1.11.0 compatibility
        return model

    def load_model2(self):
       # model = attempt_load(self.pt_path).to(self.device)  # load FP32 model
       #  deploy_state_dict = torch.load(
       #      'best_deploy.pt', map_location='cuda:0')
        model = Model(
            "/media/py/8ee085fc-6bf0-41db-8bee-dbf383701c5c/code/pycharm/flexible-yolov5/configs/model_repvgg_deploy.yaml").to(
            'cuda:0')  #


        checkpoint = torch.load("best_deploy.pt")

        model_dict = model.state_dict()
        model.load_state_dict(checkpoint, strict=False)

       # 查看模型的权重和biass系数

      #  pretrained_dict = {k1: v for (k, v), k1 in zip(checkpoint.items(), model_dict)}

       # model_dict.update(pretrained_dict)  # 更新model网络模型的参数的权值和biass，这相当于是一个浅拷贝，对这个更新改变会更改模型的权重和biass
      #  model.cuda()
      #  model.eval()
       # model.load_state_dict(deploy_state_dict, strict=False)

       # model.eval()

        # # Compatibility updates
        # for m in model.modules():
        #     t = type(m)
        #     if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, YOLOHead):
        #         m.inplace = True  # torch 1.7.0 compatibility
        #         if t is YOLOHead and not isinstance(m.anchor_grid, list):
        #             delattr(m, 'anchor_grid')
        #             setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
        #     elif t is Conv:
        #         m._non_persistent_buffers_set = set()  # torch 1.6.0 compatibility
        #     elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
        #         m.recompute_scale_factor = None  # torch 1.11.0 compatibility
        return model

    # def load_checkpoint(self,model, ckpt_path):
    #     checkpoint = torch.load(ckpt_path)
    #     if 'model' in checkpoint:
    #         checkpoint = checkpoint['model']
    #     if 'state_dict' in checkpoint:
    #         checkpoint = checkpoint['state_dict']
    #     ckpt = {}
    #     for k, v in checkpoint.items():
    #         if k.startswith('module.'):
    #             ckpt[k[7:]] = v
    #         else:
    #             ckpt[k] = v
    #     model.load_state_dict(ckpt)

    def __call__(self, ori_img, split_width=1, split_height=1):
        if split_width == 1 and split_height == 1:
            bboxes, scores, ids = self.detect_image(ori_img)
        else:
            bboxes = []
            scores = []
            ids = []
            output = self.detect_img_split(image=ori_img, split_width=split_width, split_height=split_height)['data']
            for key in output.keys():
                values = output[key]
                for value in values:
                    x_min = value[0]
                    y_min = value[1]
                    x_max = value[2]
                    y_max = value[3]
                    w = x_max - x_min
                    h = y_max - y_min
                    if self.xcycwh:
                        bboxes.append([x_min + w / 2, y_min + h / 2, w, h])
                    else:
                        bboxes.append(value[:4])
                    scores.append(value[4])
                    ids.append(key)
        return np.asarray(bboxes), np.asarray(scores), np.asarray(ids)

    def detect_image(self, image):
        bboxes = []
        scores = []
        ids = []
        im0s = image
        img = letterbox(im0s, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()
        img /= 255
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        self.model.eval()
        start =time.time()
        pred = self.model(img)[0]
        print("forward time =",time.time()-start)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, labels=[], multi_label=True, agnostic=False)
        for i, det in enumerate(pred):
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                for *xyxy, conf, cls in det:
                    x_min = xyxy[0].cpu().detach().float().numpy()
                    y_min = xyxy[1].cpu().detach().float().numpy()
                    x_max = xyxy[2].cpu().detach().float().numpy()
                    y_max = xyxy[3].cpu().detach().float().numpy()
                    score = conf.cpu().detach().float().numpy()
                    clas = cls.cpu().detach().float().numpy()
                    w = x_max - x_min
                    h = y_max - y_min
                    if self.xcycwh:
                        # center coord, w, h
                        bboxes.append([x_min + w / 2, y_min + h / 2, w, h])
                    else:
                        bboxes.append([x_min, y_min, x_max, y_max])
                    scores.append(score)
                    ids.append(clas)
        return np.asarray(bboxes), np.asarray(scores), np.asarray(ids)

    @SPLITINFERENCE(split_width=2, split_height=1)
    def detect_img_split(self, image='', **kwargs):
        outputs_json = {}
        im0s = image
        img = letterbox(im0s, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        self.model.cuda().eval()
        pred = self.model(img)[0]
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes,
                                   agnostic=self.agnostic_nms)
        for i, det in enumerate(pred):
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                for *xyxy, conf, cls in det:
                    x_min = xyxy[0].cpu()
                    y_min = xyxy[1].cpu()
                    x_max = xyxy[2].cpu()
                    y_max = xyxy[3].cpu()
                    score = conf.cpu()
                    clas = cls.cpu()
                    if clas in outputs_json:
                        outputs_json[clas].append([x_min, y_min, x_max, y_max, score])
                    else:
                        outputs_json[clas] = [[x_min, y_min, x_max, y_max, score]]
        return {'data': outputs_json}

import time
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='/media/py/8ee085fc-6bf0-41db-8bee-dbf383701c5c/code/pycharm/flexible-yolov5/scripts/mobileone-yolov5_deploy2.pt', help='weights path')
    parser.add_argument('--imgs_root', type=str, default='../test_imgs', help='test images root')
    parser.add_argument('--save_dir', type=str, default='./results', help='save result dir')
    parser.add_argument('--xcycwh', type=bool, default=False, help='box format')
    parser.add_argument('--img_size', type=int, default=640, help='test image size')
    parser.add_argument('--conf_thresh', type=float, default=0.4, help='confidence thresh')
    parser.add_argument('--iou_thresh', type=float, default=0.3, help='nms iou thresh')
    parser.add_argument('--filter_class', type=int, default=None, help='filter specify class id')

    opt = parser.parse_args()
    pt_path = opt.weights
    model = Detector(pt_path, opt.img_size, conf_thres=opt.conf_thresh, iou_thres=opt.iou_thresh, classes=opt.filter_class, xcycwh=opt.xcycwh)
    imgs_root = opt.imgs_root
    imgs = os.listdir(imgs_root)
    save_dir = opt.save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for img in imgs:
        im = cv2.imread(os.path.join(imgs_root, img))
        start =time.time()
        bboxes, scores, ids = model.detect_image(im)
        print("using time =",time.time()-start)
        for idx in range(bboxes.shape[0]):
            bbox = bboxes[idx].astype(int)
            score = scores[idx]
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2, 1)
        cv2.imwrite(os.path.join(save_dir, img), im)
