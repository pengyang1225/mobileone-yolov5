# from model.yolov3 import Yolov3
# import config.yolov3_config_yoloformat as cfg
import torch
from od import Model
import copy
def repvgg_model_convert(model:torch.nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model


import  time
if __name__ == '__main__':
    train_state_dict =torch.load('/media/py/8ee085fc-6bf0-41db-8bee-dbf383701c5c/code/pycharm/flexible-yolov5/scripts/runs/train/exp26/weights/best.pt')
    train_model = Model("/media/py/8ee085fc-6bf0-41db-8bee-dbf383701c5c/code/pycharm/flexible-yolov5/configs/model_mobileone.yaml").to('cuda:0')  #
    train_model.load_state_dict(train_state_dict['model'].state_dict())
    deploy_model = Model("/media/py/8ee085fc-6bf0-41db-8bee-dbf383701c5c/code/pycharm/flexible-yolov5/configs/model_mobileone_deploy.yaml").to('cuda:0')  #
    repvgg_model_convert(train_model,'mobileone-yolov5_deploy2.pt')
    deploy_state_dict =torch.load('mobileone-yolov5_deploy2.pt')
    deploy_model.load_state_dict(deploy_state_dict)
    #whole_model_convert(train_model, deploy_model)
    # repvgg_model_convert

    #deploy_state_dict = deploy_model.state_dict()
    # torch.save(deploy_state_dict, '/media/py/8ee085fc-6bf0-41db-8bee-dbf383701c5c/code/pycharm/flexible-yolov5/scripts/runs/train/exp14/weights/best_deploy.pt')
    #
    train_model.eval()
    deploy_model.eval()
    x = torch.randn(1, 3, 640, 640).cuda()
    start =time.time()
    with torch.no_grad():
        train_p= train_model(x)[0]
        print()
        deploy_p = deploy_model(x)[0]
        print(((train_p - deploy_p) ** 2).sum())


    # deploy_model = Model("/media/py/8ee085fc-6bf0-41db-8bee-dbf383701c5c/code/pycharm/flexible-yolov5/configs/model_mobileone_deploy.yaml")  #
    # deploy_state_dict =torch.load('mobileone-yolov5_deploy2.pt')
    # deploy_model.load_state_dict(deploy_state_dict)
    # deploy_model.cuda().eval()
    # for i in range(10):
    #     x = torch.randn(1, 3, 640, 640).cuda()
    #     start =time.time()
    #     with torch.no_grad():
    #         deploy_p = deploy_model(x)[0]
    #         print("using time is",time.time() -start)
