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
def whole_model_convert(train_model:torch.nn.Module, deploy_model:torch.nn.Module, save_path=None):
    all_weights = {}
    train_dict = train_model.state_dict()
    deploy_dict = deploy_model.state_dict()
    for name, module in train_model.named_modules():
        if (hasattr(module, 'backbone')):
            continue
        if hasattr(module, 'repvgg_convert'):
            kernel, bias = module.repvgg_convert()
            all_weights[name + '.rbr_reparam.weight'] = kernel
            all_weights[name + '.rbr_reparam.bias'] = bias
            print('convert RepVGG block')
        else:
            for p_name, p_tensor in module.named_parameters():    # p_name is weight and bias
                full_name = name + '.' + p_name
                if full_name not in all_weights:
                    # all_weights[full_name] = p_tensor.detach().cpu().numpy()
                    all_weights[full_name] = p_tensor.detach()
            for p_name, p_tensor in module.named_buffers():
                full_name = name + '.' + p_name
                if full_name not in all_weights:
                    # all_weights[full_name] = p_tensor.cpu().numpy()
                    all_weights[full_name] = p_tensor

    deploy_model.load_state_dict(all_weights, strict=False)
    if save_path is not None:
        torch.save(deploy_model.state_dict(), save_path)

    return deploy_model

# train_state_dict = torch.load('./weight/repA1g4_seed1_20each/best.pt')
# train_model = Yolov3(cfg=cfg, deploy=False)
# train_model.load_state_dict(train_state_dict, strict=False)
# deploy_model = Yolov3(cfg=cfg, deploy=True)
# train_state_dict =torch.load('/media/py/8ee085fc-6bf0-41db-8bee-dbf383701c5c/code/pycharm/flexible-yolov5/scripts/runs/train/exp14/weights/best.pt')
# train_model = Model("/media/py/8ee085fc-6bf0-41db-8bee-dbf383701c5c/code/pycharm/flexible-yolov5/configs/model_repvgg.yaml").to('cuda:0')  #
# train_model.load_state_dict(train_state_dict['model'].state_dict(), strict=False)
# deploy_model = Model("/media/py/8ee085fc-6bf0-41db-8bee-dbf383701c5c/code/pycharm/flexible-yolov5/configs/model_repvgg_deploy.yaml").to('cuda:0')  #
# repvgg_model_convert(train_model,'best_deploy.pt')
# deploy_state_dict =torch.load('best_deploy.pt')
# deploy_model.load_state_dict(deploy_state_dict, strict=False)
# #whole_model_convert(train_model, deploy_model)
# # repvgg_model_convert
#
# #deploy_state_dict = deploy_model.state_dict()
# # torch.save(deploy_state_dict, '/media/py/8ee085fc-6bf0-41db-8bee-dbf383701c5c/code/pycharm/flexible-yolov5/scripts/runs/train/exp14/weights/best_deploy.pt')
# #
# train_model.eval()
# deploy_model.eval()
# x = torch.randn(1, 3, 640, 640).cuda()
# #
# with torch.no_grad():
#     train_p= train_model(x)[0]
#     deploy_p = deploy_model(x)[0]
#     print(((train_p[0] - deploy_p[0]) ** 2).sum())


deploy_model = Model("/media/py/8ee085fc-6bf0-41db-8bee-dbf383701c5c/code/pycharm/flexible-yolov5/configs/model_mobileone_deploy.yaml")  #
deploy_state_dict =torch.load('mobileone-yolov5_deploy.pt')
deploy_model.load_state_dict(deploy_state_dict, strict=False)
deploy_model.eval()
for i in range(10):
    x = torch.randn(1, 3, 512, 512)
    start =time.time()
    with torch.no_grad():
        deploy_p = deploy_model(x)[0]
        print("using time is",time.time() -start)
