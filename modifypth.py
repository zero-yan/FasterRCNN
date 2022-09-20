import torch
# fy=torch.load("/root/yanjun/faster-rcnn/model_data/ep021-loss2.346-val_loss2.380.pth")
# for i in fy.keys():
#    print(i+'   '+str(list(fy[i].size())))
from nets.resnet50 import Bottleneck, ResNet
# from torchvision import models
# model_weight_path = "/root/yanjun/faster-rcnn-pytorch/logs/ep063-loss0.764-val_loss0.960.pth"   #自己的pth文件路径
# out_onnx = './sarm.onnx'           #保存生成的onnx文件路径
# # model = ResNet(Bottleneck,[3,4,6,3])   # 加载自己的的网络
# # model.load_state_dict(torch.load(model_weight_path)) #加载自己的pth文件
# # model.eval()
# model = FasterRCNN(21, anchor_scales=[8,16,32], backbone='sarmnet50', pretrained=False)
# model_dict = model.state_dict()
# pretrained_dict = torch.load(model_weight_path)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # model_dict = model.state_dict()
# # pretrained_dict = torch.load(model_path, map_location=device)
# a = {}
# for k, v in pretrained_dict.items():
#     try:    
#         if np.shape(model_dict[k]) ==  np.shape(v):
#             a[k]=v
#     except:
#         pass
# model_dict.update(a)
# model.load_state_dict(model_dict)
# x = torch.randn(1, 3, 224, 224)
# #define input and output nodes, can be customized
# input_names = ["input"]
# output_names = ["output"]
# #convert pytorch to onnx
# torch_out = torch.onnx.export(model, x, out_onnx, input_names=input_names, output_names=output_names)
from torchsummary import summary
model = ResNet(Bottleneck,[3,4,6,3])
input = torch.randn(size=(1, 3, 224, 224), dtype=torch.float32)
# print(model)
summary(model, ( 3, 224, 224), device='cpu')