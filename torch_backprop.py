# https://github.com/rgeirhos/texture-vs-shape

import os
import sys
from collections import OrderedDict
import torch
import torchvision
import torchvision.models
from torchvision import transforms
from torchvision.utils import save_image
import pickle
import numpy as np


torch.backends.cudnn.enabled = True

from PIL import Image
img = Image.open("003-5-gthumb-gwdata1200-ghdata1200-gfitdatamax.jpg_0.png").convert('RGB')



invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])


'''
transform = transforms.Compose([            #[1]
 transforms.Resize(256),                    #[2]
 transforms.CenterCrop(224),                #[3]
 transforms.ToTensor(),                     #[4]
 transforms.Normalize(                      #[5]
 mean=[0.485, 0.456, 0.406],                #[6]
 std=[0.229, 0.224, 0.225]                  #[7]
 )])
'''


transform = transforms.Compose([            #[1]
 transforms.ToTensor(),                     #[4]
 transforms.Normalize(                      #[5]
 mean=[0.485, 0.456, 0.406],                #[6]
 std=[0.229, 0.224, 0.225]                  #[7]
 )])

model = torchvision.models.vgg16(pretrained=False)

# they need dataparallel to load weights, but its not iterable
#model.features = torch.nn.DataParallel(model.features)




# ----------------------------

# load my save of this checkpoint
with open('vgg16_state_dict.pickle', 'rb') as f:
    state_dict = pickle.load(f)
    for k, v in state_dict.items():
        state_dict[k] = torch.from_numpy(v)
model.load_state_dict(state_dict)


'''
# ------------ Load Author's checkpoint and save pickle

filepath = "./texture-vs-shape-pretrained-models/vgg16_train_60_epochs_lr0.01-6c6fcc9f.pth.tar"

checkpoint = torch.load(filepath)

new_state_dict = OrderedDict()

state_dict = checkpoint['state_dict']
for k, v in state_dict.items():
    name = k.replace('.module','') # remove `module.`
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)





with open('vgg16_state_dict.pickle', 'wb') as f:
    for k,v in new_state_dict.items():
        new_state_dict[k] = v.cpu().numpy()
    pickle.dump(new_state_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

# ---------------------------------
'''


# Change pooling to avg:
for idx, module in model.features._modules.items():
    if module.__class__.__name__ == 'MaxPool2d':
        #print("Pool!")
        model.features._modules[idx] = torch.nn.AvgPool2d(2, stride=2)


res = list(model.features)
for idx, r in enumerate(res):
    print(idx, r)

# 3rd, 8th, 15th,22nd layer is relu1_2,relu2_2,relu3_3,relu4_3.
feature_model = torch.nn.Sequential(*list(model.features)[:15])

# # Parameters of newly constructed modules have requires_grad=True by default
#for param in feature_model.parameters():
#    param.requires_grad = False


feature_model.cuda()
feature_model.eval()


img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0).cuda()

criterion = torch.nn.MSELoss()


img_outputs = feature_model.forward(batch_t.cuda())

latent = torch.autograd.Variable(torch.zeros_like(batch_t), requires_grad=True).cuda()

opt = torch.optim.Adam([latent], lr=0.1)

# backpropagate
for i in range(1001):

    opt.zero_grad()
    latent_outputs = feature_model(latent)
    loss = criterion(latent_outputs, img_outputs)
    loss.backward(retain_graph=True)
    opt.step()

    if i % 100 == 0:

        print(loss.cpu().detach().numpy())
        inv_tensor = invTrans(latent[0].cpu().data)
        save_image(inv_tensor, 'out.png')








