import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')
import torch
import torchvision.models as models
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('/home/avic/OOD_Orientation_Generalization')
from my_dataclasses import *
from train.dataset import *


x = int(sys.argv[-1])
e = ExpData.get_experiments(x)

batch_size = 1000
rd = RotationDataset(e)
df = pd.read_csv(rd.testing.full.path.merged_annotation_path())
dataset = rd._Dataset(rd, df)

dataloader = DataLoader(dataset, batch_size, num_workers=4, shuffle=False, pin_memory=True)
checkpoint = torch.load(e.checkpoint)
model = models.resnet18(pretrained=False, num_classes=50)
model.load_state_dict(checkpoint)
model.cuda()
pass

assert len(dataset) % batch_size == 0
image_activations = np.zeros((len(dataloader) * batch_size, 512))

w = model.fc.weight.cpu().detach().numpy()

global avgpool_output
def hook(model, input, output):
    global avgpool_output
    avgpool_output = torch.squeeze(output).cpu().detach().numpy()

model.avgpool.register_forward_hook(hook)
model.eval()
for b, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
    images, targets = map(lambda t: t.to(device='cuda', non_blocking=True), batch)
    with torch.no_grad():
        preds = model(images)
    image_activations[batch_size * b : batch_size * (b + 1)] = avgpool_output
np.save(e.image_activations, image_activations)