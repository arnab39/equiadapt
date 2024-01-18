import torch, math
import numpy as np
from torchvision import transforms
from collections import defaultdict

from PIL import Image

class SO2Inference:
    def __init__(self, eval_num_rotations, network, model_type, im_shape=(3, 32, 32)):
        self.eval_num_rotations = eval_num_rotations
        self.network = network
        self.model_type = model_type
        self.logit_dicts = {}
        self.y_dicts = {}
        self.id_metric = defaultdict(lambda : 0)

        self.pad = transforms.Pad(math.ceil(im_shape[-2] * 0.4), padding_mode='edge')
        self.crop = transforms.CenterCrop((im_shape[-2], im_shape[-1]))

    def save_image_test(self, x, rot, prefix=''):
        import copy
        x_numpy = copy.deepcopy(x.detach())
        x_numpy[0] = x_numpy[0] * 0.247 + 0.4914
        x_numpy[1] = x_numpy[1] * 0.243 + 0.4822 
        x_numpy[2] = x_numpy[2] * 0.261 + 0.4465 

        x_numpy = x_numpy.permute(1, 2, 0).detach().cpu().numpy()
        im = Image.fromarray((x_numpy * 255).astype(np.uint8))
        im.save(f"{prefix}image_{rot}.png")

    def infer_classification(self, x, y):
        degrees = torch.linspace(0, 360, self.eval_num_rotations + 1)[:-1]
        for rot, degree in enumerate(degrees):
            
            x_rot = self.pad(x)
            x_rot = transforms.functional.rotate(x_rot, int(degree))
            x_rot = self.crop(x_rot)
            
            if self.model_type in ('steerable'):
                logits, inverse_rotation, vectors = self.network(x_rot)
            elif self.model_type in ('equivariant', 'opt_equivariant'):
                logits, fibres, x_canonized, group = self.network(x_rot)
                self.id_metric[degree] = (torch.argmax(fibres, dim=1) == rot).sum()/fibres.shape[0]
            else:
                logits = self.network(x_rot)
            self.logit_dicts[rot] = logits
            self.y_dicts[rot] = y

        return self.logit_dicts, self.y_dicts, self.id_metric

class O2Inference:
    def __init__(self, eval_num_rotations, network, model_type, im_shape=(3, 32, 32)):
        self.eval_num_rotations = eval_num_rotations
        self.network = network
        self.model_type = model_type
        self.logit_dicts = {}
        self.y_dicts = {}

        self.pad = transforms.Pad(math.ceil(im_shape[-2] * 0.3), padding_mode='edge')
        self.crop = transforms.CenterCrop((im_shape[-2], im_shape[-1]))

    def infer_classification(self, x, y):
        degrees = torch.linspace(0, 360, self.eval_num_rotations + 1)[:-1]

        # no reflection
        for rot, degree in enumerate(degrees):
            
            x_rot = self.pad(x)
            x_rot = transforms.functional.rotate(x_rot, int(degree))
            x_rot = self.crop(x_rot)

            if self.model_type in ('steerable'):
                logits, inverse_rotation, vectors = self.network(x_rot)
            elif self.model_type in ('equivariant', 'opt_equivariant'):
                logits, fibres, x_canonized, group = self.network(x_rot)
            else:
                logits = self.network(x_rot)
            self.logit_dicts[rot] = logits
            self.y_dicts[rot] = y

        # with reflection
        for rot, degree in enumerate(degrees):

            x_rot = self.pad(x)
            x_rot = transforms.functional.hflip(x_rot)
            x_rot = transforms.functional.rotate(x_rot, int(degree))
            x_rot = self.crop(x_rot)

            if self.model_type in ('steerable'):
                logits, inverse_rotation, vectors = self.network(x_rot)
            elif self.model_type in ('equivariant', 'opt_equivariant'):
                logits, fibres, x_canonized, group = self.network(x_rot)
            else:
                logits = self.network(x_rot)
            self.logit_dicts[rot + len(degrees)] = logits
            self.y_dicts[rot + len(degrees)] = y

        return self.logit_dicts, self.y_dicts