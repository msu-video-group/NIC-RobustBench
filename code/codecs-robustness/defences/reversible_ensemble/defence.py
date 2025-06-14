import random
import functools

import torch
from torchvision.transforms.functional import rotate


class Defense:
    def __init__(self, num=10):
        self.defence_name = f'reversible_ensemble'
        self.num = num

    def preprocess(self, image):
        tau_list = []
        reversed_tau_list = []
        transforms_list = random.choices(["roll_c", "roll_h", "roll_w", "rotate"], weights=(1, 4, 4, 0), k=self.num)

        for transform_type in transforms_list:
            if transform_type == "roll_h":
                shift = random.randint(0, image.shape[2] - 1)
                tau_list.append(lambda x: torch.roll(x, shift, dims=2))
                reversed_tau_list.append(lambda x: torch.roll(x, -shift, dims=2))
            elif transform_type == "roll_w":
                shift = random.randint(0, image.shape[3] - 1)
                tau_list.append(lambda x: torch.roll(x, shift, dims=3))
                reversed_tau_list.append(lambda x: torch.roll(x, -shift, dims=3))
            elif transform_type == "roll_c":
                permute = [0, 1, 2]
                while permute == [0, 1, 2]:
                    random.shuffle(permute)
                def color_roll(image):
                    return image[:, permute]

                def reverse_color_roll(image):
                    _, indices = torch.sort(torch.tensor(permute))
                    return image[:, indices]

                tau_list.append(color_roll)
                reversed_tau_list.append(reverse_color_roll)
            else:
                angle = random.randint(0, 359)
                h, w = image.shape[-2:]
                
                def rot(image):
                    image = torch.nn.functional.pad(image, [w//2, w//2, h//2, h//2])
                    return rotate(image, angle)

                def reversed_rot(image):
                    reversed_rot = rotate(image, -angle)
                    reversed_rot = reversed_rot[..., h//2:-h//2, w//2:-w//2]
                    return reversed_rot
                tau_list.append(rot)
                reversed_tau_list.append(reversed_rot)

        tau = functools.partial(functools.reduce, lambda res, f: f(res), tau_list)
        reversed_tau_list.reverse()
        self.reverse_tau = functools.partial(functools.reduce, lambda res, f: f(res), reversed_tau_list)
        return tau(image)

    def postprocess(self, image):
        reversed_image = self.reverse_tau(image)
        return self.reverse_tau(image)
