import os
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from model.completionformer import CompletionFormer
from config import args as args_config
import torch
from PIL import Image
import time
import argparse


class AutoMine(Dataset):
    def __init__(self, root_folder):
        self.raw_folder = os.path.join(root_folder, 'raw')
        self.depth_folder = os.path.join(root_folder, 'depth')
        self.raw_file_list = os.listdir(self.raw_folder)
        self.depth_file_list = os.listdir(self.depth_folder)

    def __len__(self):
        return len(self.raw_file_list)

    def __getitem__(self, idx):
        raw_img_name = os.path.join(self.raw_folder, self.raw_file_list[idx])
        depth_img_name = os.path.join(self.depth_folder, self.depth_file_list[idx])
        raw_image = Image.open(raw_img_name)
        image_depth = np.array(Image.open(depth_img_name))
        depth_image = image_depth.astype(np.float32) / 256.0
        rgb = TF.to_tensor(raw_image)
        rgb = TF.normalize(rgb, (0.485, 0.456, 0.406),
                           (0.229, 0.224, 0.225), inplace=True)

        depth = TF.to_tensor(np.array(depth_image))
        output = {'rgb': rgb, 'dep': depth,
                  'name': self.depth_file_list[idx].split('.')[0] + "." + self.depth_file_list[idx].split('.')[
                      1]}
        return output


def main(args):
    save_dir = os.path.join(args.dir_data.replace("AutoMine-Depth", "AutoMine-DenseDepth"), "dense_depth")
    if os.path.exists(save_dir):
        os.system("rm -rf " + save_dir)
    os.system("mkdir -p " + save_dir)
    dataset = AutoMine(args.dir_data)
    loader_test = DataLoader(dataset=dataset, batch_size=8,
                             shuffle=False, num_workers=4)
    net = CompletionFormer(args)
    net.cuda()
    if args.pretrain is not None:
        assert os.path.exists(args.pretrain), \
            "file not found: {}".format(args.pretrain)

        checkpoint = torch.load(args.pretrain)
        key_m, key_u = net.load_state_dict(checkpoint['net'], strict=False)
        if key_u:
            print('Unexpected keys :')
            print(key_u)
        if key_m:
            print('Missing keys :')
            print(key_m)
            raise KeyError
        print('Checkpoint loaded from {}!'.format(args.pretrain))

    net.eval()
    for index, sample in enumerate(loader_test):
        data = {key: val.cuda() for key, val in sample.items()
                if val is not None and key != 'name'}

        with torch.no_grad():
            output = net(data)
            output = output['pred'].cpu().numpy().astype(np.uint8).squeeze()
            Image.fromarray(output).save(save_dir + f"/{sample['name'][0]}.png")


if __name__ == "__main__":
    main(args_config)
