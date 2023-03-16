import os

import numpy as np
from PIL import Image
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('npz_path',type=str)
parser.add_argument('-n','--n_ims',default=10,type=int)
args=parser.parse_args()

# import pytorch_fid

images_data = np.load(args.npz_path)
# print(images_data.files)
images_np = images_data['arr_0']
# print(images_np.shape)
if not os.path.exists('out_ims'):
    os.mkdir('out_ims')
print(f"tot images {images_np.shape}")
for i in range(images_np.shape[0])[:args.n_ims]:
    image_np = images_np[i,:,:,:].astype(int)
    # image_np = image_np.transpose(2,0,1)
    # image_np = image_np.astype(dtype=np.float32)
    # print(image_np,image_np.shape)
    # print(image_np.dtype)

    PIL_image = Image.fromarray(np.uint8(image_np)).convert('RGB')
    # generated_image_dir = os.mkdir()
    save_file='out_ims/generated_image_{}.png'.format(str(i))
    PIL_image.save(save_file)
    print(f"save to {save_file}")
    # PIL_image.show()

    # rgb_img = np.float32(image_np / 255)
    # print(rgb_img)
    # plt.imshow(image_np)
    # plt.show()
