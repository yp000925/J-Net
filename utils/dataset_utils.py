import torch
from glob import glob
import numpy as np
from torch.nn.functional import pad
import torchvision.transforms
from PIL import Image
from typing import Callable, Optional
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from pathlib import Path
import os
import skimage
import scipy
from torchvision.transforms.transforms import Resize, ToTensor
import matplotlib.pyplot as plt

import yaml

def one_pad_image(image, new_size):
    if isinstance(image, torch.Tensor):
        if len(image.shape) == 4:
            B, C, H, W = image.shape
            padded_image = torch.ones([B,C,new_size[0], new_size[1]]).to(image.dtype).to(image.device)
            start_row = (new_size[0] - H) // 2
            start_col = (new_size[1] - W) // 2
            padded_image[:,:,start_row:start_row + H, start_col:start_col + W] = image
            return padded_image

        elif len(image.shape) == 3:
            C, H, W = image.shape
            padded_image = torch.ones([C,new_size[0], new_size[1]]).to(image.dtype).to(image.device)
            start_row = (new_size[0] - H) // 2
            start_col = (new_size[1] - W) // 2
            padded_image[:,start_row:start_row + H, start_col:start_col + W] = image
            return padded_image

        elif len(image.shape) == 2:
            H, W = image.shape
            padded_image = torch.ones([new_size[0], new_size[1]]).to(image.dtype).to(image.device)
            start_row = (new_size[0] - H) // 2
            start_col = (new_size[1] - W) // 2
            padded_image[start_row:start_row + H, start_col:start_col + W] = image
            return padded_image
        else:
            raise Exception(f'Unsupported shape {image.shape}')

    elif isinstance(image, np.ndarray):
        if len(image.shape) == 4:
            B, C, H, W = image.shape
            padded_image = np.ones([B,C,new_size[0], new_size[1]])
            start_row = (new_size[0] - H) // 2
            start_col = (new_size[1] - W) // 2
            padded_image[:,:,start_row:start_row + H, start_col:start_col + W] = image
            return padded_image

        elif len(image.shape) == 3:
            C, H, W = image.shape
            padded_image = np.ones([C,new_size[0], new_size[1]])
            start_row = (new_size[0] - H) // 2
            start_col = (new_size[1] - W) // 2
            padded_image[:,start_row:start_row + H, start_col:start_col + W] = image
            return padded_image

        elif len(image.shape) == 2:
            H, W = image.shape
            padded_image = np.ones([new_size[0], new_size[1]])
            start_row = (new_size[0] - H) // 2
            start_col = (new_size[1] - W) // 2
            padded_image[start_row:start_row + H, start_col:start_col + W] = image
            return padded_image
        else:
            raise Exception(f'Unsupported shape {image.shape}')
    else:
        raise Exception(f'Unsupported type {type(image)}')

def center_crop(image, new_size):
    if isinstance(image, torch.Tensor):
        if len(image.shape) == 4:
            start_row = (image.shape[-2] - new_size[-2]) // 2
            start_col = (image.shape[-1] - new_size[-1]) // 2
            return image[:,:,start_row:start_row + new_size[0], start_col:start_col + new_size[1]]
        elif len(image.shape) == 3:
            start_row = (image.shape[-2] - new_size[-2]) // 2
            start_col = (image.shape[-1] - new_size[-1]) // 2
            return image[:,start_row:start_row + new_size[0], start_col:start_col + new_size[1]]
        elif len(image.shape) == 2:
            start_row = (image.shape[-2] - new_size[-2]) // 2
            start_col = (image.shape[-1] - new_size[-1]) // 2
            return image[start_row:start_row + new_size[0], start_col:start_col + new_size[1]]
        else:
            raise Exception(f'Unsupported shape {image.shape}')
    elif isinstance(image, np.ndarray):
        start_row = (image.shape[0] - new_size[0]) // 2
        start_col = (image.shape[1] - new_size[1]) // 2
        return image[start_row:start_row + new_size[0], start_col:start_col + new_size[1]]
    else:
        raise Exception(f'Unsupported type {type(image)}')

# def center_crop(self, image, new_size):
#     start_row = (image.shape[0] - new_size[0]) // 2
#     start_col = (image.shape[1] - new_size[1]) // 2
#     return image[start_row:start_row + new_size[0], start_col:start_col + new_size[1]]
def zero_pad_image(image, new_size):
    padded_image = torch.zeros(new_size)
    start_row = (new_size[0] - image.shape[0]) // 2
    start_col = (new_size[1] - image.shape[1]) // 2
    padded_image[start_row:start_row + image.shape[0], start_col:start_col + image.shape[1]] = image
    return padded_image

# def generate_otf_torch(wavelength, nx, ny, deltax, deltay, distance, pad_size=None):
#     """
#     Generate the otf from [0,pi] not [-pi/2,pi/2] using torch
#     :param wavelength:
#     :param nx:
#     :param ny:
#     :param deltax:
#     :param deltay:
#     :param distance:
#     :return:
#     """
#     if pad_size:
#         nx = pad_size[0]
#         ny = pad_size[1]
#     r1 = torch.linspace(-nx / 2, nx / 2 - 1, nx)
#     c1 = torch.linspace(-ny / 2, ny / 2 - 1, ny)
#     deltaFx = 1 / (nx * deltax) * r1
#     deltaFy = 1 / (nx * deltay) * c1
#     mesh_qx, mesh_qy = torch.meshgrid(deltaFx, deltaFy)
#     k = 2 * torch.pi / wavelength
#     otf = np.exp(1j * k * distance * torch.sqrt(1 - wavelength ** 2 * (mesh_qx ** 2
#                                                                        + mesh_qy ** 2)))
#     otf = torch.fft.ifftshift(otf)
#     return otf
def generate_otf_torch(wavelength, nx, ny, deltax, deltay, distance, pad_size=None, device='cpu'):
    """
    Generate the otf from [0,pi] not [-pi/2,pi/2] using torch
    :param wavelength:
    :param nx:
    :param ny:
    :param deltax:
    :param deltay:
    :param distance:
    :return:
    """
    if not isinstance(nx, torch.Tensor):
        nx = torch.tensor(nx)
    if not isinstance(ny, torch.Tensor):
        ny = torch.tensor(ny)
    if not isinstance(deltax, torch.Tensor):
        deltax = torch.tensor(deltax)
    if not isinstance(deltay, torch.Tensor):
        deltay = torch.tensor(deltay)
    if not isinstance(distance, torch.Tensor):
        distance = torch.tensor(distance)
    if not isinstance(wavelength, torch.Tensor):
        wavelength = torch.tensor(wavelength)

    if pad_size:
        nx = pad_size[0]
        ny = pad_size[1]
    nx = torch.tensor(nx).to(device)
    ny = torch.tensor(ny).to(device)
    r1 = torch.linspace(-nx / 2, nx / 2 - 1, nx).to(device)
    c1 = torch.linspace(-ny / 2, ny / 2 - 1, ny).to(device)
    deltaFx = 1 / (nx * deltax) * r1
    deltaFy = 1 / (nx * deltay) * c1
    mesh_qx, mesh_qy = torch.meshgrid(deltaFx, deltaFy)
    k = 2 * torch.pi / wavelength
    otf = torch.exp(
        1j * k.to(device) * distance.to(device) * torch.sqrt(1 - wavelength.to(device) ** 2 * (mesh_qx.to(device) ** 2
                                                                                               + mesh_qy.to(
                    device) ** 2)))
    otf = torch.fft.ifftshift(otf).to(device)
    return otf


def psnr(x, im_orig):
    def norm_tensor(x):
        return (x - torch.min(x)) / (torch.max(x) - torch.min(x)+1e-8)

    x = norm_tensor(x)
    im_orig = norm_tensor(im_orig)
    mse = torch.mean(torch.square(im_orig - x))
    psnr = torch.tensor(10.0) * torch.log10(1 / mse)
    if psnr>1000:
        return torch.tensor(0.0)
    return psnr


def get_dataloader(dataset,
                   batch_size: int,
                   num_workers: int,
                   train: bool):
    dataloader = DataLoader(dataset,
                            batch_size,
                            shuffle=train,
                            num_workers=num_workers,
                            drop_last=train)
    return dataloader


def get_dataset(data_config, type='sim'):
    if type == 'sim':
        gt = DHDataset(data_config['PATH'])
        prop_kernel = data_config['measurement']['prop_kernel']
        operator = DH_operator(**prop_kernel)
        y = operator.forward(gt)


def prepross_bg(img, bg):
    temp = img / bg
    out = (temp - np.min(temp)) / (1 - np.min(temp))
    return out


def comp_field_norm(comp_field):
    # comp_field [N, C=2 (real&imag), H, W] or [C, H, W]
    if isinstance(comp_field, np.ndarray):
        if comp_field.ndim == 3:
            comp_field = comp_field[0, ...] + 1j * comp_field[1, ...]
            comp_field /= (np.mean(np.abs(comp_field), axis=(-2, -1), keepdims=True) * np.exp(
                1j * np.mean(np.angle(comp_field), axis=(-2, -1), keepdims=True)))
            comp_field /= np.mean(comp_field, axis=(-2, -1), keepdims=True)
            return np.stack((np.real(comp_field), np.imag(comp_field)), axis=0)
        elif comp_field.ndim == 4:
            comp_field = comp_field[:, 0, ...] + 1j * comp_field[:, 1, ...]
            comp_field /= (np.mean(np.abs(comp_field), axis=(-2, -1), keepdims=True) * np.exp(
                1j * np.mean(np.angle(comp_field), axis=(-2, -1), keepdims=True)))
            comp_field /= np.mean(comp_field, axis=(-2, -1), keepdims=True)
            return np.stack((np.real(comp_field), np.imag(comp_field)), axis=1)
    elif isinstance(comp_field, torch.Tensor):
        comp_field = comp_field[:, 0, ...] + 1j * comp_field[:, 1, ...]
        comp_field /= (torch.mean(torch.abs(comp_field), dim=(-2, -1), keepdim=True) * torch.exp(
            1j * torch.mean(torch.angle(comp_field), dim=(-2, -1), keepdim=True)))
        comp_field /= torch.mean(comp_field, dim=(-2, -1), keepdim=True)


class DHDataset(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable] = None, img_size: int = 256):
        super().__init__(root, transforms)
        self.img_size = img_size
        # self.transforms =  transforms.Compose([transforms.ToTensor(),
        #                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        try:
            f = []
            for p in root if isinstance(root, list) else [root]:
                p = Path(root)
                if p.is_dir():
                    f += glob(str(p / '**' / "*.*"), recursive=True)
                else:
                    raise Exception(f'{p} does not exist')
            self.fpaths = sorted([x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in ['png', 'jpg']])
            assert self.fpaths, f'{root}{p}No mat found'
        except Exception as e:
            raise Exception(f'Error loading data from {root}:{e}\n')

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert('RGB').resize([self.img_size, self.img_size])

        if self.transforms is not None:
            img = self.transforms(img)

        return img


class DH_operator:
    def __init__(self, **prop_kernel):
        self.A = generate_otf_torch(**prop_kernel)
        self.AT = torch.conj(self.A)
        self.device = 'cpu'
        self.nx = prop_kernel['nx']
        self.ny = prop_kernel['ny']
        self.pad_size = prop_kernel.get('pad_size', None)


    def forward(self, data, **kwargs):
        self.device = data.device
        self.normalized = kwargs.get('normalized', False)

        # if self.pad_size:
        #     data = one_pad_image(data, self.pad_size)
        if self.pad_size:
            h_p = (self.pad_size[0] - data.shape[-2]) // 2
            w_p = (self.pad_size[1] - data.shape[-1]) // 2
            if len(data.shape) == 3:
                data = data.unsqueeze(0)
                data = pad(data, [h_p, h_p, w_p, w_p], mode='replicate')[0, :, :, :]

            elif len(data.shape) == 2:
                data =  data.unsqueeze(0).unsqueeze(0)
                data = pad(data, [h_p,h_p,w_p,w_p], mode='replicate')[0,0,:,:]

            elif len(data.shape) == 4:
                data = pad(data, [h_p, h_p, w_p, w_p], mode='replicate')
            else:
                raise Exception(f'Unsupported shape {data.shape}')

        fs_out = torch.multiply(torch.fft.fft2(data), self.A.expand(data.shape).to(self.device))
        f_out = torch.fft.ifft2(fs_out)
        amplitude = f_out.abs()
        if self.pad_size:
            amplitude = center_crop(amplitude, [self.nx, self.ny])
        if self.normalized:
            amplitude = amplitude/torch.max(amplitude)
        return amplitude

    def backward(self, data):
        self.device = data.device
        fs_out = torch.multiply(torch.fft.fft2(data), self.AT.expand(data.shape).to(self.device))
        f_out = torch.fft.ifft2(fs_out)
        amplitude = f_out.abs()
        # amplitude = amplitude/torch.max(amplitude)
        return amplitude


class AdpNetDataset(torch.utils.data.Dataset):
    def __init__(self, path, z_range, prop_kernel, mode='R', noise_level=None):

        try:
            f = []
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)
                if p.is_dir():
                    f += glob(str(p / '**' / "*.*"), recursive=True)
                else:
                    raise Exception(f'{p} does not exist')
            self.img_files = sorted([x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in ['png', 'jpg']])
            assert self.img_files, f'{path}{p}No mat found'
        except Exception as e:
            raise Exception(f'Error loading data from {path}:{e}\n')

        self.file_paths = self.img_files
        self.z_range = z_range
        # self.l = length
        self.mode = mode
        self.prop_kernel = prop_kernel
        self.pad_size = prop_kernel.get('pad_size', None)
        self.nx = prop_kernel['nx']
        self.ny = prop_kernel['ny']
        self.operator = DH_operator(**self.prop_kernel)
        self.noise_level = noise_level



    def update_operator(self, depth_ratio):
        self.prop_kernel['distance'] = self.z_range[0] + (self.z_range[1] - self.z_range[0]) * depth_ratio
        self.operator = DH_operator(**self.prop_kernel)

    def __len__(self):
        return len(self.file_paths)

    def rescale_phase(self, phase, range=[-1, 1]):
        return (phase - phase.min()) / (phase.max() - phase.min()) * (range[1] - range[0]) + range[0]
    def zero_pad_image(self, image, new_size):
        padded_image = np.zeros(new_size)
        start_row = (new_size[0] - image.shape[0]) // 2
        start_col = (new_size[1] - image.shape[1]) // 2
        padded_image[start_row:start_row + image.shape[0], start_col:start_col + image.shape[1]] = image
        return padded_image

    def one_pad_image(self, image, new_size):
        padded_image = np.ones(new_size)
        start_row = (new_size[0] - image.shape[0]) // 2
        start_col = (new_size[1] - image.shape[1]) // 2
        padded_image[start_row:start_row + image.shape[0], start_col:start_col + image.shape[1]] = image
        return padded_image

    def center_crop(self, image, new_size):
        start_row = (image.shape[0] - new_size[0]) // 2
        start_col = (image.shape[1] - new_size[1]) // 2
        return image[start_row:start_row + new_size[0], start_col:start_col + new_size[1]]

    def __getitem__(self, index):

        if self.mode == 'C':
            amp_img = np.array(Image.open(self.file_paths[index]).convert('L')).astype('float32') / 255.0
            phase_img = np.array(Image.open(self.file_paths[index // 2]).convert('L')).astype('float32') / 255.0
            phase_img = self.rescale_phase(phase_img, range=[-np.pi, np.pi])
            # if self.pad_size:
            #     amp_img = self.one_pad_image(amp_img, self.pad_size)
            #     phase_img = self.zero_pad_image(phase_img, self.pad_size)


        elif self.mode == 'P':  # phase-only artificial objects
            phase_img = 1-np.array(Image.open(self.file_paths[index]).convert('L')).astype('float32') / 255
            phase_img = self.rescale_phase(phase_img, range=[-np.pi, np.pi])
            # if self.pad_size:
            #     phase_img = self.zero_pad_image(phase_img, self.pad_size)
            amp_img = np.ones_like(phase_img)

        elif self.mode == 'A':  # amplitude-only artificial objects
            amp_img = np.array(Image.open(self.file_paths[index]).convert('L')).astype('float32') / 255.0
            # if self.pad_size:
            #     amp_img = self.one_pad_image(amp_img, self.pad_size)
            phase_img = np.zeros_like(amp_img)

        comp_field = amp_img * np.exp(1j * phase_img)

        # 2x downsampling & upsampling
        # comp_field = skimage.measure.block_reduce(comp_field, block_size=2, func=np.mean)
        # comp_field = scipy.ndimage.zoom(comp_field, 2, order=1)
        # # Gaussian smooth
        # comp_field = scipy.ndimage.gaussian_filter(comp_field, sigma=1.0, mode='constant', cval=0)
        # # add white noise
        try:
            if self.noise_level:
                s = comp_field.shape[-1]
                comp_field += np.random.normal(loc=0, scale=self.noise_level, size=(s, s, 2)).view(
                    np.complex128).reshape(s, s).astype(np.complex64)
        except:
            pass

        # propagation to a random distance
        random_ratio = np.random.uniform(0, 1)
        self.update_operator(random_ratio)
        y = self.operator.forward(torch.tensor(comp_field))
        y = y.unsqueeze(0)  # [N, C=1, H, W]
        gt = np.stack((amp_img, phase_img), axis=0)  # [C=2, H, W]
        return y, torch.Tensor(gt), torch.Tensor([random_ratio]), torch.Tensor([self.prop_kernel['distance']])


if __name__ == "__main__":
    data_path = '/Users/zhangyunping/PycharmProjects/PATrans/ExpSample/cellsample'
    # data_path = "/Users/zhangyunping/PycharmProjects/PATrans/ExpSample/simUSAF"
    with open('/Users/zhangyunping/PycharmProjects/PATrans/configs/AdaptDH_C.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    prop_kernel = config['MEASUREMENT']['prop_kernel']
    # prop_kernel['pad_size'] = [1024, 1024]
    prop_kernel['pad_size'] = None
    prop_kernel['nx'] = 256
    prop_kernel['ny'] = 256
    dataset = AdpNetDataset(data_path, z_range=[1.065e-3, 1.065e-3], prop_kernel=prop_kernel, mode='A',
                            noise_level=None)
    dataloader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)
    for batch_i, (y, gt, random_ratio, random_distance) in enumerate(dataloader):
        break

    holo = y[0, 0, :, :]
    prop_kernel['pad_size'] = None
    A = generate_otf_torch(**prop_kernel)
    AT = torch.conj(A)
    bp = torch.fft.ifft2(torch.multiply(torch.fft.fft2(holo), AT))
    bp_amp = bp.abs()
    bp_phase = bp.angle()
    plt.imshow(holo, cmap='gray')
    plt.show()
    # plt.imsave('bp_amp.png', bp_amp, cmap='gray')
    # plt.imsave('bp_phase.png', bp_phase, cmap='gray')
    # plt.imsave('bp_phase_c.png', bp_phase, cmap='hot')


    # plt.subplot(121)
    # plt.imshow(bp_amp,cmap='gray')
    # plt.subplot(122)
    # plt.imshow(bp_phase,cmap='viridis')
    # plt.show()


