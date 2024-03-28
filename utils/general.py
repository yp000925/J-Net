import logging
import yaml
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from .dataset_utils import prepross_bg

def get_logger():
    logger = logging.getLogger(name='Physics-Aware Transformer')
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s [%(name)s] >> %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def parse_task(exp_name):
    if exp_name == "SimUSAF_complex":
        img = Image.open('data/simUSAF/holo_complex.png').convert('L')
        processed_img = np.array(img).astype(np.float32)
        plt.imshow(processed_img)
        plt.title("diffraction pattern")
        plt.show()
        gt_img = Image.open('data/simUSAF/GT.png').convert('L')
        gt = np.array(gt_img).astype(np.float32)
        gt = gt / gt.max()
        plt.imshow(gt)
        plt.title("ground truth")
        plt.show()
        with open('data/simUSAF/params.json','r') as f:
            params = json.load(f)
        deltax = params['deltax']
        deltay = params['deltay']
        distance = params['distance']
        w = params['w']
        nx,ny = processed_img.shape
        prop_kernel = dict(wavelength=w, deltax=deltax, deltay=deltay, distance=distance, nx=nx, ny=ny)
        return {'prop_kernel': prop_kernel, 'measurement': processed_img,  'gt': [gt,gt] }

    elif exp_name == "SimUSAF_phase":
        img = Image.open('data/simUSAF/holo_phase.png').convert('L')
        processed_img = np.array(img).astype(np.float32)
        plt.imshow(processed_img,cmap='gray')
        plt.title("diffraction pattern")
        plt.show()
        gt_img = Image.open('data/simUSAF/GT.png').convert('L')
        gt = np.array(gt_img).astype(np.float32)
        gt = gt/gt.max()*0.5
        plt.imshow(gt,cmap='gray')
        plt.title("ground truth")
        plt.show()
        with open('data/simUSAF/params.json','r') as f:
            params = json.load(f)
        deltax = params['deltax']
        deltay = params['deltay']
        distance = params['distance']
        w = params['w']
        nx,ny = processed_img.shape
        prop_kernel = dict(wavelength=w, deltax=deltax, deltay=deltay, distance=distance, nx=nx, ny=ny)
        return {'prop_kernel': prop_kernel, 'measurement': processed_img,  'gt':[np.ones_like(gt),gt] }

    elif exp_name == "SimUSAF_intensity":
        img = Image.open('data/simUSAF/holo_intensity.png').convert('L')
        processed_img = np.array(img).astype(np.float32)
        plt.imshow(processed_img)
        plt.title("diffraction pattern")
        plt.show()
        gt_img = Image.open('data/simUSAF/GT.png').convert('L')
        gt = np.array(gt_img).astype(np.float32)
        gt = gt / gt.max()
        plt.imshow(gt)
        plt.title("ground truth")
        plt.show()
        with open('data/simUSAF/params.json', 'r') as f:
            params = json.load(f)
        deltax = params['deltax']
        deltay = params['deltay']
        distance = params['distance']
        w = params['w']
        nx, ny = processed_img.shape
        prop_kernel = dict(wavelength=w, deltax=deltax, deltay=deltay, distance=distance, nx=nx, ny=ny)
        return {'prop_kernel': prop_kernel, 'measurement': processed_img, 'gt': [gt,np.zeros_like(gt)]}


    elif exp_name == "SimBio_intensity":
        img = Image.open('data/simBio/holo_intensity.png').convert('L')
        processed_img = np.array(img).astype(np.float32)
        plt.imshow(processed_img,cmap='gray')
        plt.title("diffraction pattern")
        plt.show()
        gt_img = Image.open('data/simBio/GT.png').convert('L')
        gt = np.array(gt_img).astype(np.float32)
        gt = gt/gt.max()
        plt.imshow(gt,cmap='gray')
        plt.title("ground truth")
        plt.show()
        with open('data/simUSAF/params.json','r') as f:
            params = json.load(f)
        deltax = params['deltax']
        deltay = params['deltay']
        distance = params['distance']
        w = params['w']
        nx,ny = processed_img.shape
        prop_kernel = dict(wavelength=w, deltax=deltax, deltay=deltay, distance=distance, nx=nx, ny=ny)
        return {'prop_kernel': prop_kernel, 'measurement': processed_img,  'gt':[gt,np.zeros_like(gt)] }

    elif exp_name == "SimBio_phase":
        img = Image.open('data/simBio/holo_phase.png').convert('L')
        processed_img = np.array(img).astype(np.float32)
        plt.imshow(processed_img,cmap='gray')
        plt.title("diffraction pattern")
        plt.show()
        gt_img = Image.open('data/simBio/GT.png').convert('L')
        gt = np.array(gt_img).astype(np.float32)
        gt = gt/gt.max()*0.5
        plt.imshow(gt,cmap='gray')
        plt.title("ground truth")
        plt.show()
        with open('data/simUSAF/params.json','r') as f:
            params = json.load(f)
        deltax = params['deltax']
        deltay = params['deltay']
        distance = params['distance']
        w = params['w']
        nx,ny = processed_img.shape
        prop_kernel = dict(wavelength=w, deltax=deltax, deltay=deltay, distance=distance, nx=nx, ny=ny)
        return {'prop_kernel': prop_kernel, 'measurement': processed_img,  'gt':[np.ones_like(gt),gt] }
    else:
        raise ValueError("Invalid experiment name, please declare the experiment in the parse_task function.")
