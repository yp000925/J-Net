import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import normalized_mutual_information as nmi

def smooth_line_segment(x, y, window_length=9, polyorder=3):
    from scipy.signal import savgol_filter
    """
    Smooths a line segment defined by x and y coordinates using the Savitzky-Golay filter.

    Parameters:
        x (array-like): x-coordinates of the line segment.
        y (array-like): y-coordinates of the line segment.
        window_length (int): The length of the window used for filtering.
        polyorder (int): The order of the polynomial used for fitting.

    Returns:
        smoothed_x (ndarray): Smoothed x-coordinates of the line segment.
        smoothed_y (ndarray): Smoothed y-coordinates of the line segment.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")

    if len(x) < window_length:
        raise ValueError("The window length must be smaller than the number of data points.")

    smoothed_x = savgol_filter(x, window_length, polyorder)
    smoothed_y = savgol_filter(y, window_length, polyorder)

    return smoothed_x, smoothed_y
def ssim_score(img1, img2, **kwargs):
    return ssim(img1, img2, **kwargs)


def rescale_img(x, min_val=0, max_val=1):
    x_mins = x.min()
    x_maxs = x.max()
    rangex = x_maxs - x_mins
    if rangex == 0:
        return x
    else:
        scaled_data = (max_val - min_val) * ((x - x_mins) / rangex) + min_val
        return scaled_data

def mse_score(img1, img2, **kwargs):
    return mse(img1, img2, **kwargs)

def psnr_score(ref, test, **kwargs):
    return psnr(ref, test, **kwargs)

def nmi_score(img1, img2, **kwargs):
    return nmi(img1, img2, **kwargs)

def edge_detection(img):
    from skimage.filters import sobel
    return sobel(img)


def compare_curve(ref, test, label=None, ref_color = "g",test_color="r",yrange=None, show_axis=False,**kwargs):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(ref, label="GT", color=ref_color,  **kwargs)
    ax.plot(test, label=label, color=test_color, **kwargs)
    if yrange:
        ax.set_ylim(yrange)
    if label:
        plt.legend()
    if not show_axis:
        ax.yaxis.set_ticks([])
        ax.xaxis.set_ticks([])
    plt.show()
    return fig,ax

def shift_curve(ori_curve, level=0):
    mean = np.mean(ori_curve)
    shifted = ori_curve - (mean-level)
    return shifted

def SNR(img):
    m = np.mean(img)
    sd = np.std(img)
    return m/sd

def mean_edge_factor(edge_mtr):
    return np.mean(edge_mtr)


def BRISQUE_score(img_path,resized = None):
    '''
    BRISQUE is a no-reference image quality score.
    Blind/Referenceless Image Spatial Quality Evaluator (BRISQUE)
    '''
    from brisque import BRISQUE
    from PIL import  Image
    if resized:
        img_arr = np.array(Image.open(img_path).convert('RGB').resize(resized))
    else:
        img_arr = np.array(Image.open(img_path).convert('RGB'))
    obj = BRISQUE(url=False)
    score = obj.score(img_arr)
    return score


def total_variance(img):
    element = img.shape[0]*img.shape[1]
    diff1 = img[1:,:]-img[:-1,:]
    diff2 = img[:,1:]-img[:,:-1]

    res1 = np.sum(np.abs(diff1))
    res2 = np.sum(np.abs(diff2))
    score = res1+res2
    return score/element

def  mean_gradient_value(img):
    element = img.shape[0]*img.shape[1]
    diff1 = img[1:,:]-img[:-1,:]
    diff2 = img[:,1:]-img[:,:-1]

    res1 = np.sum(np.abs(diff1))
    res2 = np.sum(np.abs(diff2))
    score = (res1+res2)/element
    return score