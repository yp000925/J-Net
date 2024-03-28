'''
add more loss function
'''

import argparse
import random
import tqdm
from utils.dataset_utils import *
from utils.general import *
from utils.model_utils import *
from model.JNet import JNet
from utils.loss import *
seed = 11
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)

def train(holo, epoch, model, optim, device, save_dir, **kwargs):
    H, W = holo.shape
    best_psnr = 0
    best_loss = 1e5
    pbar = tqdm.tqdm(range(epoch))
    ref_amp = kwargs.get('ref_amp', torch.ones((H,W)))
    ref_phase = kwargs.get('ref_phase', torch.zeros((H,W)))
    scale = kwargs.get('scale', [20, 0, 0])
    use_ref = kwargs.get('use_ref', False)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=2, T_mult=2)
    eval_period = kwargs.get('eval_period', 200)



    plt.imshow(ref_amp,cmap='gray')
    plt.title("reference amplitude")
    plt.show()
    plt.imshow(ref_phase,cmap= 'gray')
    plt.title("reference phase")
    plt.show()
    for step in pbar:
        y_hat = model(holo.expand(1, 1, H, W).to(device))
        pred_depth = model.prop_kernel['distance']
        pred_phase = model.phase_pred
        pred_amp = model.amp_pred  # output is positive
        mse_loss = nn.functional.mse_loss(y_hat[0, 0, :, :], holo.to(device))
        # tv_loss = TV_LOSS(y_hat)
        tv_loss_amp = TV_LOSS(pred_amp)
        tv_loss_phase = TV_LOSS(pred_phase)
        loss = scale[0]*mse_loss +scale[1]* tv_loss_amp + scale[2]*tv_loss_phase

        optim.zero_grad()
        loss.backward()
        optim.step()
        ckpt = {
            'model_state_dict': model.state_dict(),
            'last_epoch': step,
            'optimizer': optim.state_dict(),
            'loss': loss.item()
        }
        # print to screen as evaluation metric
        if use_ref:
            _psnr_p = psnr(pred_phase[0, 0, :, :], ref_phase.to(device))
            _psnr_a = psnr(pred_amp[0, 0, :, :], ref_amp.to(device))
            _psnr = (_psnr_p + _psnr_a)/2
            _psnr = _psnr.item()
            if _psnr > best_psnr:
                best_psnr = _psnr
                torch.save(model.state_dict(), os.path.join(save_dir, 'best.pt'))
                print('\tSaved best model at step {}  Loss: {:.4f} PSNR: {:.4f}'.format(step + 1, loss.item(),
                                                                                _psnr))
            pbar.set_description(
                    'mse: {:.4f} tv_a:{:.4f} tv_p:{:.4f} loss: {:.4f} PSNR:{:.4f} Z:{:.3f}'.format(
                        mse_loss.item(),
                        tv_loss_amp.item(),
                        tv_loss_phase.item(), loss.item(), _psnr,
                        1000 * pred_depth[0, 0].item()))

        else:
            _psnr = 0
            if loss.cpu().data.numpy() < best_loss:
                best_loss = loss.cpu().data.numpy()
                torch.save(model.state_dict(), os.path.join(save_dir, 'best.pt'))
                print('\tSaved best model at step {}  Loss: {}'.format(step + 1, loss.cpu().data.numpy()))

            pbar.set_description('mse: {:.4f} tv_a:{:.4f} tv_p:{:.4f} loss: {:.4f} Z:{:.3f}'.format(mse_loss.cpu().data.numpy(),
                                                                                                                             tv_loss_amp.cpu().data.numpy(),
                                                                                                                             tv_loss_phase.cpu().data.numpy(),loss.cpu().data.numpy(), 1000*pred_depth[0,0].cpu().data.numpy()))

        if ((step+1) % eval_period) == 0:
            # save amp
            netout = pred_amp[0, 0, :, :].cpu().data.numpy()
            netout = (netout - np.min(netout))/(np.max(netout)-np.min(netout))
            plt.imshow(netout, cmap='gray')
            plt.show()
            netout = netout * 255
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            plt.imsave(os.path.join(save_dir, f'Amp_{step+1}.png'), netout, cmap='gray')

            #save phase map
            netout = pred_phase[0, 0, :, :].cpu().data.numpy()
            netout = (netout - np.min(netout))/(np.max(netout)-np.min(netout))
            plt.imshow(netout, cmap='gray')
            plt.show()
            netout = netout * 255
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            plt.imsave(os.path.join(save_dir, f'Phase_{step+1}.png'), netout, cmap='gray')
            plt.imsave(os.path.join(save_dir, f'Phase_{step+1}_c.png'), netout, cmap='viridis')
            print('\tSaved middle result')

    torch.save(ckpt, os.path.join(save_dir, 'last.pt'))
    return model,pred_amp, pred_phase,pred_depth
def AssignPropKernel(train_config, prop_kernel):
    out_prop_kernel = train_config['MEASUREMENT']['prop_kernel']
    for key in prop_kernel.keys():
        out_prop_kernel[key] = prop_kernel.get(key, out_prop_kernel[key])
    train_config['MEASUREMENT']['prop_kernel'] = out_prop_kernel
    return train_config

if __name__ =="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', default='configs/JNet.yaml', type=str)
    parser.add_argument('--task_config', default='configs/simUSAF_complex.yaml', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results')
    try:
        args = parser.parse_args()
    except:
        args = parser.parse_args(args=[])
    # logger
    logger = get_logger()

    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)

    with open(args.task_config) as f:
        task_config = yaml.load(f,Loader=yaml.FullLoader)
    # parse task
    task = parse_task(task_config['TASK_NAME'])
    prop_kernel = task['prop_kernel']
    holo = task['measurement']
    gt_amp = task['gt'][0]
    gt_phase = task['gt'][1]
    with open(args.model_config) as f:
        model_config = yaml.load(f,Loader=yaml.FullLoader)
    # update the default prop_kernel with the specified task prop_kernel
    model_config = AssignPropKernel(model_config, prop_kernel)

    # parse the training configuration
    epoch = task_config['PARAMS']['EPOCH']
    scale = task_config['PARAMS']['SCALE']
    eval_period = task_config['PARAMS']['EVAL_PERIOD']
    use_ref = task_config['USE_REF']


    model = JNet(model_config).to(device)
    model.summary()
    optim = torch.optim.Adam(model.parameters(), lr=task_config['PARAMS']['LR'])

    A = generate_otf_torch(**prop_kernel)
    AT = torch.conj(A)
    save_dir = os.path.join(args.save_dir, task_config['TASK_NAME'])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    holo = holo/255.0 # normalize the holo
    bp = torch.fft.ifft2(torch.multiply(torch.fft.fft2(torch.tensor(holo)), AT))
    bp = bp.abs()
    plt.imshow(bp)
    plt.title("backpropagation")
    plt.show()

    model, pred_amp, pred_phase, pred_depth = train(torch.tensor(holo), epoch, model, optim, device, save_dir, scale=scale,eval_period=eval_period,
                                                    ref_amp=torch.tensor(gt_amp),ref_phase=torch.tensor(gt_phase), use_ref= use_ref)


