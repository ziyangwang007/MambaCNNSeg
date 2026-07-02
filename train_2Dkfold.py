import os
import torch
import argparse
import torch.optim
from torch.backends import cudnn
from tensorboardX import SummaryWriter
import numpy as np
import time
import random
from utils import BinaryDiceBCE,MultiClassDiceCE, CosineAnnealingWarmRestarts
from sklearn.model_selection import KFold
from Load_Dataset import RandomGenerator,ValGenerator, ImageToImage2D_kfold
from torch.utils.data import DataLoader
from nets.UNet import UNet,R34_UNet
from nets.UDTransNet import UDTransNet
from nets.TF_configs import get_model_config

from models.vmunet.vmunet import VMUNet
from models.vision_mamba import MambaUnet as MambaUnet
from models.Mamba_CNN import MBUnet as MCUnet
from models.two_encode import Two_encode
# from models.UNet import UNet
from models.Seg_UKAN.archs import UKAN
from models.H2Former.H2Former import Res34_Swin_MS
from models.ConvNextUNet.network import ConvNextUNet
from models.MAXFormer import MAXFormer
from models.MISSFormer.MISSFormer import MISSFormer
from models.MT_UNet import MTUNet
from models.UDTransNet.UDTransNet import UDTransNet
from models.UDTransNet.TF_configs import get_model_config
from models.MBSNet import MBSNet
from models.SwinUNet.SwinUNet import SwinUnet,SwinUnet_config
from models.TransUNet.TransUNet import get_transNet
from models.AMSUNet import AMSUnet
from models.CMUNeXt import CMUNeXt
from models.attenUNet import AttUNet

from models.cswin_UNet.config import get_cswin_unet_config
from models.cswin_UNet.vision_transformer import CSwinUnet as ViT_seg


import logging
import warnings
from train_one_epoch import train_one_epoch
warnings.filterwarnings("ignore")
# import Config as config

def logger_config(log_path):
    loggerr = logging.getLogger()
    loggerr.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    loggerr.addHandler(handler)
    loggerr.addHandler(console)
    return loggerr

def save_checkpoint(state, save_path):
    '''
        Save the current model.
        If the model is the best model since beginning of the training
        it will be copy
    '''
    logger.info('\t Saving to {}'.format(save_path))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    epoch = state['epoch']  # epoch no
    best_model = state['best_model']  # bool
    model = state['model']  # model type
    if best_model:
        filename = save_path + '/' + 'best_model-{}.pth.tar'.format(model)
    else:
        filename = save_path + '/' + 'model-{}-{:02d}.pth.tar'.format(model, epoch)
    torch.save(state, filename)

def worker_init_fn(worker_id):
    random.seed(config.seed + worker_id)

##################################################################################
#=================================================================================
#          Main Loop
#=================================================================================
##################################################################################
def main_loop(train_loader,val_loader, batch_size, model_type='', fold=0, tensorboard=True, kfold=0):

    lr = learning_rate
    logger.info(model_type)
    if model_type == 'UNet':
        model = UNet(n_channels=config.n_channels,n_classes=config.n_labels)
    elif model_type == 'R34_UNet':
        model = R34_UNet(n_channels=config.n_channels,n_classes=config.n_labels)
    elif model_type == 'MCFusionNet':
        model = MCUnet(224,1)
    elif model_type == 'MCFusionNet_wo_reverse_scale_fusion':
        model = MCUnet(224,1)
    elif model_type == 'MBSNet':
        model = MBSNet(3,1)
    elif model_type == 'CMUNeXt':
        model = CMUNeXt(3,1)
    elif model_type == 'AMSUnet':
        model = AMSUnet(3,1)
    elif model_type == 'AttUNet':
        model = AttUNet(3,1)
    elif model_type == 'H2Former':
        model = Res34_Swin_MS(in_ch=3, num_classes=1)
    elif model_type == 'TransUNet':
        model = get_transNet(1)
    elif model_type == 'UKAN':
        model = UKAN(1,3,False,img_size=config.img_size)
    elif model_type == 'UDTransNet':
        config_vit = get_model_config()
        model = UDTransNet(config_vit,n_channels=3,n_classes=1,img_size=config.img_size)
    elif model_type == 'vmunet':
        model = VMUNet(
                num_classes=1,
                input_channels=3,
                depths=[2,2,2,2],
                depths_decoder=[2,2,2,1],
                drop_path_rate=0.2,
                load_ckpt_path='pretrained_ckpt/vmamba_small_e238_ema.pth',
            )
        model.load_from()  
    elif model_type == 'cswin_unet':
        config_cswin_unet = get_cswin_unet_config()
        model = ViT_seg(config_cswin_unet, img_size=config.img_size, num_classes=1)
        model.load_from()
    elif model_type == 'MambaUnet':
        model = MambaUnet(224,1)
        model.load_from()
    else: raise TypeError('Please enter a valid name for the model type')

    model = model.cuda()

    if config.n_labels == 1:
        criterion = BinaryDiceBCE(dice_weight=1,BCE_weight=1)
    else:
        criterion = MultiClassDiceCE(num_classes=config.n_labels)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Choose optimize

    if config.cosineLR is True:
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2, eta_min=0)
    else:
        lr_scheduler =  None

    if tensorboard:
        log_dir = tensorboard_folder
        logger.info('log dir: '.format(log_dir))
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    max_dice = 0.0
    best_epoch = 1
    for epoch in range(config.epochs):  # loop over the dataset multiple times
        logger.info('\n========= {} | Fold [{}/{}] | Epoch [{}/{}] ========='.format(config.model_name,fold,kfold, epoch + 1, config.epochs + 1))
        logger.info(session_name)
        # train for one epoch
        model.train(True)
        logger.info('Training with batch size : {}'.format(batch_size))
        train_one_epoch(train_loader, model, criterion, optimizer, writer, epoch, None, fold, kfold, logger,print_frequency)
        # evaluate on validation set
        logger.info('Validation')
        with torch.no_grad():
            model.eval()
            val_loss, val_dice = train_one_epoch(val_loader, model, criterion,
                                                          optimizer, writer, epoch, lr_scheduler,fold,kfold,logger,print_frequency)

        # =============================================================
        #       Save best model
        # =============================================================
        if val_dice > max_dice:
            if epoch+1 > 1:
                logger.info('\t Saving best model, mean dice increased from: {:.4f} to {:.4f}'.format(max_dice,val_dice))
                max_dice = val_dice
                best_epoch = epoch + 1
                save_checkpoint({'epoch': epoch,
                                 'best_model': True,
                                 'model': model_type,
                                 'state_dict': model.state_dict(),
                                 'val_loss': val_loss,
                                 'optimizer': optimizer.state_dict()}, model_path+"fold_"+str(fold)+"/")
            else:pass
        elif val_dice == 0:
            best_epoch = epoch + 1
            logger.info('\t Reset count number')
        else:
            logger.info('\t Mean dice:{:.4f} does not increase, '
                        'the best is still: {:.4f} in epoch {}'.format(val_dice,max_dice, best_epoch))
        early_stopping_count = epoch - best_epoch + 1
        logger.info('\t early_stopping_count: {}/{}'.format(early_stopping_count,early_stopping_patience))

        if early_stopping_count >  early_stopping_patience:
            logger.info('\t early_stopping!')
            break

    return max_dice


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', default="ISIC17", help='Dataset name') 
    parser.add_argument('--model_name', default="AttUNet", help='Network name')
    parser.add_argument('--kfold', default=5)
    parser.add_argument('--cosineLR', default=True)
    parser.add_argument('--n_channels', default=3)
    parser.add_argument('--n_labels', default=1)
    parser.add_argument('--epochs', default=500)
    parser.add_argument('--img_size', default=224)
    parser.add_argument('--print_frequency', default=1)
    parser.add_argument('--save_frequency', default=5000)
    parser.add_argument('--vis_frequency', default=5000)
    parser.add_argument('--save_model', default=True)
    parser.add_argument('--tensorboard', default=True)
    parser.add_argument('--seed', default=666)
                  
    config = parser.parse_args()
    
    session_name       = 'Test_session' + '_' + time.strftime('%m.%d_%Hh%M')
    save_path          = config.task_name +'_kfold/'+ config.model_name +'/' + session_name + '/'
    model_path         = save_path + 'models/'
    tensorboard_folder = save_path + 'tensorboard_logs/'
    logger_path        = save_path + session_name + ".log"
    visualize_path     = save_path + 'visualize_val/'


    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    use_cuda = torch.cuda.is_available()
    os.environ['PYTHONHASHSEED'] = str(config.seed)
    deterministic = False # set `True' can make the results reproducible, but costs more training time
    if not deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    logger = logger_config(log_path=logger_path)

    if config.task_name == 'GlaS':
        learning_rate = 1e-3
        batch_size = 4
        early_stopping_patience = 40
        print_frequency = 5
        train_dataset_path = "datasets/GlaS/Train_Folder/"
        test_dataset_path = "datasets/GlaS/Test_Folder/"
    if config.task_name == 'ISIC17':
        learning_rate = 1e-4
        batch_size = 20
        early_stopping_patience = 40
        print_frequency = 30
        train_dataset_path = "/home/test/MC/isic2017/train/"
        test_dataset_path  = "/home/test/MC/isic2017/val/"


    filelists = os.listdir(train_dataset_path+"img")
    filelists = np.array(filelists)
    kfold = config.kfold
    kf = KFold(n_splits=kfold, shuffle=True, random_state=config.seed)
    dice_list = []
    iou_list = []

    for fold, (train_index, val_index) in enumerate(kf.split(filelists)):
        train_filelists = filelists[train_index]
        val_filelists = filelists[val_index]
        np.savetxt(save_path+"val_fold_"+str(fold+1)+".txt", val_filelists,'%s')
        logger.info("Total Nums: {}, train: {}, val: {}".format(len(filelists), len(train_filelists), len(val_filelists)))

        train_tf= RandomGenerator(output_size=[config.img_size, config.img_size])

        val_tf = ValGenerator(output_size=[config.img_size, config.img_size])
        train_dataset = ImageToImage2D_kfold(train_dataset_path,
                                             train_tf,
                                             image_size=config.img_size,
                                             filelists=train_filelists,
                                             task_name=config.task_name)
        val_dataset = ImageToImage2D_kfold(train_dataset_path,
                                           val_tf,
                                           image_size=config.img_size,
                                           filelists=val_filelists,
                                           task_name=config.task_name)
        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  worker_init_fn=worker_init_fn,
                                  num_workers=8,
                                  pin_memory=True)
        val_loader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                worker_init_fn=worker_init_fn,
                                num_workers=8,
                                pin_memory=True)

        dice = main_loop(train_loader,val_loader, batch_size=batch_size, model_type=config.model_name, fold=fold+1, tensorboard=True, kfold=kfold)
        dice_list.append(dice.item())

    dice=0.0
    for j in range(len(dice_list)):
        logging.info("fold {0}: {1:2.4f}".format(j+1, dice_list[j]))
        dice+=dice_list[j]
    logging.info("mean dice: {:.4f} \n".format(dice/kfold))







