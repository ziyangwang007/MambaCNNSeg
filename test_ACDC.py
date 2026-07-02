import os
import sys
import logging
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from utils_acdc.utils import test_single_volume
from utils_acdc.dataset_ACDC import ACDCdataset, RandomGenerator
from models.Mamba_CNN import MBUnet as MCUnet
from models.vmunet.vmunet import VMUNet
from models.vision_mamba import MambaUnet as MambaUnet
from models.Mamba_CNN import MBUnet as MCUnet
from models.two_encode import Two_encode
from models.UNet import UNet,R34_UNet
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

        
def inference(args, model, testloader, test_save_path=None):
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    with torch.no_grad():
        for i_batch, sampled_batch in tqdm(enumerate(testloader)):
            # print(sampled_batch["image"].shape)
            # print(sampled_batch["label"].shape)
            h, w = sampled_batch["image"].size()[2:]
            image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
            metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                          test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
            metric_list += np.array(metric_i)
            logging.info('idx %d case %s mean_dice %f mean_hd95 %f, mean_jacard %f mean_asd %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1], np.mean(metric_i, axis=0)[2], np.mean(metric_i, axis=0)[3]))
        metric_list = metric_list / len(testloader)
        for i in range(1, args.num_classes):
            logging.info('Mean class (%d) mean_dice %f mean_hd95 %f, mean_jacard %f mean_asd %f' % (i, metric_list[i-1][0], metric_list[i-1][1], metric_list[i-1][2], metric_list[i-1][3]))
        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        mean_jacard = np.mean(metric_list, axis=0)[2]
        mean_asd = np.mean(metric_list, axis=0)[3]
        logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f, mean_jacard : %f mean_asd : %f' % (performance, mean_hd95, mean_jacard, mean_asd))
        logging.info("Testing Finished!")
        return performance, mean_hd95, mean_jacard, mean_asd
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='MCFusion', help="model")
    parser.add_argument("--batch_size", default=12, help="batch size")
    parser.add_argument("--lr", default=0.0001, help="learning rate")
    parser.add_argument("--max_epochs", default=150)
    parser.add_argument("--img_size", default=224)
    parser.add_argument("--save_path", default="./model_pth/ACDC")
    parser.add_argument("--n_gpu", default=1)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--list_dir", default="/home/test/MC/ACDC/lists_ACDC")
    parser.add_argument("--root_dir", default="/home/test/MC/ACDC/")
    parser.add_argument("--volume_path", default="/home/test/MC/ACDC/test")
    parser.add_argument("--z_spacing", default=10)
    parser.add_argument("--num_classes", default=4)
    parser.add_argument('--test_save_dir', default='./predictions', help='saving prediction as nii!')
    parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
    parser.add_argument('--seed', type=int,
                    default=2024, help='random seed')              
    args = parser.parse_args()

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.is_pretrain = True
    args.exp = args.model_name + str(args.img_size)
    snapshot_path = "{}/{}".format(args.save_path, args.exp)
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.lr) if args.lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

    if args.model_name == 'UNet':
        net = UNet(n_channels=1, n_classes=4)
    elif args.model_name == 'R34_UNet':
        net = R34_UNet(n_channels=1, n_classes=4)
    elif args.model_name == 'MCFusion':
        net = MCUnet(224,1)
    elif args.model_name == 'MBSNet':
        net = MBSNet(1,4)
    elif args.model_name == 'CMUNeXt':
        net = CMUNeXt(1,4)
    elif args.model_name == 'AMSUnet':
        net = AMSUnet(1,4)
    elif args.model_name == 'AttUNet':
        net = AttUNet(1,4)
    elif args.model_name == 'H2Former':
        net = Res34_Swin_MS(in_ch=1, num_classes=4)
    elif args.model_name == 'TransUNet':
        net = get_transNet(4)
    elif args.model_name == 'UKAN':
        net = UKAN(4,1,False,img_size=args.img_size)
    elif args.model_name == 'UDTransNet':
        config_vit = get_model_config()
        net = UDTransNet(config_vit,n_channels=1,n_classes=4,img_size=args.img_size)
    elif args.model_name == 'vmunet':
        net = VMUNet(
                num_classes=4,
                input_channels=3,
                depths=[2,2,2,2],
                depths_decoder=[2,2,2,1],
                drop_path_rate=0.2,
                load_ckpt_path='pretrained_ckpt/vmamba_small_e238_ema.pth',
            )
        net.load_from()  
    elif args.model_name == 'cswin_unet':
        config_cswin_unet = get_cswin_unet_config()
        net = ViT_seg(config_cswin_unet, img_size=224, num_classes=4)
        net.load_from() 
    elif args.model_name == 'MambaUnet':
        net = MambaUnet(224,4) 
    elif args.model_name == 'MCFusionNet_wo_reverse_scale_fusion':
        net = MCUnet(224,4)
    net = net.cuda() 
    snapshot = os.path.join(snapshot_path, 'best.pth')
    # snapshot = 'model_pth/ACDC/MCFusion224_pretrain_epo150_bs12_lr0.0001_224_s2024_最好/best.pth'
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best', 'epoch_'+str(args.max_epochs-1))
    net.load_state_dict(torch.load(snapshot))
    snapshot_name = snapshot_path.split('/')[-1]

    log_folder = 'test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    args.test_save_dir = os.path.join(snapshot_path, args.test_save_dir)
    test_save_path = args.test_save_dir
    print(test_save_path)
    os.makedirs(test_save_path, exist_ok=True)
    
    
    db_test =ACDCdataset(base_dir=args.volume_path,list_dir=args.list_dir, split="test")
    testloader = DataLoader(db_test, batch_size=1, shuffle=False)
    
    results = inference(args, net, testloader, test_save_path)


