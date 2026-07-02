from medpy import metric
from scipy.ndimage import zoom
from sklearn.metrics import confusion_matrix
import torchvision.transforms
import torch.optim
from Load_Dataset import ValGenerator, ImageToImage2D_kfold
from torch.utils.data import DataLoader
import warnings
import time
warnings.filterwarnings("ignore")
import Config as config
import matplotlib.pyplot as plt
from tqdm import tqdm
from nets.UNet import *
from nets.UDTransNet import UDTransNet
from nets.TF_configs import get_model_config

from models.vmunet.vmunet import VMUNet
from models.vision_mamba import MambaUnet as MambaUnet
from models.Mamba_CNN import MBUnet as MCUnet
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
import pylab
import os
from utils import *
import cv2


def show_ens(predict_save,input_img, labs, save_path):
    fig, ax = plt.subplots()
    plt.imshow(predict_save, cmap='gray')
    plt.axis("off")
    height, width = predict_save.shape
    fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(save_path, dpi=300)
    plt.close()

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        iou = metric.binary.jc(pred, gt)
        return dice, iou
    elif pred.sum()==0 and gt.sum()==0:
        return 1, 1
    else:
        return 0, 0

def show_image_with_dice(predict_save, labs, save_path):

    
    tmp_lbl = (labs).astype(np.float32)
    tmp_3dunet = (predict_save).astype(np.float32)

    dice_pred = 2 * np.sum(tmp_lbl * tmp_3dunet) / (np.sum(tmp_lbl) + np.sum(tmp_3dunet) + 1e-5)

    iou_pred = jaccard_score(tmp_lbl.reshape(-1),tmp_3dunet.reshape(-1))
    return dice_pred, iou_pred


def vis_and_save_heatmap(ensemble_models, input_img, img_RGB, labs,lab_img, vis_save_path):
    outputs = []
    dice_pred, iou_pred = [], []
    dice_pred_my, iou_pred_my, acc_pred_my, se_pred_my, mae_pred_my= [],[],[],[],[]
    for model_ in ensemble_models:
        output= model_(input_img.cuda())
        pred_class = torch.where(output>0.5,torch.ones_like(output),torch.zeros_like(output))
        predict_save = pred_class[0].cpu().data.numpy()
        outputs.append(predict_save)
        dice_pred_tmp, iou_tmp = show_image_with_dice(predict_save, labs, save_path=vis_save_path+'_predict'+model_type+'.jpg')
        dice_pred.append(dice_pred_tmp)
        iou_pred.append(iou_tmp)

        output = output.cpu().detach().numpy()
        preds = np.array(output).reshape(-1)
        gts = np.array(labs).reshape(-1)

        y_pre = np.where(preds>=0.5, 1, 0)
        y_true = np.where(gts>=0.5, 1, 0)
        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        mae = np.mean(np.abs(y_true - y_pre))
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0
        dice_pred_my.append(f1_or_dsc)
        iou_pred_my.append(miou)
        acc_pred_my.append(accuracy)
        se_pred_my.append(sensitivity)
        mae_pred_my.append(mae)

    predict_save = np.array(outputs).mean(0)
    predict_save = np.reshape(predict_save, (config.img_size, config.img_size))
    predict_save = np.where(predict_save>0.5,1,0)
    show_ens(predict_save, img_RGB, lab_img, save_path=vis_save_path+'_pred5f_'+model_type+'.jpg')
    return dice_pred, iou_pred,dice_pred_my,iou_pred_my,acc_pred_my,se_pred_my,mae_pred_my




if __name__ == '__main__':
    ## PARAMS
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    ensemble_models=[]
    test_session = config.test_session

    for i in range(0,5):
        if config.task_name == "GlaS":
            test_num = 80
            model_type = config.model_name
            model_path = "./GlaS_kfold/"+model_type+"/"+test_session+"/models/fold_"+str(i+1)+"/best_model-"+model_type+".pth.tar"

        elif config.task_name == "ISIC17":
            test_num = 650
            model_type = config.model_name
            model_path = "./ISIC17_kfold/"+model_type+"/"+test_session+"/models/fold_"+str(i+1)+"/best_model-"+model_type+".pth.tar"


        save_path = config.task_name +'/'+ model_type +'/' + test_session + '/'

        att_vis_path = "./" + config.task_name + '_visualize_test/'

        if not os.path.exists(att_vis_path):
            os.makedirs(att_vis_path)

        maxi = 5


        checkpoint = torch.load(model_path, map_location='cuda')


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
        if torch.cuda.device_count() > 1:
            print ("Let's use {0} GPUs!".format(torch.cuda.device_count()))
            model = nn.DataParallel(model, device_ids=[0,1,2,3])

        model.load_state_dict(checkpoint['state_dict'])
        print('Model loaded !')
        model.eval()
        ensemble_models.append(model)

    
    filelists = os.listdir(config.test_dataset+"/img")

    tf_test = ValGenerator(output_size=[config.img_size, config.img_size])
    test_dataset = ImageToImage2D_kfold(config.test_dataset,
                                        tf_test,
                                        image_size=config.img_size,
                                        task_name=config.task_name,
                                        filelists=filelists,
                                        split='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    test_num = len(test_loader)
    dice_pred = np.zeros((maxi))
    iou_pred = np.zeros((maxi))

    dice_list= np.zeros((maxi))
    iou_list = np.zeros((maxi))
    acc_list= np.zeros((maxi))
    se_list = np.zeros((maxi))
    mae_list= np.zeros((maxi))

    dice_ens = 0.0
    dice_5folds = []
    iou_5folds = []
    end = time.time()
    with tqdm(total=test_num, desc='Test visualize', unit='img', ncols=70, leave=True) as pbar:
        for i, (sampled_batch, names) in enumerate(test_loader, 1):
            test_data, test_label = sampled_batch['image'], sampled_batch['label']

            arr=test_data.numpy()
            arr = arr.astype(np.float32())
            lab=test_label.data.numpy()
            if config.task_name == 'GLaS':
                img_lab = np.reshape(lab, (lab.shape[1], lab.shape[2])) * 255
         
            else:
                img_lab = np.reshape(lab, (lab.shape[1], lab.shape[2])) * 255
    

            fig, ax = plt.subplots()
            plt.imshow(img_lab, cmap='gray')
            plt.axis("off")
            height, width = config.img_size, config.img_size
            fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.savefig(att_vis_path+str(i)+"_lab.jpg", dpi=300)
            plt.close()

            img_RGB = cv2.imread(config.test_dataset+"img/"+names[0],1)
            img_RGB = cv2.resize(img_RGB,(config.img_size,config.img_size))

            if config.task_name == "ISIC17":
                lab_img = cv2.imread(config.test_dataset+"labelcol/"+names[0][:-4]+"_segmentation.png",0)
                lab_img = cv2.resize(lab_img,(config.img_size,config.img_size))
            else:
                lab_img = cv2.imread(config.test_dataset+"labelcol/"+names[0][:-3]+"png",0)
                lab_img = cv2.resize(lab_img,(config.img_size,config.img_size))
            input_img = torch.from_numpy(arr)

            dice_pred_t,iou_pred_t,dice,iou,acc,se,mae = vis_and_save_heatmap(ensemble_models, input_img, img_RGB, lab, lab_img,
                                                            att_vis_path+str(i))

            dice_pred_t = np.array(dice_pred_t)
            iou_pred_t = np.array(iou_pred_t)

            dice_pred+=dice_pred_t
            iou_pred+=iou_pred_t



            dice_list += dice
            iou_list += iou
            acc_list += acc
            se_list += se
            mae_list += mae
            


            torch.cuda.empty_cache()
            pbar.update()
    inference_time = (time.time() - end)/test_num
    print("inference_time",inference_time)
    dice_pred = dice_pred/test_num * 100.0
    iou_pred = iou_pred/test_num * 100.0

    dice_pred_my = dice_list/test_num * 100.0
    iou_pred_my = iou_list/test_num * 100.0
    acc_pred_my = acc_list/test_num * 100.0
    se_pred_my = se_list/test_num * 100.0
    mae_pred_my = mae_list/test_num 


    np.set_printoptions(formatter={'float': '{:.4f}'.format})
    print ("dice_5folds:",dice_pred)
    print ("iou_5folds:",iou_pred)
    print ("dice_my_5folds:",dice_pred_my)
    print ("iou_my_5folds:",iou_pred_my)
    print ("acc_my_5folds:",acc_pred_my)
    print ("se_my_5folds:",se_pred_my)
    print ("mae_my_5folds:",mae_pred_my)

    dice_pred_mean = dice_pred.mean()
    iou_pred_mean = iou_pred.mean()
    dice_pred_my_mean = dice_pred_my.mean()
    iou_pred_my_mean = iou_pred_my.mean()
    acc_pred_my_mean = acc_pred_my.mean()
    se_pred_my_mean = se_pred_my.mean()
    mae_pred_my_mean = mae_pred_my.mean()


    dice_pred_std = np.std(dice_pred,ddof=1)
    iou_pred_std = np.std(iou_pred,ddof=1)

    dice_pred_my_std = np.std(dice_pred_my,ddof=1)
    iou_pred_my_std = np.std(iou_pred_my,ddof=1)
    acc_pred_my_std = np.std(acc_pred_my,ddof=1)
    se_pred_my_std = np.std(se_pred_my,ddof=1)
    mae_pred_my_std = np.std(mae_pred_my,ddof=1)

    print ("dice: {:.2f}+{:.2f}".format(dice_pred_mean, dice_pred_std))
    print ("iou: {:.2f}+{:.2f}".format(iou_pred_mean, iou_pred_std))

    print ("dice: {:.2f}+{:.2f}".format(dice_pred_my_mean, dice_pred_my_std))
    print ("iou: {:.2f}+{:.2f}".format(iou_pred_my_mean, iou_pred_my_std))
    print ("acc {:.2f}+{:.2f}".format(acc_pred_my_mean, acc_pred_my_std))
    print ("se: {:.2f}+{:.2f}".format(se_pred_my_mean, se_pred_my_std))
    print ("mae: {:.4f}+{:.4f}".format(mae_pred_my_mean, mae_pred_my_std))
