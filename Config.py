import os
import torch
import time
## PARAMETERS OF THE MODEL
save_model = True
tensorboard = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = torch.cuda.is_available()
seed = 666
os.environ['PYTHONHASHSEED'] = str(seed)

kfold = 5
cosineLR = True
n_channels = 3
n_labels = 1
epochs = 500
img_size = 224
img_size2 = 224
print_frequency = 1
save_frequency = 5000
vis_frequency = 5000

task_name = 'GlaS'
# task_name = 'ISIC17'
# task_name = 'ISIC18'
# task_name = 'Synapse'

# model_name = 'UNet'
# model_name = 'MBSNet'
# model_name = 'MCFusionNet'
# model_name = 'R34_UNet'  
# model_name = 'AMSUnet'
# model_name = 'H2Former'
# model_name = 'TransUNet'
# model_name = 'CMUNeXt'
# model_name = 'AttUNet'
# model_name = 'UKAN'
# model_name = 'vmunet'
# model_name = 'UDTransNet'
# model_name = 'MCFusionNet'
# model_name = 'cswin_unet'
# model_name = 'MambaUnet'
model_name = 'MCFusionNet_wo_reverse_scale_fusion'

if task_name == 'GlaS':
    learning_rate = 1e-3
    batch_size = 4
    early_stopping_patience = 40
    print_frequency = 5
elif task_name == 'ISIC17':
    learning_rate = 1e-4
    batch_size = 20
    early_stopping_patience = 40
    print_frequency = 30



if task_name == "ISIC17":
    if model_name == "UNet":
        test_session = "Test_session_10.11_00h42"
    if model_name == "MCFusionNet":
        test_session = "Test_session_10.08_22h32" 
    if model_name == "MBSNet":
        test_session = "Test_session_10.09_08h51"
    if model_name == "CMUNeXt":
        test_session = "Test_session_10.11_05h58"
    if model_name == "R34_UNet":
        test_session = "Test_session_10.11_03h36"
    if model_name == "UDTransNet":
        test_session = "Test_session_10.11_23h17"
    if model_name == "AttUNet":
        test_session = "Test_session_10.13_01h58"
    if model_name == "vmunet":
        test_session = "Test_session_10.13_05h01"
    if model_name == "UKAN":
        test_session = "Test_session_10.13_09h39"
    if model_name == "H2Former":
        test_session = "Test_session_10.11_15h43"
    if model_name == "TransUNet":
        test_session = "Test_session_10.11_19h35"
    if model_name == "AMSUnet":
        test_session = "Test_session_10.11_08h47"
    if model_name == "cswin_unet":
        test_session = "Test_session_10.14_01h56" 
    if model_name == "MambaUnet":
        test_session = "Test_session_10.13_15h36" 


elif task_name == "GlaS":
    if model_name == "UNet":
        test_session = "Test_session_10.07_04h15"
    if model_name == "MCFusionNet":
        test_session = "Test_session_10.07_01h58" 
    if model_name == "MBSNet":
        test_session = "Test_session_10.07_04h34"
    if model_name == "CMUNeXt":
        test_session = "Test_session_10.07_05h15"
    if model_name == "R34_UNet":
        test_session = "Test_session_10.10_10h44"
    if model_name == "UDTransNet":
        test_session = "Test_session_10.07_03h10"
    if model_name == "AttUNet":
        test_session = "Test_session_10.10_14h16"
    if model_name == "vmunet":
        test_session = "Test_session_10.10_14h58"
    if model_name == "UKAN":
        test_session = "Test_session_10.10_15h24"
    if model_name == "H2Former":
        test_session = "Test_session_10.10_12h09"
    if model_name == "TransUNet":
        test_session = "Test_session_10.10_12h51"
    if model_name == "AMSUnet":
        test_session = "Test_session_10.10_11h36"
    if model_name == "cswin_unet":
        test_session = "Test_session_10.10_15h47" 
    if model_name == "MambaUnet":
        test_session = "Test_session_10.12_10h38"
    if model_name == "MCFusionNet_wo_reverse_scale_fusion":
        test_session = "Test_session_10.28_00h50"




if task_name == 'GlaS':
    train_dataset = "datasets/GlaS/Train_Folder/"
    test_dataset = "datasets/GlaS/Test_Folder/"
elif task_name == 'ISIC17':
    train_dataset = "/home/test/MC/isic2017/train/"
    test_dataset = "/home/test/MC/isic2017/val/"


session_name       = 'Test_session' + '_' + time.strftime('%m.%d_%Hh%M')
save_path          = task_name +'_kfold/'+ model_name +'/' + session_name + '/'
model_path         = save_path + 'models/'
tensorboard_folder = save_path + 'tensorboard_logs/'
logger_path        = save_path + session_name + ".log"
visualize_path     = save_path + 'visualize_val/'