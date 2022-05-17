"""
训练
"""

import argparse
import logging
from pathlib import Path
# import time
import os
import numpy as np
import cv2
# import matplotlib.pyplot as plt

import torch
# from torch.utils import data
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# import torchvision
from torchvision import datasets, transforms
# import torchvision.utils as vutils

from dataset.datasets import WLFWDatasets
from models.pfld import PFLDInference, AuxiliaryNet
from pfld.loss import PFLDLoss
from pfld.utils import AverageMeter

from config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_args(args):
    for arg in vars(args):
        s = arg + ': ' + str(getattr(args, arg))
        logging.info(s)

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    logging.info('Save checkpoint to {0:}'.format(filename))

def str2bool(v: str):
    """
    Transform string arguments to bool type
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')

def draw_batch(landmarks:np.ndarray, imgs:torch.Tensor, writer:SummaryWriter, tag:str, step:int, draw_landmarks=True):
    """
    绘制图片和landmarks
    """
    if draw_landmarks:
        lmrz=landmarks.reshape(landmarks.shape[0], -1,2).cpu().numpy()
    img_in=np.asarray(imgs)
    img_in=np.transpose(img_in,(0,2,3,1))
    img_in = (img_in * 255).astype(np.uint8)
    np.clip(img_in, 0, 255)
    img_out=np.zeros(img_in.shape,dtype=np.uint8)
    for i in range(img_in.shape[0]):
        img_i=img_in[i]
        img_i=cv2.cvtColor(img_i,cv2.COLOR_BGR2RGB)
        if draw_landmarks:
            pre_landmark=lmrz[i]*[112,112]
            for (x,y) in pre_landmark.astype(np.int32):
                cv2.circle(img_i,(x,y),1,(14,184,180),-1)
        img_out[i]=img_i

    writer.add_images(tag,img_out, step,dataformats='NHWC')

def train(train_loader, pfld_backbone, auxiliaryNet, criterion, optimizer, epoch ,writer, args):
    losses=AverageMeter()
    weighted_loss, loss=None,None
    to_draw = True
    for img, landmark_gt, attribute_gt, euler_angle_gt in train_loader:
        if to_draw and args.show_before_train:
            to_draw = False
            draw_batch(landmark_gt, img, writer, "train/ground_true", epoch, True)
        img = img.to(device)
        attribute_gt = attribute_gt.to(device)
        landmark_gt = landmark_gt.to(device)
        euler_angle_gt = euler_angle_gt.to(device)
        pfld_backbone = pfld_backbone.to(device)
        auxiliaryNet = auxiliaryNet.to(device)
        # 计算loss
        features, landmarks = pfld_backbone(img)
        angle = auxiliaryNet(features)
        weighted_loss, loss = criterion(attribute_gt, landmark_gt, euler_angle_gt, angle, landmarks, args.train_batch_size)
        # 更新参数
        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()
    
        losses.update(loss.item())

    return weighted_loss, loss

def validate(val_loader, pfld_backbone, auxiliaryNet, criterion, epoch, writer, args):
    to_draw = True
    pfld_backbone.eval()
    auxiliaryNet.eval()
    losses = []
    with torch.no_grad():
        for img, landmark_gt, attribute_gt, euler_angle_gt in val_loader:
            img = img.to(device)
            attribute_gt = attribute_gt.to(device)
            landmark_gt = landmark_gt.to(device)
            euler_angle_gt = euler_angle_gt.to(device)
            pfld_backbone = pfld_backbone.to(device)
            auxiliaryNet = auxiliaryNet.to(device)

            _, landmark = pfld_backbone(img)
            loss = torch.mean(torch.sum((landmark_gt - landmark)**2, axis=1)) # l2 distance
            losses.append(loss.cpu().numpy()) 
            # 展示每轮结果
            if to_draw and args.show_each_epoch:
                to_draw = False
                draw_batch(landmark, img.cpu(), writer,"train/val", epoch, True)
    print("===> Evaluate:")
    print('Eval set: Average loss: {:.4f} '.format(np.mean(losses)))
    return np.mean(losses)

def main(args):
    # step0: Create dirs and files
    (log_dir, log_file) = os.path.split(LOG_FILE)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(LOG_FILE,'w'):
        pass
    if not os.path.exists(TENSORBOARD):
        os.makedirs(TENSORBOARD)
    if not os.path.exists(args.snapshot):
        os.makedirs(args.snapshot)
    # Step1: parse args and config
    logging.basicConfig(
        format='[%(asctime)s] [p%(process)s] [%(pathname)s:%(lineno)d] [%(levelname)s] %(message)s',
        level=logging.INFO,
        handlers=[logging.FileHandler(args.log_file, mode='w'), logging.StreamHandler()]
    )
    print_args(args)

    # Step2: modle, criterion, optimizer, scheduler
    print("--------Loading modle, criterion, optimizer, scheduler...-----------")
    pfld_backbone = PFLDInference().to(device)
    auxiliaryNet = AuxiliaryNet().to(device)
    criterion  = PFLDLoss()
    optimizer = torch.optim.Adam(
        [{'params': pfld_backbone.parameters()},{'params': auxiliaryNet.parameters()}],
        lr=args.base_lr,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=args.lr_patience,verbose=True
    )
    # 加载检查点模型
    print("--------Loading checkpoint--------")
    if args.resume:
        checkpoint = torch.load(args.resume)
        auxiliaryNet.load_state_dict(checkpoint["auxilirynet"]) # key不太对，但已经保存了，没关系了
        pfld_backbone.load_state_dict(checkpoint["pfld_backbone"])
        args.start_epoch = checkpoint["epoch"]
    # Step3: data
    print("----------Setting Dataloaders--------")
    transform = transforms.Compose([transforms.ToTensor()]) # 数据格式转换
    wlfwdataset = WLFWDatasets(args.dataroot, transform)
    dataloader=DataLoader(wlfwdataset, batch_size=args.train_batch_size,shuffle=True, num_workers=args.workers,drop_last=False)
    wlfw_val_dataset = WLFWDatasets(args.val_dataroot,transform)
    wlfw_val_dataloader = DataLoader(wlfw_val_dataset, batch_size=args.val_batch_size,shuffle=False,num_workers=0)
    # Have a look (我觉得最好还是放在预处理部分)
    # if args.show_original_image:
    #     pass
    # Step4: train loop
    print("--------Training-------")
    writer=SummaryWriter(args.tensorboard)
    for epoch in range(args.start_epoch,args.end_epoch):
        # 训练，并获得loss
        weighted_train_loss, train_loss = train(dataloader, pfld_backbone, auxiliaryNet, criterion, optimizer, epoch, writer, args)
        # 保存检查点
        checkpoint_filename=os.path.join(args.snapshot, "checkpoint_epoch_"+str(epoch)+'.pth.tar')
        save_checkpoint(
            {
                'epoch':epoch,
                'pfld_backbone':pfld_backbone.state_dict(),
                'auxilirynet':auxiliaryNet.state_dict()
            },
            checkpoint_filename
        )
        # 验证
        val_loss = validate(wlfw_val_dataloader, pfld_backbone,auxiliaryNet, criterion, epoch, writer, args)
        scheduler.step(val_loss)
        # loss写入tensorboard
        writer.add_scalar('data/weighted_loss', weighted_train_loss, epoch)
        writer.add_scalars('data/loss', {'val loss': val_loss,'train loss': train_loss}, epoch)
        writer.close()

def parse_args():
# 获取参数
    parser=argparse.ArgumentParser()
    parser.add_argument('-j', '--workers', default=WORKERS, type=int)
    parser.add_argument('--devices_id', default=DEVICES_ID, type=str)  #TBD
    parser.add_argument('--test_initial', default=TEST_INITIAL, type=str2bool)  #TBDaining
    ##  -- optimizer
    parser.add_argument('--base_lr', default=BASE_LR, type=int)
    parser.add_argument('--weight_decay', '--wd', default=WEIGHT_DECAY, type=float)
    # -- lr
    parser.add_argument("--lr_patience", default=LR_PATIENCE, type=int)
    # -- epoch
    parser.add_argument('--start_epoch', default=START_EPOCH, type=int)
    parser.add_argument('--end_epoch', default=END_EPOCH, type=int)
    # -- snapshot、tensorboard log and checkpoint
    parser.add_argument('--snapshot', default=SNAPSHOT, type=str, metavar='PATH')
    parser.add_argument('--log_file', default=LOG_FILE, type=str)
    parser.add_argument('--tensorboard', default=TENSORBOARD, type=str)
    parser.add_argument('--resume', default=RESUME,type=str, metavar='PATH')
    # --dataset
    parser.add_argument('--dataroot', default=DATA_ROOT, type=str, metavar='PATH')
    parser.add_argument('--val_dataroot', default=VAL_DATAROOT, type=str, metavar='PATH')
    parser.add_argument('--train_batch_size', default=TRAIN_BATCH_SIZE, type=int)
    parser.add_argument('--val_batch_size', default=VAL_BATCH_SIZE, type=int)
    # --version
    parser.add_argument('--show_original_image', default=SHOW_ORIGINAL_IMAGE, type=str2bool)
    # parser.add_argument('--show_original_image', default=VAL_BATCH_SIZE, type=int)
    parser.add_argument('--show_before_train', default=SHOW_BEFORE_TRAIN, type=str2bool)
    parser.add_argument('--show_each_epoch', default=SHOW_EACH_EPOCH, type=str2bool)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)