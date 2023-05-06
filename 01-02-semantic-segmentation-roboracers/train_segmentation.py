import argparse
import os
import shutil
import time
import sys

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

import cv2
import numpy as np
from torch.optim import SGD, Adam, lr_scheduler

from model import fast_scnn_model
from dataset_helper import freicar_segreg_dataloader
from dataset_helper import color_coder

from visdom import Visdom

#################################################################
# AUTHOR: Johan Vertens (vertensj@informatik.uni-freiburg.de)
# DESCRIPTION: Training script for FreiCAR semantic segmentation
##################################################################

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epoch',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')


def TensorImage1ToCV(data):
    cv = data.cpu().byte().data.numpy().squeeze()
    return cv

def visJetColorCoding(name, img):
    img = img.detach().cpu().squeeze().numpy()
    color_img = np.zeros(img.shape, dtype=img.dtype)
    cv2.normalize(img, color_img, 0, 255, cv2.NORM_MINMAX)
    color_img = color_img.astype(np.uint8)
    color_img = cv2.applyColorMap(color_img, cv2.COLORMAP_JET, color_img)
    cv2.imshow(name, color_img)

def visImage3Chan(data, name):
    cv = np.transpose(data.cpu().data.numpy().squeeze(), (1, 2, 0))
    cv = cv2.cvtColor(cv, cv2.COLOR_RGB2BGR)
    cv2.imshow(name, cv)

# def calculate_iou(seg_GT, prediction, batchsize = 1):
#     prediction = torch.argmax(prediction, 1)
#     intersection = torch.eq(seg_GT.squeeze(1), prediction)
#     ones_count = torch.count_nonzero(intersection)
#     #print(ones_count)
#     IoU = ones_count / (batchsize * prediction.shape[1] * prediction.shape[2])
#
#     return IoU

def calculate_iou(seg_GT, predication):
    numclasses = 4
    predication = torch.argmax(predication, 1)
    seg_GT = seg_GT.squeeze(1)
    intersection = seg_GT * torch.eq(seg_GT, predication)

    predication = predication.cpu().numpy()
    seg_GT = seg_GT.cpu().numpy()
    intersection = intersection.cpu().numpy()

    area_inter,_ = np.histogram(intersection, bins=numclasses, range=(0, 3))
    area_pred,_ = np.histogram(predication, bins=numclasses, range=(0, 3))
    area_truth,_ = np.histogram(seg_GT, bins=numclasses, range=(0, 3))
    area_union = (area_pred + area_truth) - area_inter
    IoU = area_inter / (np.spacing(1) + area_union)

    mIoU = IoU.mean()
    if mIoU > 1: mIoU = 0

    return mIoU


parser = argparse.ArgumentParser(description='Segmentation and Regression Training')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', default=0, type=int, help="Start at epoch X")
parser.add_argument('--batch_size', default=12, type=int, help="Batch size for training")
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--convert_torch', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

best_iou = 0
args = None
reg_weighing_factor = 16
is_best_iou = False
eval_freq = 2
# Number of max epochs, TODO: Change it to a reasonable number!
num_epochs = 16
train_iou_list = []
eval_iou_list = []
train_l1_list = []
eval_l1_list = []
plotter = VisdomLinePlotter(env_name='FreiCar Semantic Segmentation')


def main():
    global args, best_iou
    args = parser.parse_args()
    count = 0

    # Create Fast SCNN model...
    model = fast_scnn_model.Fast_SCNN(3, 4)
    model = model.cuda()


    optimizer = Adam(model.parameters(), 5e-3)
    lambda1 = lambda epoch: pow((1 - ((epoch - 1) / num_epochs)), 0.9)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    ############ Scripting Pytorch to Torchscript ########
    i = args.convert_torch
    if args.convert_torch:
        if os.path.isfile(os.path.join(os.path.abspath(os.getcwd()), i)):
            print("INFO: loading model weights ..... '{}'".format(args.resume))
            model.load_state_dict(torch.load(os.path.join(os.path.abspath(os.getcwd()), i))['state_dict'])
            scripted_model = torch.jit.script(model)
            scripted_model.save("scripted_model.pt")
            print("INFO: Saved to Torchscript successfully. Exiting ...")
            sys.exit(0)
        else:
            print("=> no weights found at '{}'".format(args.convert_torch))
            print("Please check the given path and try again.")

    ######################################

    i = args.resume
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(os.path.join(os.path.abspath(os.getcwd()), i)):
            print("=> loading checkpoint '{}'".format(args.resume))
            model.load_state_dict(torch.load(os.path.join(os.path.abspath(os.getcwd()), i))['state_dict'])
            args.start_epoch = 0

        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        args.start_epoch = 0

    # Data loading code
    load_real_images = False
    train_dataset = freicar_segreg_dataloader.FreiCarLoader("../data/", padding=(0, 0, 12, 12),
                                       split='training', load_real=load_real_images)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                               pin_memory=False, drop_last=True)

    #load evaluate data during training
    if not args.evaluate:
        eval_dataset = freicar_segreg_dataloader.FreiCarLoader("../data/", padding=(0, 0, 12, 12),
                                                               split='validation', load_real=load_real_images)

        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, shuffle=True, num_workers=1,
                                                  pin_memory=False, drop_last=False)

    # If --evaluate is passed from the command line --> evaluate
    if args.evaluate:
        print("Inside evaluate")
        eval_dataset = freicar_segreg_dataloader.FreiCarLoader("../data/", padding=(0, 0, 12, 12),
                                                                split='validation', load_real=load_real_images)

        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, shuffle=True, num_workers=1,
                                                   pin_memory=False, drop_last=False)

        eval(eval_loader, model)

    for epoch in range(args.start_epoch, num_epochs):
        # train for one epoch
        train(train_loader, model, optimizer, scheduler, epoch)
        plotter.plot('IOU Score', 'train', 'IoU', epoch, train_iou_list[epoch])
        plotter.plot('L1 Loss', 'train', 'L1 loss', epoch, train_l1_list[epoch])
        #evaluating every Nth Epoch;
        if epoch % eval_freq == 0:
            torch.cuda.empty_cache()
            eval(eval_loader, model)
            plotter.plot('IOU Score', 'eval', 'IoU', epoch, eval_iou_list[count])
            plotter.plot('L1 Loss', 'eval', 'L1 loss', epoch, eval_l1_list[count])
            count = count + 1

        # remember best iou and save checkpoint

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_iou': best_iou,
            'optimizer': optimizer.state_dict(),
        }, False)



def train(train_loader, model, optimizer, scheduler, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    IoUValues = AverageMeter()
    REGL1Values = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (sample) in enumerate(train_loader):

        data_time.update(time.time() - end)

        image = sample['rgb'].cuda().float()
        lane_reg = sample['reg'].cuda().float()
        seg_ids = sample['seg'].cuda()
        ######################################
        # TODO: Implement me! Train Loop
        ######################################

        optimizer.zero_grad()

        seg_ids_train, lane_reg_train = model(image)

        CE_loss = torch.nn.CrossEntropyLoss()
        RG_loss = torch.nn.L1Loss()

        cls_loss = CE_loss(seg_ids_train, seg_ids.long().squeeze(1))
        reg_loss = RG_loss(lane_reg_train, lane_reg) / reg_weighing_factor
        REGL1Values.update(reg_loss.detach().cpu().numpy())

        loss = cls_loss + reg_loss
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), image.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        trainIoU = calculate_iou(seg_ids, seg_ids_train)
        IoUValues.update(trainIoU)

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))

        if i+1 == len(train_loader):
            train_iou_list.append(IoUValues.avg)
            train_l1_list.append(REGL1Values.avg)
            print('train_iou_list', train_iou_list,' ','reg_l1_list', train_l1_list)

            # if epoch % eval_freq == 0:
            #     trainIoU = calculate_iou(seg_ids, seg_ids_train)
            #     IoUValues.update(trainIoU)
            #     print('Epoch: [{0}][{1}/{2}]\t'
            #          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            #          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #          'IoU Score {iou.val:.4f} ({iou.avg:.4f})\t'.format(
            #         epoch, i, len(train_loader), batch_time=batch_time,
            #         data_time=data_time, loss=losses, iou=IoUValues))

    scheduler.step(epoch)

def eval(data_loader, model):
    IoUValue_eval = AverageMeter()
    REGLoss_eval =AverageMeter()

    #switch to evaluation mode
    model.eval()
    color_conv = color_coder.ColorCoder()

    for i, (sample) in enumerate(data_loader):

        image = sample['rgb'].cuda().float()
        lane_reg = sample['reg'].cuda().float()
        seg_ids = sample['seg'].cuda()

        seg_eval, lreg_eval = model(image)
        ######################################
        # TODO: Implement me!
        # You should calculate the IoU every N
        # epochs for both the training
        # and the evaluation set.
        # For segmentation color coding
        # see: color_coder.py
        ######################################

        reg_loss_eval = torch.nn.L1Loss()
        reg_loss = reg_loss_eval(lreg_eval, lane_reg) / reg_weighing_factor

        reg_loss_cpu = reg_loss.detach().cpu().numpy()
        REGLoss_eval.update(reg_loss_cpu)

        evalIoU = calculate_iou(seg_ids, seg_eval)
        IoUValue_eval.update(evalIoU)

        if i+1 == len(data_loader):
            eval_iou_list.append(IoUValue_eval.avg)
            eval_l1_list.append(REGLoss_eval.avg)
            print('eval_iou_list', eval_iou_list, ' ', 'eval_l1_list', eval_l1_list)


        # pred_rgb_colorcoded = color_conv.color_code_labels(seg_eval)#
        # seg_ids_rgb_colorcoded = color_conv.color_code_labels(seg_ids, argmax=False)
        #
        # visJetColorCoding('Reg GT', lane_reg)
        # visJetColorCoding('Reg PRED', lreg_eval)
        #
        # print('Seg_ID', pred_rgb_colorcoded.shape, 'Pred', seg_ids_rgb_colorcoded.shape)
        # cv2.imshow('Segmentation Predection', pred_rgb_colorcoded)
        #
        # cv2.imshow('Segmentation Ground Truth', seg_ids_rgb_colorcoded)
        # cv2.waitKey()


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()
