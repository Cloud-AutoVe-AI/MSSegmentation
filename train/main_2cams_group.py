####### import modules ######
import os
import random
import time
import numpy as np
import torch
import math
import lovasz_losses as L
import importlib
from shutil import copyfile


from PIL import Image
from argparse import ArgumentParser

from torch.optim import SGD, Adam, lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, ColorJitter, ToTensor, ToPILImage
import torchvision.transforms as T

from dataset import cityscapes, KITTI360_2CAMs
from transform import Relabel, ToLabel
from iouEval import iouEval


###### pre-processing images ###### 
class MyCoTransform(object):
    def __init__(self, augment=True, resize_width=1280, crop_height=512, crop_width=512):
        self.augment = augment
        self.resize_width = resize_width
        self.crop_height = crop_height
        self.crop_width = crop_width        
        pass
    def __call__(self, input, input_r, target):
        re_width = int(self.resize_width)
        re_height = int(input.size[1] * re_width/input.size[0] +0.5)        

        if(self.augment):
            # Random hflip
            if (random.random() < 0.5):
                input = input.transpose(Image.FLIP_LEFT_RIGHT)
                input_r = input_r.transpose(Image.FLIP_LEFT_RIGHT)
                target = target.transpose(Image.FLIP_LEFT_RIGHT)
            # color jitter
            input = ColorJitter(hue=.1, saturation=.1)(input)
            input_r = ColorJitter(hue=.1, saturation=.1)(input_r)
            
            # resize image
            if(random.random() < 0.5):
                resize_factor = (0.8 + 0.25 * random.random())
                re_width = int(resize_factor * re_width + 0.5)
                re_height = int(resize_factor * re_height + 0.5)

            else:
                resize_factor = (0.95 + 0.25 * random.random())
                re_width = int(resize_factor * re_width + 0.5)
                re_height = int(resize_factor * re_height + 0.5)                           

        input =  Resize((int(re_height), int(re_width)), T.InterpolationMode.BILINEAR)(input)
        input_r =  Resize((int(re_height), int(re_width)), T.InterpolationMode.BILINEAR)(input_r)
        target = Resize((int(re_height), int(re_width)), T.InterpolationMode.NEAREST)(target)              
        transX = random.randint(0, re_width-self.crop_width)            
        transY = random.randint(0, re_height-self.crop_height)

        if re_width < self.crop_width or re_height < self.crop_height:
            print('crop size must be smaller than input size !')
            return
        
        input = input.crop((transX, transY, transX+self.crop_width, transY+self.crop_height))
        input_r = input_r.crop((transX, transY, transX+self.crop_width, transY+self.crop_height))
        target = target.crop((transX, transY, transX+self.crop_width, transY+self.crop_height))    
        
        input_r =  Resize((int(self.crop_height/2), int(self.crop_width/2)), T.InterpolationMode.BILINEAR)(input_r)
        
        input = ToTensor()(input)
        input_r = ToTensor()(input_r)
        target = ToLabel()(target)

        target = Relabel(0, 1)(target)        
        target = Relabel(1, 1)(target)
        target = Relabel(2, 2)(target)
        target = Relabel(3, 2)(target)
        target = Relabel(4, 2)(target)
        target = Relabel(5, 3)(target)
        target = Relabel(6, 3)(target)
        target = Relabel(7, 3)(target)
        target = Relabel(8, 4)(target)
        target = Relabel(9, 4)(target)
        target = Relabel(10, 5)(target)
        target = Relabel(11, 6)(target)
        target = Relabel(12, 6)(target)
        target = Relabel(13, 7)(target)
        target = Relabel(14, 7)(target)
        target = Relabel(15, 7)(target)
        target = Relabel(16, 7)(target)
        target = Relabel(17, 7)(target)
        target = Relabel(18, 7)(target)
        target = Relabel(255, 0)(target)
            
        return input, input_r, target

def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
    own_state = model.state_dict()

    for name, param in own_state.items():
#         name = name[7:]
        if name not in state_dict:
            print('not loaded:', name)
            continue
        own_state[name].copy_(param)
    return model        
    
#     for name, param in state_dict.items():
# #         name = name[7:]
#         if name not in own_state:
#             print('not loaded:', name)
#             continue
#         own_state[name].copy_(param)
#     return model    


class CrossEntropyLoss2d(torch.nn.Module):

    def __init__(self, weight=None):
        super().__init__()

        self.loss = torch.nn.NLLLoss(weight)

    def forward(self, outputs, targets):
        return self.loss(torch.nn.functional.log_softmax(outputs, dim=1), targets)

###### training procedure ######
def train(args, model):

    
    weight = torch.ones(args.num_classes)
    weight[0] = 20.0
    weight[1] = 2.8149201869965	
    weight[2] = 7.8698215484619
    weight[3] = 9.5110931396484	
    weight[4] = 4.6323022842407	
    weight[5] = 7.8698215484619
    weight[6] = 9.5168733596802	
    weight[7] = 6.6616044044495	
   

    if args.cuda:
        weight = weight.cuda()
    criterion = CrossEntropyLoss2d(weight)
        
    assert os.path.exists(args.datadir1), "Error: datadir1 (dataset directory) could not be loaded"

    ### input data pre-processing ###
    co_transform = MyCoTransform(augment=True, crop_height=args.crop_height, crop_width=args.crop_width, resize_width=args.resize_width)
    co_transform_val = MyCoTransform(augment=False, crop_height=args.crop_height, crop_width=args.crop_width, resize_width=args.resize_width)
#     dataset_train = cityscapes(args.datadir1, co_transform, 'train')
#     dataset_val = cityscapes(args.datadir1, co_transform, 'val')
    dataset_train = KITTI360_2CAMs(args.datadir1, co_transform, 'train')
    dataset_val = KITTI360_2CAMs(args.datadir1, co_transform, 'val')
    loader = DataLoader(dataset_train, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True,drop_last=True)
    loader_val = DataLoader(dataset_val, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,drop_last=True)

    savedir = f'../save/{args.savedir}'

    automated_log_path = savedir + "/automated_log.txt"
    modeltxtpath = savedir + "/model.txt"
    
    if (not os.path.exists(automated_log_path)):    #dont add first line if it exists 
        with open(automated_log_path, "a") as myfile:
            myfile.write("Epoch\t\tTrain-loss\t\tTest-loss\t\tTrain-IoU\t\tTest-IoU\t\tlearningRate")

    with open(modeltxtpath, "w") as myfile:
        myfile.write(str(model))

    optimizer = Adam(model.parameters(), args.init_learningrate, (0.9, 0.999),  eps=1e-08, weight_decay=1e-4)
    start_epoch = 1
    best_acc = 0
    best_epoch = 1
    if args.resume:
        filenameCheckpoint = savedir + '/checkpoint.pth.tar'
        assert os.path.exists(filenameCheckpoint), "Error: resume option was used but checkpoint was not found in folder"
        checkpoint = torch.load(filenameCheckpoint)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc = checkpoint['best_acc']
        print("=> Loaded checkpoint at epoch {})".format(checkpoint['epoch']))
  
   
    lambda1 = lambda epoch: pow((1-((epoch-1)/args.num_epochs)),0.9)  ## scheduler 2
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)                             ## scheduler 2

    for epoch in range(start_epoch, args.num_epochs+1):
        print("----- TRAINING - EPOCH", epoch, "-----")

        epoch_loss = []
        time_train = []
     
        doIouVal =  args.iouVal

        usedLr = 0
        for param_group in optimizer.param_groups:
            print("LEARNING RATE: ", param_group['lr'])
            usedLr = float(param_group['lr'])

        model.train()
        for step, (images, images2, labels) in enumerate(loader):

            start_time = time.time()

            if args.cuda:
                images = images.cuda()
                images2 = images2.cuda()
                labels = labels.cuda()

            inputs = Variable(images)
            inputs2 = Variable(images2)
            targets = Variable(labels)

            outputs = model(inputs, inputs2, False)

            optimizer.zero_grad()

            if args.loss_type=='CE':
                loss = criterion(outputs, targets[:, 0])
            if args.loss_type=='lovasz':
                outputs2 = torch.nn.functional.softmax(outputs, dim=1)     
                loss = L.lovasz_softmax(outputs2, targets, ignore=args.ignore_class, only_present=True) 

            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
            time_train.append(time.time() - start_time)

            if args.steps_loss > 0 and step % args.steps_loss == 0:
                average = sum(epoch_loss) / len(epoch_loss)
                print(f'loss: {average:0.4} (epoch: {epoch}, step: {step})', 
                        "// Avg time/img: %.4f s" % (sum(time_train) / len(time_train) / args.batch_size))

        average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
        iouTrain = 0
        
        print("----- VALIDATING - EPOCH", epoch, "-----")
        model.eval()
        epoch_loss_val = []
        time_val = []

        
        iouEvalVal = iouEval(args.num_classes,ignoreIndex=args.ignore_class)

        for step, (images, images2, labels) in enumerate(loader_val):
            start_time = time.time()
            if args.cuda:
                images = images.cuda()
                images2 = images2.cuda()
                labels = labels.cuda()
            with torch.no_grad():
                inputs = Variable(images)    #volatile flag makes it free backward or outputs for eval
                inputs2 = Variable(images2)
                targets = Variable(labels)
                
            outputs = model(inputs, inputs2, False)
            
            if args.loss_type=='CE':
                loss = criterion(outputs, targets[:, 0])            
            if args.loss_type=='lovasz':
                outputs2 = torch.nn.functional.softmax(outputs, dim=1)            
                loss = L.lovasz_softmax(outputs2, targets, ignore=args.ignore_class, only_present=True) 

            
            epoch_loss_val.append(loss.item())
            time_val.append(time.time() - start_time)

            iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)

            if args.steps_loss > 0 and step % args.steps_loss == 0:
                average = sum(epoch_loss_val) / len(epoch_loss_val)
                print(f'VAL loss: {average:0.4} (epoch: {epoch}, step: {step})', 
                        "// Avg time/img: %.4f s" % (sum(time_val) / len(time_val) / args.batch_size))
                       

        average_epoch_loss_val = sum(epoch_loss_val) / len(epoch_loss_val)

        iouVal = 0
        if (doIouVal):
            iouVal, iou_classes = iouEvalVal.getIoU()
            iouStr = '{:0.2f}'.format(iouVal*100) + '\033[0m'
            print ("EPOCH IoU on VAL set: ", iouStr, "%") 

        if iouVal == 0:
            current_acc = -average_epoch_loss_val
        else:
            current_acc = iouVal 
        is_best = current_acc > best_acc
        best_acc = max(current_acc, best_acc)
        filenameCheckpoint = savedir + '/checkpoint.pth.tar'
        filenameBest = savedir + '/model_best.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best, filenameCheckpoint, filenameBest)

        #SAVE MODEL AFTER EPOCH

        filename = f'{savedir}/model-{epoch:03}.pth'
        filenamebest = f'{savedir}/model_best.pth'
        
        if args.epochs_save > 0 and step > 0 and step % args.epochs_save == 0:
            torch.save(model.state_dict(), filename)
            print(f'save: {filename} (epoch: {epoch})')
        if (is_best):
            best_epoch = epoch
            torch.save(model.state_dict(), filenamebest)
            print(f'save: {filenamebest} (epoch: {epoch})')

            with open(savedir + "/best.txt", "w") as myfile:
                myfile.write("Best epoch is %d, with Val-IoU= %.4f" % (epoch, iouVal))   
         

        #SAVE TO FILE A ROW WITH THE EPOCH RESULT (train loss, val loss, train IoU, val IoU)
        #Epoch		Train-loss		Test-loss	Train-IoU	Test-IoU		learningRate
        with open(automated_log_path, "a") as myfile:
            myfile.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.8f" % (epoch, average_epoch_loss_train, average_epoch_loss_val, iouTrain, iouVal, usedLr ))
        scheduler.step()    ## scheduler 2
        print ("Best mIoU is", best_acc.item(), "at", best_epoch , "-th epoch" ) 
    return(model)   #return model (convenience for encoder-decoder training)

def save_checkpoint(state, is_best, filenameCheckpoint, filenameBest):
    torch.save(state, filenameCheckpoint)
    if is_best:
        print ("Saving model as best")
        torch.save(state, filenameBest)


def main(args):
    savedir = f'../save/{args.savedir}'

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    with open(savedir + '/opts.txt', "w") as myfile:
        myfile.write(str(args))

    #Load Model
    assert os.path.exists(args.model + ".py"), "Error: model definition not found"
    model_file = importlib.import_module(args.model)
    model = model_file.Net(args.num_classes)
    copyfile(args.model + ".py", savedir + '/' + "network.py")
    
    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()
    
    if args.load_init_model:                    
        filenameCheckpoint = args.loaddir + '/model_best.pth.tar'   
        weight_path = args.loaddir + '/model_best.pth'
        assert os.path.exists(filenameCheckpoint), "Error: load_init_model option was used but checkpoint was not found in folder"
        checkpoint = torch.load(filenameCheckpoint)
        load_my_state_dict(model, torch.load(weight_path))
#         model.load_state_dict(checkpoint['state_dict'], strict=False)             
        print("=> mIoU of loaded checkpoint : {})".format(checkpoint['best_acc']))           

    print("========== EnDecoder TRAINING ===========")    
    model = train(args, model)   #Train decoder
    print("========== TRAINING FINISHED ===========")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=True)  
    parser.add_argument('--model', default="network")

    parser.add_argument('--num-workers', type=int, default=12)
    parser.add_argument('--steps-loss', type=int, default=50)    
    parser.add_argument('--epochs-save', type=int, default=0)    #You can use this value to save model every X epochs
    parser.add_argument('--savedir', required=True)    
    parser.add_argument('--iouVal', action='store_true', default=True) # 
    parser.add_argument('--resume', action='store_true')    #Use this flag to load last checkpoint for training  
    parser.add_argument('--load_init_model', action='store_true')    #Use this flag to load certain checkpoint for training  
    parser.add_argument('--loaddir', default=None)
      
    parser.add_argument('--datadir1', default=None)
    parser.add_argument('--datadir2', default=None)
    parser.add_argument('--datadir3', default=None)
    parser.add_argument('--datadir4', default=None)    
    parser.add_argument('--num-epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=24)

    parser.add_argument('--init_learningrate', type=float, default=5e-4)     
    parser.add_argument('--loss_type', default='lovasz')
    parser.add_argument('--crop_height', type=int, default=384)    
    parser.add_argument('--crop_width', type=int, default=384)        
    parser.add_argument('--num_classes', type=int, default=8)
    parser.add_argument('--ignore_class', type=int, default=-1)
    parser.add_argument('--resize_width', type=int, default=1280)    
    main(parser.parse_args())
    
    