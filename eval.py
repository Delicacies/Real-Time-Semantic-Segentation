import os
import torch
import torch.utils.data as data
import time # added

from torchvision import transforms
from data_loader import get_segmentation_dataset
from models.fast_scnn import get_fast_scnn
from utils.metric import SegmentationMetric
from utils.visualize import get_color_pallete
from demo import decode_labels #added @yjy
from PIL import Image  #added
#from model.ContextNet import get_ContextNet  # added
#from utils_yjy.metrics import Evaluator as Eval  #added

from train import parse_args


class Evaluator(object):
    def __init__(self, args):
        self.args = args
        # output folder
        self.outdir = 'test_result'
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        # dataset and dataloader
        val_dataset = get_segmentation_dataset(args.dataset, split='val', mode='testval', #'train'
                                               transform=input_transform)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_size=1,
                                          shuffle=False)
        #print(self.val_loader)
        # create network
        self.model = get_fast_scnn(args.dataset, aux=args.aux, pretrained=True, root=args.save_folder).to(args.device)
        #self.model = get_ContextNet(args.dataset, aux=args.aux,pretrained=True, root=args.save_folder).to(args.device)
        print('Finished loading model!')

        self.metric = SegmentationMetric(val_dataset.num_class)

    def eval(self):
        self.model.eval()
        #evaluator = Eval(19)  #added
        for i, (image, label) in enumerate(self.val_loader):
            image = image.to(self.args.device)
            #print(image)

            start_time = time.time()
            outputs = self.model(image)
            duration = time.time() - start_time
            #print(outputs)

            pred = torch.argmax(outputs[0], 1)
            pred = pred.cpu().data.numpy()
            label = label.numpy()

            self.metric.update(pred, label)
            pixAcc, mIoU = self.metric.get()
            print('Sample %d, validation pixAcc: %.3f%%, mIoU: %.3f%%,fps: %.1f' % (i + 1, pixAcc * 100, mIoU * 100,1.0/duration))

            predict = pred.squeeze(0)
            mask = decode_labels(predict)  #added
            mask = Image.fromarray(mask)  #added
            #mask = get_color_pallete(predict, self.args.dataset)
            mask.save(os.path.join(self.outdir, 'seg_{}.png'.format(i+1)))  # original=i
            
            '''
            # Fast test during the training
            Acc = evaluator.Pixel_Accuracy()
            Acc_class = evaluator.Pixel_Accuracy_Class()
            mIoU = evaluator.Mean_Intersection_over_Union()
            FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
        #self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        #self.writer.add_scalar('val/mIoU', mIoU, epoch)
        #self.writer.add_scalar('val/Acc', Acc, epoch)
        #self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        #self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
            print('Validation:')
        #print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
            print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        #print('Loss: %.3f' % test_loss)
            
        '''

if __name__ == '__main__':
    args = parse_args()
    evaluator = Evaluator(args)
    print('Testing model: ', args.model)
    evaluator.eval()
