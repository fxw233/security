import os
import time

import numpy as np
import torch
from torchvision import transforms


class Eval_thread():
    def __init__(self, loader, output_dir, cuda):
        self.loader = loader
        self.cuda = cuda
        self.output_dir = output_dir
        self.logfile = os.path.join(output_dir, 'result.txt')

    def run(self):
        auc, TPR, FPR = self.Eval_auc()
        TPR = TPR.cpu().numpy()
        FPR = FPR.cpu().numpy()
        print(TPR)
        print(FPR)
        return ' {:.4f} AUC.'.format(auc)


    def Eval_auc(self):

        avg_tpr, avg_fpr, avg_auc, img_num = 0.0, 0.0, 0.0, 0.0

        with torch.no_grad():
            trans = transforms.Compose([transforms.ToTensor()])
            for pred, gt in self.loader:
                if self.cuda:
                    pred = trans(pred).cuda()
                    pred = (pred - torch.min(pred)) / (torch.max(pred) -
                                                       torch.min(pred) + 1e-20)
                    gt = trans(gt).cuda()
                else:
                    pred = trans(pred)
                    pred = (pred - torch.min(pred)) / (torch.max(pred) -
                                                       torch.min(pred) + 1e-20)
                    gt = trans(gt)
                TPR, FPR = self._eval_roc(pred, gt, 255)
                avg_tpr += TPR
                avg_fpr += FPR
                img_num += 1.0
            avg_tpr = avg_tpr / img_num
            avg_fpr = avg_fpr / img_num

            sorted_idxes = torch.argsort(avg_fpr)
            avg_tpr = avg_tpr[sorted_idxes]
            avg_fpr = avg_fpr[sorted_idxes]
            avg_auc = torch.trapz(avg_tpr, avg_fpr) 

            return avg_auc.item(), avg_tpr, avg_fpr


    def _eval_roc(self, y_pred, y, num):
        if self.cuda:
            TPR, FPR = torch.zeros(num).cuda(), torch.zeros(num).cuda()
            thlist = torch.linspace(0, 1 - 1e-10, num).cuda()
        else:
            TPR, FPR = torch.zeros(num), torch.zeros(num)
            thlist = torch.linspace(0, 1 - 1e-10, num)
        for i in range(num):
            y_temp = (y_pred >= thlist[i]).float()
            tp = (y_temp * y).sum()
            fp = (y_temp * (1 - y)).sum()
            tn = ((1 - y_temp) * (1 - y)).sum()
            fn = ((1 - y_temp) * y).sum()

            TPR[i] = tp / (tp + fn + 1e-20)
            FPR[i] = fp / (fp + tn + 1e-20)

        return TPR, FPR
