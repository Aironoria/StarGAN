import torch
import math
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self,val, n=1):
        self.val =val
        self.sum += val * n
        self.count += n
        self.avg = self.sum /self.count


def psnr_score(mse):
    return 20* math.log10(1 / mse)
