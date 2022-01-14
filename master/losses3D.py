import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from scipy.ndimage import morphology


class WeightedDiceLoss(nn.Module):
    def __init__(self, weights):
        super(WeightedDiceLoss, self).__init__()
        self.weights = weights

    def forward(self, inputs, targets, smooth=1e-5):

        w = self.weights

        intersections = (inputs * targets).sum(dim=[2, 3, 4])

        cardinalities = (inputs + targets).sum(dim=[2, 3, 4])

        dice = ((2 * (w * intersections).sum(dim=1) + smooth) / ((w * cardinalities).sum(dim=1) + smooth)).mean()

        return 1 - dice


class VectorFieldSmoothness(nn.Module):
    def __init__(self, h=1):
        super(VectorFieldSmoothness, self).__init__()
        self.h = h
        print('Warning: Implementation for isotropic voxel spacing.')

    def forward(self, deformation_field):

        h = self.h

        # Penalize deformations in non-corresponding regions ten times higher
        # to avoid compensation of intensity differences by spatial deformations
        d1u1 = F.pad((deformation_field[:, 0:1, :, 2:, :] -
                      deformation_field[:, 0:1, :, :-2, :]), (0, 0, 1, 1, 0, 0), mode='replicate') / (2 * h)
        d2u1 = F.pad((deformation_field[:, 0:1, :, :, 2:] -
                      deformation_field[:, 0:1, :, :, :-2]), (1, 1, 0, 0, 0, 0), mode='replicate') / (2 * h)
        d3u1 = F.pad((deformation_field[:, 0:1, 2:, :, :] -
                      deformation_field[:, 0:1, :-2, :, :]), (0, 0, 0, 0, 1, 1), mode='replicate') / (2 * h)

        d1u2 = F.pad((deformation_field[:, 1:2, :, 2:, :] -
                      deformation_field[:, 1:2, :, :-2, :]), (0, 0, 1, 1, 0, 0), mode='replicate') / (2 * h)
        d2u2 = F.pad((deformation_field[:, 1:2, :, :, 2:] -
                      deformation_field[:, 1:2, :, :, :-2]), (1, 1, 0, 0, 0, 0), mode='replicate') / (2 * h)
        d3u2 = F.pad((deformation_field[:, 1:2, 2:, :, :] -
                      deformation_field[:, 1:2, :-2, :, :]), (0, 0, 0, 0, 1, 1), mode='replicate') / (2 * h)

        d1u3 = F.pad((deformation_field[:, 2:3, :, 2:, :] -
                      deformation_field[:, 2:3, :, :-2, :]), (0, 0, 1, 1, 0, 0), mode='replicate') / (2 * h)
        d2u3 = F.pad((deformation_field[:, 2:3, :, :, 2:] -
                      deformation_field[:, 2:3, :, :, :-2]), (1, 1, 0, 0, 0, 0), mode='replicate') / (2 * h)
        d3u3 = F.pad((deformation_field[:, 2:3, 2:, :, :] -
                      deformation_field[:, 2:3, :-2, :, :]), (0, 0, 0, 0, 1, 1), mode='replicate') / (2 * h)

        r = (d1u1.square() + d2u1.square() + d3u1.square() +
             d1u2.square() + d2u2.square() + d3u2.square() +
             d1u3.square() + d2u3.square() + d3u3.square()).sum(dim=[1, 2, 3, 4])

        return r.mean()


class NCC(nn.Module):
    def __init__(self):
        super(NCC, self).__init__()

    def forward(self, moving, fixed):
        eps = 1e-3
        tmp1 = torch.square(torch.sum(fixed * moving, dim=[1, 2, 3, 4]))
        tmp2 = torch.sum(fixed**2, dim=[1, 2, 3, 4])
        tmp3 = torch.sum(moving**2, dim=[1, 2, 3, 4])
        ncc = (tmp1 + eps) / (tmp2 * tmp3 + eps)

        # Masked D_NCC
        D = 1 - ncc

        return D.mean()


class NCC_masked(nn.Module):
    def __init__(self):
        super(NCC, self).__init__()

    def forward(self, moving, fixed, segm):
        eps = 1e-3
        mask = 1 - segm
        tmp1 = torch.square(torch.sum(mask * fixed * moving, dim=[1, 2, 3, 4]))
        tmp2 = torch.sum(mask * fixed**2, dim=[1, 2, 3, 4])
        tmp3 = torch.sum(mask * moving**2, dim=[1, 2, 3, 4])
        ncc = (tmp1 + eps) / (tmp2 * tmp3 + eps)

        # Masked D_NCC
        D = 1 - ncc

        return D.mean()


class SegmVol(nn.Module):
    def __init__(self):
        super(SegmVol, self).__init__()

    def forward(self, segm):

        H2 = segm.mean(dim=[1, 2, 3, 4])

        return H2.mean()


class SegmPerimeter(nn.Module):
    def __init__(self):
        super(SegmPerimeter, self).__init__()
        self.filter = torch.tensor([[[[[-1, 0, 1],
                                       [-2, 0, 2],
                                       [1, 0, 1]],
                                      [[-2, 0, 2],
                                       [-4, 0, 4],
                                       [-2, 0, 2]],
                                      [[-1, 0, 1],
                                       [-2, 0, 2],
                                       [1, 0, 1]]], # x
                                     [[[-1, -2, -1],
                                       [0, 0, 0],
                                       [1, 2, 1]],
                                      [[-2, -4, 2],
                                       [0, 0, 0],
                                       [2, 4, 2]],
                                      [[-1, -2, -1],
                                       [0, 0, 0],
                                       [1, 2, 1]]], # y
                                     [[[-1, -2, -1],
                                       [-2, -4, -2],
                                       [-1, -2, -1]],
                                      [[0, 0, 0],
                                       [0, 0, 0],
                                       [0, 0, 0]],
                                      [[1, 2, 1],
                                       [2, 4, 2],
                                       [1, 2, 1]]]]]).float().cuda()
        self.tanh = nn.Tanh()
        print('Warning: Implementation for isotropic pixel spacing.')

    def forward(self, segm):

        sobel1 = F.conv3d(segm, self.filter[:, 0:1, :, :, :])
        sobel2 = F.conv3d(segm, self.filter[:, 1:2, :, :, :])
        sobel3 = F.conv3d(segm, self.filter[:, 2:3, :, :, :])
        sobel_abs = torch.square(sobel1) + torch.square(sobel2) + torch.square(sobel3)
        sobel_final = self.tanh(sobel_abs)
        H1 = sobel_final.mean(dim=[1, 2, 3, 4])

        return H1.mean()