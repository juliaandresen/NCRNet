import random
import torch.nn as nn
import torch.nn.functional as F
import torch


# Diffeomorphic registration and segmentation of non-corresponding regions network (NCR-Net 2D)
class NoCoRegNet(nn.Module):
    def __init__(self, n_feat, device='cuda'):
        super().__init__()
        self.device = device

        # self.block0 = ...
        self.localBlock0 = nn.Sequential(
            nn.Conv2d(2, n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm2d(n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm2d(n_feat),
            nn.ReLU()
        )

        self.localBlock1 = nn.Sequential(
            nn.Conv2d(n_feat, 2 * n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm2d(2 * n_feat),
            nn.ReLU(),
            nn.Conv2d(2*n_feat, 2*n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm2d(2 * n_feat),
            nn.ReLU()
        )

        self.localBlock2 = nn.Sequential(
            nn.Conv2d(2*n_feat, 4*n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm2d(4 * n_feat),
            nn.ReLU(),
            nn.Conv2d(4*n_feat, 4*n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm2d(4 * n_feat),
            nn.ReLU()
        )

        self.localBlock3 = nn.Sequential(
            nn.Conv2d(4*n_feat, 8*n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm2d(8 * n_feat),
            nn.ReLU(),
            nn.Conv2d(8*n_feat, 8*n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm2d(8 * n_feat),
            nn.ReLU()
        )

        self.localBlock4 = nn.Sequential(
            nn.Conv2d(8 * n_feat, 16 * n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm2d(16 * n_feat),
            nn.ReLU(),
            nn.Conv2d(16 * n_feat, 16 * n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm2d(16 * n_feat),
            nn.ReLU()
        )

        self.localBlock5 = nn.Sequential(
            nn.Conv2d(24 * n_feat, 16 * n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm2d(16 * n_feat),
            nn.ReLU(),
            nn.Conv2d(16 * n_feat, 8 * n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm2d(8 * n_feat),
            nn.ReLU()
        )

        self.localBlock5_2 = nn.Sequential(
            nn.Conv2d(8 * n_feat, 4 * n_feat, 1),
            nn.ReLU(),
            nn.Conv2d(4 * n_feat, 2, 1)
        )

        self.localBlock6 = nn.Sequential(
            nn.Conv2d(12 * n_feat, 8 * n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm2d(8 * n_feat),
            nn.ReLU(),
            nn.Conv2d(8 * n_feat, 6 * n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm2d(6 * n_feat),
            nn.ReLU()
        )

        self.localBlock6_2 = nn.Sequential(
            nn.Conv2d(6 * n_feat, 3 * n_feat, 1),
            nn.ReLU(),
            nn.Conv2d(3 * n_feat, 2, 1)
        )

        self.localBlock7 = nn.Sequential(
            nn.Conv2d(8 * n_feat, 6 * n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm2d(6 * n_feat),
            nn.ReLU(),
            nn.Conv2d(6 * n_feat, 4 * n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm2d(4 * n_feat),
            nn.ReLU(),
            nn.Conv2d(4 * n_feat, 2 * n_feat, 1),
            nn.ReLU(),
            nn.Conv2d(2 * n_feat, 2, 1)
        )

        self.segmBlock1 = nn.Sequential(
            nn.Conv2d(24 * n_feat, 16 * n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm2d(16 * n_feat),
            nn.ReLU(),
            nn.Conv2d(16 * n_feat, 8 * n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm2d(8 * n_feat),
            nn.ReLU()
        )

        self.segmBlock2 = nn.Sequential(
            nn.Conv2d(12 * n_feat, 8 * n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm2d(8 * n_feat),
            nn.ReLU(),
            nn.Conv2d(8 * n_feat, 6 * n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm2d(6 * n_feat),
            nn.ReLU()
        )

        self.segmBlock2_2 = nn.Sequential(
            nn.Conv2d(6 * n_feat, 3 * n_feat, 1),
            nn.ReLU(),
            nn.Conv2d(3 * n_feat, 1, 1),
            nn.Sigmoid()
        )

        self.segmBlock3 = nn.Sequential(
            nn.Conv2d(8 * n_feat, 6 * n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm2d(6 * n_feat),
            nn.ReLU(),
            nn.Conv2d(6 * n_feat, 4 * n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm2d(4 * n_feat),
            nn.ReLU()
        )

        self.segmBlock3_2 = nn.Sequential(
            nn.Conv2d(4 * n_feat, 2 * n_feat, 1),
            nn.ReLU(),
            nn.Conv2d(2 * n_feat, 1, 1),
            nn.Sigmoid()
        )

        self.segmBlock4 = nn.Sequential(
            nn.Conv2d(5 * n_feat, 4 * n_feat, 3, padding=1, bias=False),
            nn.BatchNorm2d(4 * n_feat),
            nn.ReLU(),
            nn.Conv2d(4 * n_feat, 2 * n_feat, 3, padding=1),
            nn.BatchNorm2d(2 * n_feat),
            nn.ReLU(),
            nn.Conv2d(2 * n_feat, n_feat, 1),
            nn.ReLU(),
            nn.Conv2d(n_feat, 1, 1),
            nn.Sigmoid()
        )

        self.maxPool = nn.MaxPool2d(2)
        self.upSampling = nn.Upsample(scale_factor=2, mode='bilinear')


    def localSampler(self, input, velocities):

        _, _, W, H = input.shape
        baseline = input[:, :1, ...]

        nx = W
        ny = H
        if self.device == 'cpu':
            x = torch.linspace(-1, 1, steps=ny).to(dtype=torch.float)
            y = torch.linspace(-1, 1, steps=nx).to(dtype=torch.float)
        else:
            x = torch.linspace(-1, 1, steps=ny).to(dtype=torch.float).cuda()
            y = torch.linspace(-1, 1, steps=nx).to(dtype=torch.float).cuda()
        x = x.expand(nx, -1)
        y = y.expand(ny, -1).transpose(0, 1)
        x.unsqueeze_(0).unsqueeze_(3)
        y.unsqueeze_(0).unsqueeze_(3)
        id_grid = torch.cat((x, y), 3)

        grid = self.diffeomorphic_2D(velocities, id_grid)

        transformed_baseline = F.grid_sample(baseline, id_grid + grid)

        return transformed_baseline, grid


    @staticmethod
    def _compute_scaling_value(displacement):

        with torch.no_grad():
            scaling = 8
            norm = torch.norm(displacement / (2 ** scaling))

            while norm > 0.5:
                scaling += 1
                norm = torch.norm(displacement / (2 ** scaling))

        return scaling

    @staticmethod
    def diffeomorphic_2D(displacement, grid, scaling=-1):

        N, _, _, _ = displacement.shape

        for n in range(N):
            d = displacement[n, ...].transpose(0, 1).transpose(1, 2)

            if scaling < 0:
                scaling = self._compute_scaling_value(d)

            d = d / (2 ** scaling)

            d = d.transpose(2, 1).transpose(1, 0).unsqueeze(0)

            for i in range(scaling):
                d_trans = d.transpose(1, 2).transpose(2, 3)
                d = d + F.grid_sample(d, d_trans + grid)

            displacement[n, ...] = d

        return displacement.transpose(1, 2).transpose(2, 3)


    def forward(self, inputs):

        output0 = self.localBlock0(inputs)
        output0_maxPooled = self.maxPool(output0)

        output1 = self.localBlock1(output0_maxPooled)
        output1_maxPooled = self.maxPool(output1)

        output2 = self.localBlock2(output1_maxPooled)
        output2_maxPooled = self.maxPool(output2)

        output3 = self.localBlock3(output2_maxPooled)
        output3_maxPooled = self.maxPool(output3)

        output4 = self.localBlock4(output3_maxPooled)
        output4_upscaled = self.upSampling(output4)

        # Branch 1
        output5 = self.localBlock5(torch.cat([output3, output4_upscaled], dim=1))
        output5_2 = self.localBlock5_2(output5)
        output5_upscaled = self.upSampling(output5)

        output6 = self.localBlock6(torch.cat([output2, output5_upscaled], dim=1))
        output6_2 = self.localBlock6_2(output6)
        output6_upscaled = self.upSampling(output6)

        output7 = self.localBlock7(torch.cat([output1, output6_upscaled], dim=1))

        # Implicite regularization as described in "Schwach ueberwachtes Lernen nichtlinearer medizinischer
        # Bildregistrierung mit neuronalen Faltungsnetzwerken" (S. Kuckertz, 2018): Deformation field u is calculated
        # on coarse grid and interpolated -> smoothing of deformation field
        v1 = F.interpolate(output7, scale_factor=2, mode='bilinear')
        v2 = F.interpolate(output6_2, scale_factor=2, mode='bilinear')
        v3 = F.interpolate(output5_2, scale_factor=2, mode='bilinear')

        deformed1, phi1 = self.localSampler(inputs, v1)
        deformed2, phi2 = self.localSampler(inputs[:, :, ::2, ::2], v2)
        deformed3, phi3 = self.localSampler(inputs[:, :, ::4, ::4], v3)

        # Branch 2
        segm5 = self.segmBlock1(torch.cat([output3, output4_upscaled], dim=1))
        segm5_upscaled = self.upSampling(segm5)

        segm6 = self.segmBlock2(torch.cat([output2, segm5_upscaled], dim=1))
        segm6_upscaled = self.upSampling(segm6)
        segmentation3 = self.segmBlock2_2(segm6)

        segm7 = self.segmBlock3(torch.cat([output1, segm6_upscaled], dim=1))
        segm7_upscaled = self.upSampling(segm7)
        segmentation2 = self.segmBlock3_2(segm7)

        segmentation = self.segmBlock4(torch.cat([output0, segm7_upscaled], dim=1))

        return v1, phi1, deformed1, v2, phi2, deformed2, v3, phi3, deformed3, segmentation, segmentation2, segmentation3


# Diffeomorphic registration network
class RegNet(nn.Module):
    def __init__(self, n_feat, device='cuda'):
        super().__init__()
        self.device = device

        # self.block0 = ...
        self.localBlock0 = nn.Sequential(
            nn.Conv2d(2, n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm2d(n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm2d(n_feat),
            nn.ReLU()
        )

        self.localBlock1 = nn.Sequential(
            nn.Conv2d(n_feat, 2 * n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm2d(2 * n_feat),
            nn.ReLU(),
            nn.Conv2d(2*n_feat, 2*n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm2d(2 * n_feat),
            nn.ReLU()
        )

        self.localBlock2 = nn.Sequential(
            nn.Conv2d(2*n_feat, 4*n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm2d(4 * n_feat),
            nn.ReLU(),
            nn.Conv2d(4*n_feat, 4*n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm2d(4 * n_feat),
            nn.ReLU()
        )

        self.localBlock3 = nn.Sequential(
            nn.Conv2d(4*n_feat, 8*n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm2d(8 * n_feat),
            nn.ReLU(),
            nn.Conv2d(8*n_feat, 8*n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm2d(8 * n_feat),
            nn.ReLU()
        )

        self.localBlock4 = nn.Sequential(
            nn.Conv2d(8 * n_feat, 16 * n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm2d(16 * n_feat),
            nn.ReLU(),
            nn.Conv2d(16 * n_feat, 16 * n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm2d(16 * n_feat),
            nn.ReLU()
        )

        self.localBlock5 = nn.Sequential(
            nn.Conv2d(24 * n_feat, 16 * n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm2d(16 * n_feat),
            nn.ReLU(),
            nn.Conv2d(16 * n_feat, 8 * n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm2d(8 * n_feat),
            nn.ReLU()
        )

        self.localBlock5_2 = nn.Sequential(
            nn.Conv2d(8 * n_feat, 4 * n_feat, 1),
            nn.ReLU(),
            nn.Conv2d(4 * n_feat, 2, 1)
        )

        self.localBlock6 = nn.Sequential(
            nn.Conv2d(12 * n_feat, 8 * n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm2d(8 * n_feat),
            nn.ReLU(),
            nn.Conv2d(8 * n_feat, 6 * n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm2d(6 * n_feat),
            nn.ReLU()
        )

        self.localBlock6_2 = nn.Sequential(
            nn.Conv2d(6 * n_feat, 3 * n_feat, 1),
            nn.ReLU(),
            nn.Conv2d(3 * n_feat, 2, 1)
        )

        self.localBlock7 = nn.Sequential(
            nn.Conv2d(8 * n_feat, 6 * n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm2d(6 * n_feat),
            nn.ReLU(),
            nn.Conv2d(6 * n_feat, 4 * n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm2d(4 * n_feat),
            nn.ReLU(),
            nn.Conv2d(4 * n_feat, 2 * n_feat, 1),
            nn.ReLU(),
            nn.Conv2d(2 * n_feat, 2, 1)
        )

        self.maxPool = nn.MaxPool2d(2)
        self.upSampling = nn.UpsamplingNearest2d(scale_factor=2)


    def localSampler(self, input, velocities):

        _, _, W, H = input.shape
        baseline = input[:, :1, ...]

        nx = W
        ny = H
        if self.device == 'cpu':
            x = torch.linspace(-1, 1, steps=ny).to(dtype=torch.float)
            y = torch.linspace(-1, 1, steps=nx).to(dtype=torch.float)
        else:
            x = torch.linspace(-1, 1, steps=ny).to(dtype=torch.float).cuda()
            y = torch.linspace(-1, 1, steps=nx).to(dtype=torch.float).cuda()
        x = x.expand(nx, -1)
        y = y.expand(ny, -1).transpose(0, 1)
        x.unsqueeze_(0).unsqueeze_(3)
        y.unsqueeze_(0).unsqueeze_(3)
        id_grid = torch.cat((x, y), 3)

        grid = self.diffeomorphic_2D(velocities, id_grid)

        transformed_baseline = F.grid_sample(baseline, id_grid + grid)

        return transformed_baseline, grid


    @staticmethod
    def _compute_scaling_value(displacement):

        with torch.no_grad():
            scaling = 8
            norm = torch.norm(displacement / (2 ** scaling))

            while norm > 0.5:
                scaling += 1
                norm = torch.norm(displacement / (2 ** scaling))

        return scaling

    @staticmethod
    def diffeomorphic_2D(displacement, grid, scaling=-1):

        N, _, _, _ = displacement.shape

        for n in range(N):
            d = displacement[n, ...].transpose(0, 1).transpose(1, 2)

            if scaling < 0:
                scaling = mumfordShahVelocity._compute_scaling_value(d)

            d = d / (2 ** scaling)

            d = d.transpose(2, 1).transpose(1, 0).unsqueeze(0)

            for i in range(scaling):
                d_trans = d.transpose(1, 2).transpose(2, 3)
                d = d + F.grid_sample(d, d_trans + grid)

            displacement[n, ...] = d

        return displacement.transpose(1, 2).transpose(2, 3)


    def forward(self, inputs):

        output0 = self.localBlock0(inputs)
        output0_maxPooled = self.maxPool(output0)

        output1 = self.localBlock1(output0_maxPooled)
        output1_maxPooled = self.maxPool(output1)

        output2 = self.localBlock2(output1_maxPooled)
        output2_maxPooled = self.maxPool(output2)

        output3 = self.localBlock3(output2_maxPooled)
        output3_maxPooled = self.maxPool(output3)

        output4 = self.localBlock4(output3_maxPooled)
        output4_upscaled = self.upSampling(output4)

        # Branch 1
        output5 = self.localBlock5(torch.cat([output3, output4_upscaled], dim=1))
        output5_2 = self.localBlock5_2(output5)
        output5_upscaled = self.upSampling(output5)

        output6 = self.localBlock6(torch.cat([output2, output5_upscaled], dim=1))
        output6_2 = self.localBlock6_2(output6)
        output6_upscaled = self.upSampling(output6)

        output7 = self.localBlock7(torch.cat([output1, output6_upscaled], dim=1))

        # Implicite regularization as described in "Schwach ueberwachtes Lernen nichtlinearer medizinischer
        # Bildregistrierung mit neuronalen Faltungsnetzwerken" (S. Kuckertz, 2018): Deformation field u is calculated
        # on coarse grid and interpolated -> smoothing of deformation field
        v1 = F.interpolate(output7, scale_factor=2, mode='bilinear')
        v2 = F.interpolate(output6_2, scale_factor=2, mode='bilinear')
        v3 = F.interpolate(output5_2, scale_factor=2, mode='bilinear')

        deformed1, phi1 = self.localSampler(inputs, v1)
        deformed2, phi2 = self.localSampler(inputs[:, :, ::2, ::2], v2)
        deformed3, phi3 = self.localSampler(inputs[:, :, ::4, ::4], v3)

        return v1, phi1, deformed1, v2, phi2, deformed2, v3, phi3, deformed3


#
#
#

# 3D NCR-Net

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, device, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        # self.register_buffer('grid', grid)
        self.grid = grid.to(device)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps, device):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape, device)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class NoCoRegNet3D(nn.Module):
    def __init__(self, n_feat, inshape, device, int_downsize=1, int_steps=7):
        super().__init__()

        # self.block0 = ...
        self.localBlock0 = nn.Sequential(
            nn.Conv3d(2, n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm3d(n_feat),
            nn.ReLU(),
            nn.Conv3d(n_feat, n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm3d(n_feat),
            nn.ReLU()
        )

        self.localBlock1 = nn.Sequential(
            nn.Conv3d(n_feat, 2 * n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm3d(2 * n_feat),
            nn.ReLU(),
            nn.Conv3d(2*n_feat, 2*n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm3d(2 * n_feat),
            nn.ReLU()
        )

        self.localBlock2 = nn.Sequential(
            nn.Conv3d(2*n_feat, 4*n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm3d(4 * n_feat),
            nn.ReLU(),
            nn.Conv3d(4*n_feat, 4*n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm3d(4 * n_feat),
            nn.ReLU()
        )

        self.localBlock3 = nn.Sequential(
            nn.Conv3d(4*n_feat, 8*n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm3d(8 * n_feat),
            nn.ReLU(),
            nn.Conv3d(8*n_feat, 8*n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm3d(8 * n_feat),
            nn.ReLU()
        )

        self.localBlock4 = nn.Sequential(
            nn.Conv3d(8 * n_feat, 16 * n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm3d(16 * n_feat),
            nn.ReLU(),
            nn.Conv3d(16 * n_feat, 16 * n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm3d(16 * n_feat),
            nn.ReLU()
        )

        self.localBlock5 = nn.Sequential(
            nn.Conv3d(24 * n_feat, 16 * n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm3d(16 * n_feat),
            nn.ReLU(),
            nn.Conv3d(16 * n_feat, 8 * n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm3d(8 * n_feat),
            nn.ReLU()
        )

        self.localBlock5_2 = nn.Sequential(
            nn.Conv3d(8 * n_feat, 4 * n_feat, 1),
            nn.ReLU(),
            nn.Conv3d(4 * n_feat, 3, 1)
        )

        self.localBlock6 = nn.Sequential(
            nn.Conv3d(12 * n_feat, 8 * n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm3d(8 * n_feat),
            nn.ReLU(),
            nn.Conv3d(8 * n_feat, 6 * n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm3d(6 * n_feat),
            nn.ReLU()
        )

        self.localBlock6_2 = nn.Sequential(
            nn.Conv3d(6 * n_feat, 3 * n_feat, 1),
            nn.ReLU(),
            nn.Conv3d(3 * n_feat, 3, 1)
        )

        self.localBlock7 = nn.Sequential(
            nn.Conv3d(8 * n_feat, 6 * n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm3d(6 * n_feat),
            nn.ReLU(),
            nn.Conv3d(6 * n_feat, 4 * n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm3d(4 * n_feat),
            nn.ReLU(),
            nn.Conv3d(4 * n_feat, 2 * n_feat, 1),
            nn.ReLU(),
            nn.Conv3d(2 * n_feat, 3, 1)
        )

        self.segmBlock1 = nn.Sequential(
            nn.Conv3d(24 * n_feat, 16 * n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm3d(16 * n_feat),
            nn.ReLU(),
            nn.Conv3d(16 * n_feat, 8 * n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm3d(8 * n_feat),
            nn.ReLU()
        )

        self.segmBlock2 = nn.Sequential(
            nn.Conv3d(12 * n_feat, 8 * n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm3d(8 * n_feat),
            nn.ReLU(),
            nn.Conv3d(8 * n_feat, 6 * n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm3d(6 * n_feat),
            nn.ReLU()
        )

        self.segmBlock2_2 = nn.Sequential(
            nn.Conv3d(6 * n_feat, 3 * n_feat, 1),
            nn.ReLU(),
            nn.Conv3d(3 * n_feat, 1, 1),
            nn.Sigmoid()
        )

        self.segmBlock3 = nn.Sequential(
            nn.Conv3d(8 * n_feat, 6 * n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm3d(6 * n_feat),
            nn.ReLU(),
            nn.Conv3d(6 * n_feat, 4 * n_feat, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm3d(4 * n_feat),
            nn.ReLU()
        )

        self.segmBlock3_2 = nn.Sequential(
            nn.Conv3d(4 * n_feat, 2 * n_feat, 1),
            nn.ReLU(),
            nn.Conv3d(2 * n_feat, 1, 1),
            nn.Sigmoid()
        )

        self.segmBlock4 = nn.Sequential(
            nn.Conv3d(5 * n_feat, 4 * n_feat, 3, padding=1, bias=False),
            nn.BatchNorm3d(4 * n_feat),
            nn.ReLU(),
            nn.Conv3d(4 * n_feat, 2 * n_feat, 3, padding=1),
            nn.BatchNorm3d(2 * n_feat),
            nn.ReLU(),
            nn.Conv3d(2 * n_feat, n_feat, 1),
            nn.ReLU(),
            nn.Conv3d(n_feat, 1, 1),
            nn.Sigmoid()
        )

        self.maxPool = nn.MaxPool3d(2)
        self.upSampling = nn.Upsample(scale_factor=2, mode='trilinear')

        down_shape1 = (inshape[0]//2, inshape[1]//2, inshape[2]//2)
        down_shape2 = (inshape[0]//4, inshape[1]//4, inshape[2]//4)
        self.integrate1 = VecInt(inshape, int_steps, device) if int_steps > 0 else None
        self.integrate2 = VecInt(down_shape1, int_steps, device) if int_steps > 0 else None
        self.integrate3 = VecInt(down_shape2, int_steps, device) if int_steps > 0 else None

        self.transformer1 = SpatialTransformer(inshape, device)
        self.transformer2 = SpatialTransformer(down_shape1, device)
        self.transformer3 = SpatialTransformer(down_shape2, device)


    def forward(self, inputs):

        output0 = self.localBlock0(inputs)
        output0_maxPooled = self.maxPool(output0)

        output1 = self.localBlock1(output0_maxPooled)
        output1_maxPooled = self.maxPool(output1)

        output2 = self.localBlock2(output1_maxPooled)
        output2_maxPooled = self.maxPool(output2)

        output3 = self.localBlock3(output2_maxPooled)
        output3_maxPooled = self.maxPool(output3)

        output4 = self.localBlock4(output3_maxPooled)
        output4_upscaled = self.upSampling(output4)

        # Branch 1
        output5 = self.localBlock5(torch.cat([output3, output4_upscaled], dim=1))
        output5_2 = self.localBlock5_2(output5)
        output5_upscaled = self.upSampling(output5)

        output6 = self.localBlock6(torch.cat([output2, output5_upscaled], dim=1))
        output6_2 = self.localBlock6_2(output6)
        output6_upscaled = self.upSampling(output6)

        output7 = self.localBlock7(torch.cat([output1, output6_upscaled], dim=1))

        # Implicite regularization as described in "Schwach ueberwachtes Lernen nichtlinearer medizinischer
        # Bildregistrierung mit neuronalen Faltungsnetzwerken" (S. Kuckertz, 2018): Deformation field u is calculated
        # on coarse grid and interpolated -> smoothing of deformation field
        v1 = F.interpolate(output7, scale_factor=2, mode='trilinear')
        v2 = F.interpolate(output6_2, scale_factor=2, mode='trilinear')
        v3 = F.interpolate(output5_2, scale_factor=2, mode='trilinear')

        phi1 = self.integrate1(v1)
        phi2 = self.integrate2(v2)
        phi3 = self.integrate3(v3)

        deformed1 = self.transformer1(inputs[:, 0:1, :, :, :], phi1)
        deformed2 = self.transformer2(inputs[:, 0:1, ::2, ::2, ::2], phi2)
        deformed3 = self.transformer3(inputs[:, 0:1, ::4, ::4, ::4], phi3)

        # Branch 2
        segm5 = self.segmBlock1(torch.cat([output3, output4_upscaled], dim=1))
        segm5_upscaled = self.upSampling(segm5)

        segm6 = self.segmBlock2(torch.cat([output2, segm5_upscaled], dim=1))
        segm6_upscaled = self.upSampling(segm6)
        segmentation3 = self.segmBlock2_2(segm6)

        segm7 = self.segmBlock3(torch.cat([output1, segm6_upscaled], dim=1))
        segm7_upscaled = self.upSampling(segm7)
        segmentation2 = self.segmBlock3_2(segm7)

        segmentation = self.segmBlock4(torch.cat([output0, segm7_upscaled], dim=1))

        return v1, phi1, deformed1, v2, phi2, deformed2, v3, phi3, deformed3, segmentation, segmentation2, segmentation3