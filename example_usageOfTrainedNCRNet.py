import imageio
import matplotlib.pyplot as plt
import os
import torch

from master.networks import NoCoRegNet


# Small example for the usage of trained NCR-Net
# Input images are taken from the RETOUCH challenge dataset: https://retouch.grand-challenge.org/
# The used network was trained for intra-patient registration, results might not be very good when it is
# applied on images from different patients

# Contact for NCR-Net:
# Julia Andresen
# j.andresen@uni-luebeck.de
# Institute for Medical Informatics, University of Luebeck


card = 0
torch.cuda.set_device(card)
device = 'cuda:' + str(card)

path1 = 'data/RETOUCH/Spectralis/patient1'
path2 = 'data/RETOUCH/Spectralis/patient2'

net = NoCoRegNet(n_feat=8).cuda()
net.load_state_dict(torch.load('master/ncrNet.pt', map_location=device)['model_state'])
net.eval()

with torch.no_grad():
    for slice in [1, 14, 23, 30, 34]:

        img1 = imageio.imread(os.path.join(path1, 'slice' + str(slice) + '.png'))[..., 0] / 255
        img2 = imageio.imread(os.path.join(path2, 'slice' + str(slice) + '.png'))[..., 0] / 255

        moving_img = torch.tensor(img1).unsqueeze(0).unsqueeze(0)
        fixed_img = torch.tensor(img2).unsqueeze(0).unsqueeze(0)

        input = torch.cat((moving_img, fixed_img), dim=1).cuda().float()

        v, phi, warped, _, _, _, _, _, _, nocomap, _, _ = net(input)

        fig, ax = plt.subplots(2, 3)
        plt.suptitle('Slice ' + str(slice))
        ax[0, 0].imshow(moving_img[0, 0, ...], cmap='gray')
        ax[0, 0].title.set_text('Moving')
        ax[0, 1].imshow(fixed_img[0, 0, ...], cmap='gray')
        ax[0, 1].title.set_text('Reference')
        ax[0, 2].imshow((moving_img - fixed_img)[0, 0, ...], cmap='gray')
        ax[0, 2].title.set_text('Difference before')
        #
        ax[1, 0].imshow(warped[0, 0, ...].cpu(), cmap='gray')
        ax[1, 0].title.set_text('Warped')
        ax[1, 1].imshow((warped.cpu() - fixed_img)[0, 0, ...], cmap='gray')
        ax[1, 1].imshow(nocomap[0, 0, ...].cpu(), alpha=0.6)
        ax[1, 1].title.set_text('NC-Map')
        ax[1, 2].imshow((warped.cpu() - fixed_img)[0, 0, ...], cmap='gray')
        ax[1, 2].title.set_text('Difference after')
        [axi.set_axis_off() for axi in ax.ravel()]
        plt.show()