from torch import empty_like, sigmoid, load
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, Linear, ConvTranspose2d
import torchvision
from torchvision import transforms

DATA_PATH = 'data'
IMAGES_PATH = 'static'
# 2-d latent space, parameter count in same order of magnitude
# as in the original VAE paper (VAE paper has about 3x as many)
latent_dims = 4   # latent_dims = 10 for non-variational auto-encoder
num_epochs = 5
batch_size = 64
capacity = 64
learning_rate = 0.01
variational_beta = 4
use_gpu = True

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        c = capacity
        self.conv1 = Conv2d(in_channels=1, out_channels=c, kernel_size=4, stride=2, padding=1) # out: c x 14 x 14
        self.conv2 = Conv2d(in_channels=c, out_channels=c*2, kernel_size=4, stride=2, padding=1) # out: c x 7 x 7
        self.fc_mu = Linear(in_features=c*2*7*7, out_features=latent_dims)
        self.fc_logvar = Linear(in_features=c*2*7*7, out_features=latent_dims)
            
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1) # flatten batch of multi-channel feature maps to a batch of feature vectors
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        return x_mu, x_logvar

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        c = capacity
        self.fc = Linear(in_features=latent_dims, out_features=c*2*7*7)
        self.conv2 = ConvTranspose2d(in_channels=c*2, out_channels=c, kernel_size=4, stride=2, padding=1)
        self.conv1 = ConvTranspose2d(in_channels=c, out_channels=1, kernel_size=4, stride=2, padding=1)
            
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), capacity*2, 7, 7) # unflatten batch of feature vectors to a batch of multi-channel feature maps
        x = F.relu(self.conv2(x))
        x = sigmoid(self.conv1(x)) # last layer before output is sigmoid, since we are using BCE as reconstruction loss
        return x
    
class VariationalAutoencoder(nn.Module):
    def __init__(self):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar
    
    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

def load_model():
    model = VariationalAutoencoder()

    model.load_state_dict(load('./pretrained/my_vae.pth'))
    
    # print(model.train())
    return model

"""if __name__=='__main__':
    import os, cv2, torch
    from PIL import Image

    model = load_model()
    def from_filename_to_norm_tensor(file_name):
        path = os.path.join(IMAGES_PATH, file_name)
        numpy_img = cv2.imread(path, 0)
        img = Image.open(path)

        transform_norm = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.1307,), (0.3081,))
        ])

        # get normalized image
        img_normalized = transform_norm(img)

        return img_normalized
    
    def get_latent_repr(file_name, model):
        tensor_img = from_filename_to_norm_tensor(file_name)
        print(tensor_img)
        print(type(tensor_img))
        latent, _ = model.encoder(tensor_img)
        latent_np = latent.detatch().numpy().squeeze()
        return latent, latent_np

    def get_recon_img(latent, model):
        recon_img = model.decoder(latent)
        recon_img_np = recon_img.detach().numpy().squeeze()

        recon_img = Image.fromnumpy(recon_img_np)
        return recon_img
    
    latent, latent_np = get_latent_repr("10009.png", model)
    print(latent_np)"""