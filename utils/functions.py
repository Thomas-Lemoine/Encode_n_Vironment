import os
import cv2
from PIL import Image
import torch
from torchvision.transforms import Compose, ToTensor, Normalize

DATA_PATH = 'data'
IMAGES_PATH = 'static'

def from_filename_to_norm_tensor(file_name):
    # numpy_img = cv2.imread("uploaded_image.png", 0)
    img = Image.open("static/img.png")

    transform_norm = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,))
    ])

    # get normalized image
    img_normalized = transform_norm(img)

    return img_normalized[0, :, :]

def get_latent_repr(file_name, model):
    tensor_img = from_filename_to_norm_tensor(file_name)
    tensor_img = torch.reshape(tensor_img, (1, 1, 28, 28))
    latent, _ = model.encoder(tensor_img)
    latent_np = latent.detach().numpy().squeeze()
    return latent, latent_np

def get_recon_img_np(latent, model):
    recon_img = model.decoder(latent)
    recon_img_np = recon_img.detach().numpy().squeeze()
    return recon_img_np

"""def transform_img_to_numpy(file_name):
    path = os.path.join(IMAGES_PATH, file_name)
    numpy_img = cv2.imread(path, 0)
    return numpy_img

def get_latent_repr(file_name, model):
    numpy_img = transform_img_to_numpy(file_name)
    tensor_img = torch.from_numpy(numpy_img)
    latent, _ = model.encoder(tensor_img)
    latent_np = latent.detatch().numpy().squeeze()
    return latent, latent_np

def get_recon_img(latent, model):
    recon_img = model.decoder(latent)
    recon_img_np = recon_img.detach().numpy().squeeze()

    recon_img = Image.fromnumpy(recon_img_np)
    return recon_img"""

# bring the image (png, jpg, jpeg) through the model and get the reconstructed image, 
# and the latent representation of the image

if __name__ == "__main__":
    # get_latent_repr
    pass