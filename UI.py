import gradio as gr
from model import Generator
from model import Discriminator
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
from PIL import Image

def restore_model():
    """Restore the trained generator and discriminator."""
    print('Loading the trained models from step {}...')
    G_path ='200000-G.ckpt'
    D_path = '200000-D.ckpt'
    """Create a generator and a discriminator."""

    G = Generator(64,8,6)
    D = Discriminator(256, 64, 8, 6) 
    G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
    D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
    
            
def create_labels(c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
    """Generate target domain labels for debugging and testing."""
    # Get hair color indices.
    if dataset == 'CelebA':
        hair_color_indices = []
        for i, attr_name in enumerate(selected_attrs):
            if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                hair_color_indices.append(i)

    c_trg_list = []
    for i in range(c_dim):
        if dataset == 'CelebA':
            c_trg = c_org.clone()
            if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                c_trg[:, i] = 1
                for j in hair_color_indices:
                    if j != i:
                        c_trg[:, j] = 0
            else:
                c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.
        elif dataset == 'RaFD':
            c_trg = label2onehot(torch.ones(c_org.size(0))*i, c_dim)

        c_trg_list.append(c_trg.to( torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    return c_trg_list
     

def label2onehot( labels, dim):
    """Convert label indices to one-hot vectors."""
    
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), labels.long()] = 1
    return out

def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)
        

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)

def image_classifier(img, label =0):
    classes = {'neutral':0,'happy':1,'sad':2,'surprise':3,'fear':4,'disgust':5,'anger':6,'contempt':7}
    """Translate images using StarGAN trained on a single dataset."""
    # Load the trained generator.

    im = Image.fromarray(img)
    im.save("UI/class001/testy.jpeg")

    """Restore the trained generator and discriminator."""
    print('Loading the trained models from step {}...')
    G_path ='200000-G.ckpt'
    D_path = '200000-D.ckpt'
    """Create a generator and a discriminator."""

    G = Generator(64,8,6)
    D = Discriminator(256, 64, 8, 6) 
    G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
    D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
    data_loader = get_loader('UI/', None, None,256, 256, 64, 'RaFD', 'test', 1)
    print(data_loader)

    with torch.no_grad():
        for i, (x_real, c_org) in enumerate(data_loader):
    
            # Prepare input images and target domain labels.
            x_real = x_real.to( torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            c_trg_list = create_labels(c_org, 8, 'RaFD', None)

            if label == "neutral": 
                a = [[1.,0.,0.,0.,0.,0.,0.,0.,]]
            elif label == "happy": 
                a = [[0.,1.,0.,0.,0.,0.,0.,0.,]]
            elif label == "sad":
                a = [[0.,0.,1.,0.,0.,0.,0.,0.,]]
            elif label == "surprise":
                a = [[0.,0.,0.,1.,0.,0.,0.,0.,]]
            elif label == "fear":
                a = [[0.,0.,0.,0.,1.,0.,0.,0.,]]
            elif label == "disgust":
                a = [[0.,0.,0.,0.,0.,1.,0.,0.,]]
            elif label == "anger":
                a = [[0.,0.,0.,0.,0.,0.,1.,0.,]]
            elif label == "contempt":
                a = [[0.,0.,0.,0.,0.,0.,0.,1.,]]
                
                
            b = torch.FloatTensor(a) 
            img1 = G(x_real, b)
            # img_2 = tensor_to_image(x_fake_list)
            result_path = os.path.join('UI/', 'output-images.jpg')
            print(result_path)
            save_image(denorm(img1.data.cpu()), result_path, nrow=1, padding=0)
            img = Image.open("UI/output-images.jpg") 
            result = img.resize((256,256))
    
    return result

def class_predicter(img):
     classes = {'neutral':0,'happy':1,'sad':2,'surprise':3,'fear':4,'disgust':5,'anger':6,'contempt':7}
     """Translate images using StarGAN trained on a single dataset."""
     # Load the trained generator.
    
     im = Image.fromarray(img)
     im.save("UI/class001/testy.jpeg")
    
     """Restore the trained generator and discriminator."""
     print('Loading the trained models from step {}...')
     G_path ='200000-G.ckpt'
     D_path = '200000-D.ckpt'
     """Create a generator and a discriminator."""
    
     G = Generator(64,8,6)
     D = Discriminator(256, 64, 8, 6) 
     G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
     D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
     data_loader = get_loader('UI/', None, None,256, 256, 64, 'RaFD', 'test', 1)
     print(data_loader)
    
     with torch.no_grad():
         for i, (x_real, c_org) in enumerate(data_loader):
     
             # Prepare input images and target domain labels.
             x_real = x_real.to( torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            
             img2 = D(x_real)
             classy = (img2[1].tolist())[0]
             mini = classy[0]
             print(classy)
             for i in classy:
                 if i > mini:
                     mini=i
             predicted_class = (classy.index(mini))
             
             pred_class = list(classes.keys())[list(classes.values()).index(predicted_class)]
             print(pred_class)
     
     return pred_class


def mirror(x):
    return "hello"

with gr.Blocks(css=".gradio-container {width: 30}") as demo:   
    with gr.Row():
        with gr.Column(scale=1, min_width=700):
            im = gr.Image(label = "Input Image")
            drp = gr.Dropdown(choices = ["anger","fear","surprise","neutral","sad","happy","disgust","contempt"],value="neutral", label = "Emotion")
            btny = gr.Button(value="Emotion Detection")
            btn = gr.Button(value="Emotion Modification")

        with gr.Row(scale=2, min_width=300):
            im_2 = gr.Image(label = "Output Image")
            txt = gr.Textbox(value="", label="Predicted Class")

    btn.click(image_classifier, inputs=[im,drp], outputs=[im_2])
    btny.click(class_predicter, inputs=[im], outputs=[txt])
    
    
# iface = gr.Interface(image_classifier, [gr.Image(),gr.inputs.Dropdown(['neutral','happy','sad','surprise','fear','disgust','anger','contempt'], type="index")], [gr.Image()])

# iface.launch()

if __name__ == "__main__":
    demo.launch()


# with gr.Blocks() as demo:

#     im = gr.Image()
#     drp = gr.Dropdown(["anger","fear","surprise","neutral","sad","happy","disgust"])
#     btn = gr.Button(value="Emotion Modification")
#     btn.click(image_classifier, inputs=[im,drp], outputs=[im])
    
#     with gr.Row():
#         im = gr.Image()
#         txt = gr.Textbox(value="", label="Output")
        
#     btn = gr.Button(value="Emotion Recognition")
#     btn.click(mirror, inputs=[im], outputs=[txt])
