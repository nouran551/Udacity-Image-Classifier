import argparse

import json
import PIL
import torch
import numpy as np
from math import ceil
from torchvision import models

def arg_parser():
    parser = argparse.ArgumentParser(description="predict.py")
    parser.add_argument('--image',type=str, required=True)
    parser.add_argument('--checkpoint', help='Checkpoint file', type=str, required=True, default = 'checkpoint.pth')
    parser.add_argument('--top_k', help='Top K matches', type=int, default = 3)
    parser.add_argument('--category_names', dest="category_names", action="store", default = 'cat_to_name.json')
    parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")
    
    args = parser.parse_args()
    
    return args


def load_checkpoint(path):
    checkpoint = torch.load("checkpoint.pth")
    model = models.vgg19(pretrained=True)
    for param in model.parameters(): 
        param.requires_grad = False

    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def process_image(image):
   img = Image.open(image)
   process = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
   img_tensor = process(img)

   return img_tensor

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    model.to(device = 'cpu')
    processed_img = process_image(image_path)
    
    processed_img.unsqueeze_(0) # add 1 as the first argument as tensor needs batch size
    with torch.no_grad():
        output = model.forward(processed_img) # forward pass
        
    probs = torch.exp(output) 
    top_probs, top_idxs = probs.topk(topk) 
    #print('top_probs',top_probs)
    #print('top_idxs', top_idxs)
    idx_to_class = {value:key for key, value in model.class_to_idx.items()} # to invert class_to_idx
    np_top_idxs = top_idxs[0].numpy()
    
    top_class = []
    for i in np_top_idxs:
        top_class.append(int(idx_to_class[i]))

    top_flowers = [cat_to_name[str(i)] for i in top_class]
    
    return top_probs.numpy().tolist()[0], np_top_idxs.tolist(), top_flowers 
    
def check_sanity(image_path):
    top_probs, top_idxs, top_flowers = predict(image_path, model) 
    for x, y in zip(top_probs, top_flowers):
        print(f"Flower: {y} has probability {x}") 
def main():
    args = arg_parser()

    with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)

    model = load_checkpoint(args.checkpoint)

    image_tensor = process_image(args.image)

    device = check_gpu(gpu_arg=args.gpu);

    top_probs, top_labels, top_flowers = predict(args.image, model,args.top_k) # try "flowers/test/28/image_05230.jpg"
if __name__ == '__main__':
    main()
