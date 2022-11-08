from get_input_args import get_predict_input_args
from data_preprocessing import gen_data_transforms
from train import create_network, adam_optim, get_fc_params
from torch import optim
import torch
from PIL import Image
from typing import Dict, List
import json


def load_cat_to_name(filename):
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        
    return cat_to_name


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model
    data_transforms = gen_data_transforms()
    
    tensor_processor = data_transforms["test"]
    with Image.open(image) as pil_image:
        img_tensor = tensor_processor(pil_image)
    
    return img_tensor


def load_checkpoint(filepath, return_optim=False):
    checkpoint = torch.load(filepath)
#     model_arch = "vgg16"
#     model_arch = checkpoint["model_arch"] # FIXME
    model = create_network(checkpoint["model_arch"], checkpoint["hidden_units"])
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.class_to_idx = checkpoint["class_to_idx"]
    
    if return_optim:
        optimizer = adam_optim(classifier_params, learning_rate)
        optimizer = optim.Adam(get_fc_params(model, model_arch), lr=checkpoint["learning_rate"])
        optimizer.load_state_dict(checkpoint["optim_state_dict"])
        return model, optimizer
    
    return model


def predict(image_path, model, topk=5, device="cpu"):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    model.eval()
    with torch.no_grad():
        img_tensor = process_image(image_path)
        img_tensor = img_tensor.to(device)
        log_ps = model.forward(img_tensor.unsqueeze(0))
        ps = torch.exp(log_ps)
        
        top_ps, top_idx = ps.topk(topk, dim=1)
        
        # Convert to list
        top_ps = top_ps.tolist()[0]
        top_idx = top_idx.tolist()[0]
        
        # Reverse mapping
        idx_to_class = {model.class_to_idx[x]: x for x in model.class_to_idx}
        top_classes = [idx_to_class[x] for x in top_idx]
        
        return top_ps, top_classes
    

def category_to_name(classes: List[str], cat_to_name: Dict[str, str]):
    """
    Convert a list of class numbers to list of class names
    """
    class_names = [cat_to_name[x] for x in classes]
    
    return class_names


def display_topk(topk_list):
    for idx, value in enumerate(topk_list):
        print(f"-- No.{idx+1}: {value}")


def main():
    in_args = get_predict_input_args()
    
    device = torch.device("cuda:0" if in_args.gpu else "cpu")
    
    if in_args.gpu:
        print("Using GPU to predict...")
        device = torch.device("cuda:0")
    else:
        print("Using CPU to predict...")
        device = torch.device("cpu")
    
    # Load checkpoint
    model = load_checkpoint(in_args.checkpoint_path)
    model.to(device)
    
    # Model prediction
    # img_path = 'flowers/test/10/image_07090.jpg'
    probs, classes = predict(in_args.image_path, model, topk=in_args.top_k, device=device)
    
    # Load mapping json file and convert class category to class name
    cat_to_name = load_cat_to_name(in_args.category_names)
    class_names = category_to_name(classes, cat_to_name)
    
    print("\n")
    print("Image path:", in_args.image_path)
    print(f"+ Top {in_args.top_k} classes prediction:")
    display_topk(classes)
    print(f"+ Top {in_args.top_k} probability:")
    display_topk(probs)
    print(f"+ Top {in_args.top_k} class names prediction:")
    display_topk(class_names)
    

if __name__ == "__main__":
    """
    Test commands:
        + python predict.py flowers/test/10/image_07090.jpg checkpoint.pth
        + python predict.py flowers/test/10/image_07090.jpg checkpoint.pth --gpu
        + python predict.py flowers/test/10/image_07090.jpg checkpoint.pth --category_names cat_to_name.json --topk 5 --gpu
    """
    main()
