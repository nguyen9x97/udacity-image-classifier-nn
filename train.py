from get_input_args import get_train_input_args
from data_preprocessing import get_train_val_test_dirs, gen_data_transforms, load_image_datasets, gen_dataloaders

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import models


accept_models = {"vgg16", "vgg13", "resnet50"}

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_fc_input_size(model, model_arch):
    if model_arch in ["vgg16", "vgg13"]:
        fc_input_size = model.classifier[0].in_features
    elif model_arch in ["resnet50"]:
        fc_input_size = model.fc.in_features
    
    return fc_input_size


def get_fc_attribute_name(model_arch):
    """
    If model architecture is belong to VGG13, VGG16,...
        + Fully connected layer should be get by: model.classifier
    If model architecture is belong to Resnet50,...
        + Fully connected layer should be get by: model.fc
    """
    if model_arch in ["vgg16", "vgg13"]:
        return "classifier"
    elif model_arch in ["resnet50"]:
        return "fc"
    
 
def get_fc_params(model, model_arch):
    fc_attr_name = get_fc_attribute_name(model_arch)
    
    if fc_attr_name in ["classifier"]:
        fc_params = model.classifier.parameters()
    elif fc_attr_name in ["fc"]:
        fc_params = model.fc.parameters()
    
    return fc_params


def create_network(model_arch, n_hidden_units, output_size=102):
    if model_arch not in accept_models:
        raise NotImplementedError(f"The application does not support `{model_arch}` architecture, the acceptable architectures are: {accept_models}")
        
    model = models.__getattribute__(model_arch)(pretrained=True)
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
        
    fc_input_size = get_fc_input_size(model, model_arch)
        
    classifier = nn.Sequential(nn.Linear(fc_input_size, n_hidden_units),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(n_hidden_units, 256),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(256, output_size),
                                nn.LogSoftmax(dim=1))

    fc_attr_name = get_fc_attribute_name(model_arch)
    
    if fc_attr_name in ["classifier"]:
        model.classifier = classifier
    elif fc_attr_name in ["fc"]:
        model.fc = classifier
    
    return model


def adam_optim(classifier_params, learning_rate):
    optimizer = optim.Adam(classifier_params, lr=learning_rate)
    return optimizer


def train_model(dataloaders, model, criterion, optimizer, epochs, device="cpu"):
    trainloader = dataloaders['train']
    validloader = dataloaders['val']
    
    separator_line = "#"*20
    print(separator_line, "TRAINING MODEL", separator_line)
    
    print_every = 10
    steps = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        print('-' * 10)

        running_loss = 0
        for inputs, labels in trainloader:
            steps += 1

            model.train()
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward
            log_ps = model.forward(inputs)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                val_loss, accuracy = eval_model_accuracy(model, validloader, criterion, device)

                print(f".. Train loss: {running_loss/print_every:.3f}.. "
                     f"Validation loss: {val_loss:.3f}.. "
                     f"Validation accuracy: {accuracy:.3f}")
                running_loss = 0
                
    print(separator_line, "TRAINING COMPLETED", separator_line)


def eval_model_accuracy(model, dataloader, criterion, device="cpu"):
    """
    Evaluate model
    PARAMS:
        + model: model to evaluate
        + dataloader: can be validation loader or test loader
        + criterion: loss function such as nn.NLLLoss()
    RETURNS:
        + total_loss
        + accuracy
    """
    model.eval()
    total_loss = 0
    accuracy = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            log_ps = model.forward(inputs)
            loss = criterion(log_ps, labels)
            total_loss += loss

            ps = torch.exp(log_ps)
            top_k, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    
    total_loss = total_loss / len(dataloader)
    accuracy = accuracy / len(dataloader)
    
    return total_loss, accuracy


def main():
    in_args = get_train_input_args()
    
    if in_args.gpu:
        print("Using GPU to train...")
        device = torch.device("cuda:0")
    else:
        print("Using CPU to train...")
        device = torch.device("cpu")
    
    # Load and preprocess data
    train_dir, valid_dir, test_dir = get_train_val_test_dirs(in_args.data_dir)
    data_transforms = gen_data_transforms()
    image_datasets = load_image_datasets([train_dir, valid_dir, test_dir], data_transforms)
    dataloaders = gen_dataloaders(image_datasets)
    
    # Create model architecture
    model_arch = in_args.arch
    model = create_network(model_arch, in_args.hidden_units)
    model.to(device)
    
    # Criterion
    criterion = nn.NLLLoss()
    
    # Optimizer
    optimizer = adam_optim(get_fc_params(model, model_arch), in_args.learning_rate)
    
    # Training model
    train_model(dataloaders, model, criterion, optimizer, in_args.epochs, device)
    
    # Save checkpoint
    checkpoint = {
        "epochs": in_args.epochs,
        "learing_rate": in_args.learning_rate,
        "hidden_units": in_args.hidden_units,
        "model_arch": in_args.arch,
        "model_state_dict": model.state_dict(),
        "class_to_idx": image_datasets['train'].class_to_idx,
        "optim_state_dict": optimizer.state_dict()
    }
    torch.save(checkpoint, in_args.save_dir)
    print(f"Saved checkpoint to `{in_args.save_dir}`")
    
    
if __name__ == "__main__":
    """
    Test commands:
        + python train.py flowers
        + python train.py flowers --gpu
        + python train.py flower --arch vgg16 --learning_rate 0.0025 --hidden_units 512 --epochs 5 --gpu
    """
    main()
