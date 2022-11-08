import argparse


def get_train_input_args():
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    parser.add_argument("data_dir", type=str, help="data directory")
    parser.add_argument("--save_dir", type=str, default="checkpoint.pth", help="directory to save checkpoint")
    parser.add_argument("--arch", type=str, default="vgg16", help="choose architecture")
    parser.add_argument("--learning_rate", type=float, default=0.0025, help="learning rate to train")
    parser.add_argument("--hidden_units", type=int, default=512, help="number of hidden units in neural network")
    parser.add_argument("--epochs", type=int, default=5, help="number of epochs to train")
    parser.add_argument("--gpu", action="store_true")
    
    return parser.parse_args()


def get_predict_input_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("image_path", type=str, help="path to image")
    parser.add_argument("checkpoint_path", type=str, help="path to checkpoint")
    parser.add_argument("--top_k", type=int, default=5, help="return k classes having highest probability")
    parser.add_argument("--category_names", type=str, default="cat_to_name.json", help="class category to name filepath")
    parser.add_argument("--gpu", action="store_true")
    
    return parser.parse_args()
