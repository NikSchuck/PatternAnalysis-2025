from modules import ImprovedUNet
from dataset import build_loaders
from train import main as train_main
from predict import main as predict_main

if __name__ == "__main__":
    # import from working train.py
    train_main()

    #then run predict.py
    predict_main()