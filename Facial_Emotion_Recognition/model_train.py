import torch
import os
from modules.FERDataset import EmotionRecognitionDataset
from utils.config import DEVICE
import torch.nn as nn
from torch.utils.data import DataLoader
from model.FERModel import EmotionRecognitionModel
from mini_XCeption.XCeptionModel import Mini_Xception
from utils.util_funcs import get_train_transform,get_val_transform
import torchvision.transforms as transforms

def main():
    '''
    Function to train the model
    :return: None
    '''
    save_dir = "ERM_Results"
    print(f"------------TRAINING ON DEVICE: {DEVICE}-------------")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    train_dataset = EmotionRecognitionDataset('datasets/training.csv', transforms = get_train_transform())
    val_dataset = EmotionRecognitionDataset('datasets/val.csv',transforms = get_val_transform())
    dataloaders = \
    {
        'train': DataLoader(train_dataset,batch_size = 32,shuffle=True),
        'val': DataLoader(val_dataset,batch_size = 32)
    }

    base_model = Mini_Xception() 
    base_model.to(DEVICE)
    loss_crit = nn.CrossEntropyLoss() 
    lrates = [0.01,0.001]
    best_model = {"model": None, "param": None,
                "epoch": None, "measure": None, "weights": None
                }
    for lr in lrates:
        print(f'##########STARTING NEW RUN Learning Rate: {lr}###########')
        # Clear the GPU cache
        torch.cuda.empty_cache() if DEVICE == 'cuda' else None
        model = EmotionRecognitionModel(model = base_model,device = DEVICE,epochs = 50, criterion = loss_crit,lr=lr)
        best_epoch,best_measure,best_weights = model.fit(dataloaders['train'],dataloaders['val'])
        # print(avg_loss)
        if best_model["measure"] is None or best_measure > best_model["measure"]:
            best_model["model"] = model
            best_model["param"] = lr
            best_model["epoch"] = best_epoch
            best_model["measure"] = best_measure
            best_model["weights"] = best_weights
        # best_model['trg_transform'] = transform_idx
    print("Chosen Model Trained with lr: ", best_model['param'])
    print(f"Chosen Model achieved {best_model['measure']} Accuracy")
    torch.save(best_model["weights"], os.path.join(
        save_dir, "ERModel.pt"))
    with open(os.path.join(save_dir, "FERModel_params.txt"), "w+") as file:
        file.write("parameter,epoch,measure\n")
        file.write(",".join([str(best_model["param"]), str(best_model["epoch"]), str(best_model["measure"])]))
    # save losses
    with open(os.path.join(save_dir, "FERModel_train_losses.txt"), "w+") as file:
        file.write(",".join(map(str, best_model["model"].train_losses)))

    with open(os.path.join(save_dir, "FERModel_valid_losses.txt"), "w+") as file:
        file.write(",".join(map(str, best_model["model"].valid_losses)))

if __name__ == "__main__":
    main()