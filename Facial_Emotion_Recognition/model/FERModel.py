import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix

class EmotionRecognitionModel():
    def __init__(self, model, device, n_classes=3, weights=None, criterion=None, lr=None, epochs=None):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.epochs = epochs
        self.train_losses = []
        self.valid_losses = []
        self.config_model(weights)
        if(lr != None):
            self.optimizer = torch.optim.Adam(self.model.parameters(),lr=float(lr))
           
        else:
            self.optimizer = None
    def config_model(self,weights):
        if weights is not None:
            self.model.load_state_dict(torch.load(weights))
        self.model.to(self.device)
    def train(self, dataloader):
        #initialize train mode
        self.model.train()
        epoch_losses = []
        avg_loss = 0
        num_of_batches = 0
        for images,labels in tqdm(dataloader):
            images,labels = images.to(self.device),labels.to(self.device)
            pred = self.model(images)
            # Convert shape from 7,1,1 to batch,7
            pred = torch.squeeze(pred)
            loss = self.criterion(pred,labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            avg_loss = (avg_loss * num_of_batches + loss) / \
                       (num_of_batches + 1)
            epoch_losses.append(float(avg_loss))

            # update number of batches
            num_of_batches += 1
    
        return avg_loss,epoch_losses
    def evaluate(self, dataloader):
        self.model.eval()   
        with torch.no_grad():
            num_of_batches = 0
            data_size = 0
            avg_loss = 0
            all_preds = []
            all_labels = []
            for images,labels in tqdm(dataloader):
                images,labels = images.to(self.device),labels.to(self.device)
                preds = self.model(images)
                preds = torch.squeeze(preds)
                preds = preds.reshape(images.shape[0],-1)

                _, predicted_emotion = torch.max(preds,axis = 1)
                # predicted_emotion = torch.argmax(preds,dim=1)

                print(predicted_emotion)
                loss = self.criterion(preds,labels)
                avg_loss = (avg_loss * num_of_batches + loss) / \
                           (num_of_batches + 1)
                data_size += images.size(0)
                # update data size
                num_of_batches += 1
                all_preds.extend(predicted_emotion.cpu().detach().numpy())
                all_labels.extend(labels.cpu().detach().numpy())
        val_acc = accuracy_score(all_labels,all_preds)

        return avg_loss,val_acc

    def fit(self,trainloader,testloader):
        
        best_measure = -1
        best_epoch = -1
        if self.epochs is None or self.criterion is None or self.optimizer is None:
            raise ValueError(
                "Missing parameters \"epochs/criterion/optimizer\"")
        for epoch in range(self.epochs):
            print(f"Current Epoch: {epoch}")
            train_loss,_ = self.train(trainloader)
            valid_loss,measure = self.evaluate(testloader)
            print(f" Train Loss for this epoch {epoch}: {train_loss}")
            print(f" Val Loss for this epoch {epoch}: {valid_loss}")
            if measure > best_measure:
                    print(f'Updating best measure: {best_measure} -> {measure}')
                    best_epoch = epoch
                    best_weights = self.model.state_dict()
                    best_measure = measure
            self.train_losses.append(float(train_loss))
            self.valid_losses.append(float(valid_loss))
        return best_epoch,best_measure,best_weights
    # Predict 1 image
    def predict_one(self,image):
        self.model.eval()
        with torch.no_grad():
                preds = self.model(image)
                # preds = torch.squeeze(preds)
                # preds = preds.reshape(image.shape[0],-1)
                _, predicted_emotion = torch.max(preds,axis = 1)
                return predicted_emotion.item()
            
            
            