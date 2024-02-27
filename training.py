import os
import gc
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from torchvision.transforms import transforms as T
import torch.optim as optim
from torchvision.models import vgg19
from torchvision.models import VGG19_Weights
from sklearn.metrics import recall_score, f1_score, precision_score
from tqdm import tqdm


ans = input('Did you update the paths for data_train, labels_train, and save_path! Type YES or NO.\n').upper()
if ans == 'YES':
    print("Running file...")
else:
    print("Exiting program...")
    exit()

data_path = 'data_train.npy'
labels_path = 'labels_train.npy'
save_path = '/blue/eel5840/darian.jennings/final_proj/'
os.makedirs(save_path, exist_ok=True)


def train():
    data_train = np.load(data_path)
    labels_train = np.load(labels_path)

    # Standard IMAGENET vals sourced from google
    imagenet_means = [0.485, 0.456, 0.406]
    imagenet_stds = [0.229, 0.224, 0.225]

    # Define AutoAugment transform 
    # - automatically augments data based on a given auto-augmentation policy - IMAGENET
    augmenter = AutoAugment(policy=AutoAugmentPolicy.IMAGENET)

    # Resize, augment,...
    # NOTE - In PyTorch, T.ToTensor() is a transformation that converts a PIL image or numpy array to a 
    # tensor and scales the values to the range [0.0, 1.0] 
    # NOTE - Normalize expects Tensor - so convert to tensor then normalize (*order matters*)

    # create a Tensor dataset (performs transformations AND augmentation)
    class TensorDataset(Dataset):    
        def __init__(self, data, labels, transform=None):
            self.data = data 
            self.labels = labels 
            self.transform = transform

        def __len__(self):
            return self.data.shape[1]

        def __getitem__(self, idx):
            image = self.data[:, idx].reshape(300, 300, 3)
            image = Image.fromarray(image) 
            if self.transform:
                image = self.transform(image)
            label = self.labels[idx] 
            return image, label

    preprocess = T.Compose([
        augmenter,
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(imagenet_means, imagenet_stds)
    ])

    dataset = TensorDataset(data_train, labels_train, transform=preprocess)
    print("Created Tensor Dataset")

    # Create split sizes for training-validation-test, ---- use 70-15-15 rule ----
    train_size = int(0.7 * len(dataset)) 
    temp_size = len(dataset) - train_size 
    val_size = int(0.5 * temp_size) 
    test_size = temp_size - val_size

    # Split dataset into training-validation-test using sizes (random_split)
    train_dataset, temp_dataset = random_split(dataset, [train_size, temp_size])
    val_dataset, test_dataset = random_split(temp_dataset, [val_size, test_size])

    # Use DataLoader - load data into model - flexible for memory constraints
    train_dataflow = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_dataflow = DataLoader(val_dataset, batch_size=256)
    test_dataflow = DataLoader(test_dataset, batch_size=256)
    print("Created dataflows")

    # Create instance of pre-trained VGG19 model
    def pretrainedVGG19(num_classes):
        model = vgg19(weights=VGG19_Weights.DEFAULT)           # pre-trained ImageNet weights
        num_ftrs = model.classifier[6].in_features             # extract n_ftrs in output layer
        model.classifier[6] = nn.Linear(num_ftrs, num_classes) # n_output_features= n_classes
        return model

    # Define device to utilize and num_of_classes for model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    # Call model with give number of classes
    nclasses = 9
    model = pretrainedVGG19(nclasses)
    model.to(device)

    # Define loss, add weight decay to optimizer (standard is 1e_05) - L2 penalty term
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-05)

    # Track epoch with highest accuracy for epochs 1-N
    highest_val_accuracy = 0

    # Track which indices were marked as the unknown class - should be 0 at end of training since they are all known
    # Store vals for train, loss, and accuracies respectively
    unknown_indices = []
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Initialize dictionary to store information about best model
    best_model_info = {
        'epoch': None,
        'train_loss': None,
        'train_accuracy': None,
        'val_loss': None,
        'val_accuracy': None,
    }

    # Collect garbage
    collected = gc.collect()

    num_epochs = 25
    for epoch in range(num_epochs):
        model.train()
        # reset for each epoch
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in tqdm(train_dataflow, desc=f"Epoch {epoch+1}/{num_epochs} [Training]"):
            images, labels = images.to(device, dtype=torch.float), labels.to(device, dtype=torch.long)
            
            # raw output scores
            outputs = model(images)
            # calculate the probabilities
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            #max_prob, _ = torch.max(probabilities, dim=1)
            #print(max_prob)

            # check if all probabilities are below 0.1
            unknown = (probabilities < 0.1).all(dim=1)
            # get the predicted classes
            _, predicted = torch.max(outputs.data, 1)
            # assign -1 to the unknown class
            predicted[unknown] = -1
            unknown_indices.extend(i for i, x in enumerate(predicted) if x == -1)
            
            # calculate loss petween predicted outputs and labels
            loss = criterion(outputs, labels)
            # set grads to zero, make suere we don’t accumulate gradients from previous iterations
            optimizer.zero_grad()
            # computes the gradients of the loss function
            loss.backward()
            # updates the parameters of the neural network
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1) 
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_accuracy = 100 * train_correct / train_total

        # evaluate
        model.eval()
        # calculate validation loss and accuracy
        val_loss, val_correct, val_total = 0.0, 0, 0 
        # deactivate autograd engine - prevent updates & data leakage during validation
        with torch.no_grad():
            for inputs, labels in val_dataflow:
                inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device, dtype=torch.long)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_accuracy = 100 * val_correct / val_total
        
        # check if current accuracy is higher than the highest accuracy
        if val_accuracy > highest_val_accuracy:
            highest_val_accuracy = val_accuracy
                
            # Send info to dictionary
            best_model_info['epoch'] = epoch
            best_model_info['train_loss'] = train_loss / len(train_dataflow)
            best_model_info['train_accuracy'] = train_accuracy
            best_model_info['val_loss'] = val_loss / len(val_dataflow)
            best_model_info['val_accuracy'] = val_accuracy
            best_model_info['model_state_dict'] = model.state_dict()

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        print(f"Train Acc: {train_accuracy:.2f}%, Train Loss: {train_loss/len(train_dataflow)},Val Acc: {val_accuracy:.2f}%, Val Loss: {val_loss/len(val_dataflow)}")

    # Save the best model --- information is stored in dictionary (best_model_info)
    torch.save(best_model_info['model_state_dict'], os.path.join(save_path, 'opt_model.pth'))
    # Pop off model.dict -- for printing purposes (clean)
    best_model_info.popitem()
    print("---TRAINING COMPLETE---")
    print("Unknown indices: ", unknown_indices)
    print("Best model info: ", best_model_info)
    collected = gc.collect()

    # Plot learning curves - training vs validation - for loss & accuracy
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    axs[0].set_title("Training and Validation Loss")
    axs[0].plot(train_losses,label="train")
    axs[0].plot(val_losses,label="val")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    axs[1].set_title("Training and Validation Accuracy")
    axs[1].plot(train_accuracies,label="train")
    axs[1].plot(val_accuracies,label="val")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()

    plt.show()


if __name__ == "__main__":
    train()
