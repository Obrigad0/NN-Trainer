import torch 
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from timeit import default_timer as timer

# EPOCHS = 7
# LEARNING_RATE = 0.1
# ISLINEAR = False
# INPUT_SHAPE = 784 #28 * 28
# HIDDEN_LAYER_SHAPE = 64
# HIDDEN_LAYER = 1 #not used
# OUTPUT_SHAPE = 10

BATCH_SIZE = 32
DEVICE = "cuda"
MODELFILENAME = 'Trained Models/MNIST_Model_CLEAN.pth' #Change this to choose the model to test
NOMEDATASET = "" #Change this to choose the dataset to test

# !!! To test the MNIST BASE dataset you need to comment out the code below !!!

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100 
    return acc


def test_function(model, test_batch, loss_function):
     test_loss, test_acc = 0, 0
     model.eval()
     with torch.inference_mode():
         print("Testing...")

         for img, label in test_batch:
             
             img, label = img.to(DEVICE), label.to(DEVICE)
             
             pred = model(img)         

             test_loss += loss_function(pred, label)        
             test_acc += accuracy_fn(y_true = label, y_pred = pred.argmax(dim = 1))
     
         test_loss /= len(test_batch)
         test_acc /= len(test_batch)
     print("Testing result:")    
     print(f"Test loss: {test_loss:.5f} | Test acc: {test_acc:.2f}%")
     print("--------------------\n")
  
def visual_model_evaluation(model,test_data):
    
    model.eval()
    with torch.inference_mode():
        randindx = torch.randint(0,len(test_data), size = [1]).item()
        img, label = test_data[randindx]
        plt.imshow(img.squeeze(), cmap="gray")
        plt.axis(False)
        output = model(img.to(DEVICE))
        prediction = output.argmax(dim = 1, keepdim = True).item()
        
        print(f"The Model say... {prediction}!")
        print(f"The label say... {label}!")
        if( prediction == label):
            print("The model is correct!")
        else:
            print("The  model was wrong  :( ")
        print("\n")

class MNISTModelNONLinear(nn.Module):
    def __init__(self,
                 input_layer: int,
                 hidden_layer:int,
                 output_layer: int):        
        super().__init__()
        self.layer_stack = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=input_layer,out_features=hidden_layer),
                nn.ReLU(),
                nn.Linear(in_features=hidden_layer,out_features=output_layer),
                nn.ReLU()
            )
        
    def forward(self, x):
        return self.layer_stack(x)        
     
test_data =  torchvision.datasets.MNIST(
        root = "dataset",
        train = False,
        download = True,
        transform = torchvision.transforms.ToTensor(), 
        target_transform = None
    )

# test_data  = torch.load(NOMEDATASET) #COMMENT THIS CODE TO TEST THE BASIC MNIST

test_batch = DataLoader(
    dataset=test_data,
    batch_size = BATCH_SIZE,
    shuffle=True
    ) 

loss_function = nn.CrossEntropyLoss()

model = torch.load(MODELFILENAME)

test_function(model, test_batch, loss_function)
visual_model_evaluation(model, test_data)
