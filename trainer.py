import torch 
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from timeit import default_timer as timer

# -- HYPERPARAMETERS -- 

BATCH_SIZE = 32
EPOCHS = 7
LEARNING_RATE = 0.1
HIDDEN_LAYER_SHAPE = 64
DEVICE = "cuda"
ISLINEAR = False

INPUT_SHAPE = 784 #28 * 28
HIDDEN_LAYER = 1 #not used
OUTPUT_SHAPE = 10

# ----------------------
#   Best setup found  
# B = 32 , E = 5 , HLS = 64 , LR = 0.1
# B = 16



def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100 
    return acc

def train_time(start: float,stop: float):
    print(f"\nTrain time : {(stop - start):.3f} seconds")


def train_function(model, train_batch, loss_function, optimizer):

        train_loss , train_acc = 0, 0
        model.train()
        print("Training...")
        for batch, (img , label) in enumerate(train_batch):
            img, label = img.to(DEVICE), label.to(DEVICE)
            
            pred = model(img)
            
            loss = loss_function(pred, label)
            train_loss += loss
            train_acc += accuracy_fn(y_true= label, y_pred = pred.argmax(dim = 1))
            
            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()
                                        
        train_loss /= len(train_batch)
        train_acc /=  len(train_batch)  
        print("Training result:")
        print(f"Train loss: {train_loss:.5f} | Train acc: {train_acc:.2f}%")
        print("--------------------\n")

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

def final_model_evaluation(model,test_data):
    
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
            print(f"The model is correct!")
        else:
            print(f"The  model was wrong  :( ")
        print("\n")
        
        
train_data = torchvision.datasets.MNIST(
        root = "dataset",
        train = True,
        download = True,
        transform = torchvision.transforms.ToTensor(), 
        target_transform = None
    )

test_data =  torchvision.datasets.MNIST(
        root = "dataset",
        train = False,
        download = True,
        transform = torchvision.transforms.ToTensor(), 
        target_transform = None
    )


train_batch = DataLoader(
    dataset=train_data,
    batch_size = BATCH_SIZE,
    shuffle=True)

test_batch = DataLoader(
    dataset=test_data,
    batch_size = BATCH_SIZE,
    shuffle=False) 


#Linear model
class MNISTModelLinear(nn.Module):
    def __init__(self,
                 input_layer: int,
                 hidden_layer:int,
                 output_layer: int):        
        super().__init__()
        self.layer_stack = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=input_layer,
                          out_features=hidden_layer),
                #
                # nn.Linear(in_features = hidden_layer, out_features = hidden_layer),
                #
                nn.Linear(in_features=hidden_layer,
                          out_features=output_layer)
            )
        
    def forward(self, x):
        return self.layer_stack(x)

#NONLinear model
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
                #
                # nn.Linear(in_features = hidden_layer, out_features = hidden_layer),
                # nn.ReLU(),
                #
                nn.Linear(in_features=hidden_layer,out_features=output_layer),
                nn.ReLU()
            )
        
    def forward(self, x):
        return self.layer_stack(x)

print("\n Model infos:\n")


if ISLINEAR:
    print(f"Linear Model | Device: {DEVICE} | ")

    model = MNISTModelLinear(INPUT_SHAPE,
        HIDDEN_LAYER_SHAPE, 
        OUTPUT_SHAPE
    ).to(DEVICE)
    
else:
    print(f"NONLinear Model | Device: {DEVICE} | ")

    model = MNISTModelNONLinear(INPUT_SHAPE,
        HIDDEN_LAYER_SHAPE, 
        OUTPUT_SHAPE
    ).to(DEVICE)

print(f"Epochs: {EPOCHS} | Hidden Layer neurons: {HIDDEN_LAYER_SHAPE} | Batch size: {BATCH_SIZE} | LR: {LEARNING_RATE}\n\n")

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params = model.parameters(),
                            lr = LEARNING_RATE)



start_time = timer()
for epoch in range(EPOCHS):
    print("================================")
    print(f"\nEpoch: {epoch + 1}/{EPOCHS}\n--------------------")
    train_function(model, train_batch, loss_function, optimizer)
    test_function(model, test_batch, loss_function)

stop_time = timer()
train_time(start_time, stop_time)


print(f"RECAP:: Epochs: {EPOCHS} | Hidden Layer neurons: {HIDDEN_LAYER_SHAPE} | Batch size: {BATCH_SIZE} | LR: {LEARNING_RATE} \n\n")

final_model_evaluation(model, test_data)
test_function(model, test_batch, loss_function)

torch.save(model, 'MNIST_Model_CLEAN.pth')
# torch.save(model.state_dict(), 'MNIST_Model_WEIGHT_CLEAN.pth')





