from pathlib import Path
import torch
from torch.utils.data import random_split
from src.models.Classifier import Classifier
from torch import nn, optim
import matplotlib.pyplot as plt
import pickle
import logging


def train_model(trained_model_filepath,
                training_statistics_filepath, training_figures_filepath):

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    logger.info('Training a fish classifier')


    project_dir = Path(__file__).resolve().parents[2]
    train_set_path = str(project_dir) + '/data/processed/training.pt'
    labels_path = str(project_dir) + '/data/processed/labels.pt'
    train_imgs, train_labels = torch.load(train_set_path) # img, label
    labels_as_string = torch.load(labels_path) # img, label
    print(train_imgs[0])
    print(labels_path)
    print(labels_as_string)



    
    #load data
    train_set = torch.utils.data.TensorDataset(train_imgs, train_labels)

    #split data in training and validation set
    train_n = int(0.65*len(train_set))
    val_n = len(train_set) - train_n
    train_data,val_data = random_split(train_set,[train_n, val_n])
    print(f"Length of Train Data : {len(train_data)}")
    print(f"Length of Validation Data : {len(val_data)}")

    ##### Hyper parameters
    batch_size = 64
    num_classes = len(labels_as_string)
    rgb = train_imgs.shape[1]
    height = train_imgs.shape[2]
    width  = train_imgs.shape[3]
    filter1_in = rgb
    filter1_out = 6
    kernel = 2
    pool = 2
    filter2_out = 16
    filter3_out = 48
    fc_1 = 120
    fc_2 = 84 
    pad = 0
    stride = 1 
    lr = 0.001
    epochs = 30
    

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                          shuffle=True, num_workers=0) #changed num_workers to 0 because i was getting error

    valoader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,
                                          shuffle=True, num_workers=0)  
    
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    print('Image shape',images.shape)
    print('Labels shape',labels.shape)
    
    
    model = Classifier(num_classes, filter1_in, filter1_out, filter2_out, filter3_out,height, width, pad, stride, kernel,pool,fc_1,fc_2 )
    print(model)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Implement the training loop
    print('Start training')
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    for e in range(epochs):
        train_loss = 0
        train_correct = 0

        for images, labels in trainloader:
            # Set model to training mode and zero
            #  gradients since they accumulated
            model.train()
            optimizer.zero_grad()

            # Make a forward pass through the network to get the logits
            log_ps = model(images)
            ps = torch.exp(log_ps)

            # Use the logits to calculate the loss
            loss = criterion(log_ps, labels.long())
            train_loss += loss.item()

            # Perform a backward pass through the network
            #  to calculate the gradients
            loss.backward()

            # Take a step with the optimizer to update the weights
            optimizer.step()

            # Keep track of how many are correctly classified
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            train_correct += equals.type(torch.FloatTensor).sum().item()
        else:
            # Compute validattion loss and accuracy
            val_loss = 0
            val_correct = 0

            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                model.eval()     # Sets the model to evaluation mode
                for images, labels in valoader:
                    # Forward pass and compute loss
                    log_ps = model(images)
                    ps = torch.exp(log_ps)
                    val_loss += criterion(log_ps, labels.long()).item()

                    # Keep track of how many are correctly classified
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    val_correct += equals.type(torch.FloatTensor).sum().item()

            # Store and print losses and accuracies
            train_losses.append(train_loss/len(trainloader))
            train_accuracies.append(train_correct/len(train_data))
            val_losses.append(val_loss/len(valoader))
            val_accuracies.append(val_correct/len(val_data))

            logger.info(str("Epoch: {}/{}.. ".format(e+1, epochs)) +
                        str("Training Loss: {:.3f}.. ".format(train_losses[-1])) +
                        str("Training Accuracy: {:.3f}.. ".format(train_accuracies[-1])) +
                        str("Validation Loss: {:.3f}.. ".format(val_losses[-1]))         +
                        str("Validation Accuracy: {:.3f}.. ".format(val_accuracies[-1])))

    # Save the trained network
    torch.save(model.state_dict(),
               project_dir.joinpath(trained_model_filepath))

    # Save the training and validation losses and accuracies as a dictionary
    train_val_dict = {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies
    }

    with open(project_dir.joinpath(training_statistics_filepath).joinpath('train_val_dict.pickle'), 'wb') as f:
        # Pickle the 'train_val_dict' dictionary using
        #  the highest protocol available
        pickle.dump(train_val_dict, f, pickle.HIGHEST_PROTOCOL)

    # Plot the training loss curve
    f = plt.figure(figsize=(12, 8))
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses,   label='Validation loss')
    plt.xlabel('Epoch number')
    plt.ylabel('Loss')
    plt.legend()
    f.savefig(project_dir.joinpath(training_figures_filepath).joinpath('Training_Loss.pdf'),
              bbox_inches='tight')

    # Plot the training accuracy curve
    f = plt.figure(figsize=(12, 8))
    plt.plot(train_accuracies, label='Training accuracy')
    plt.plot(val_accuracies,   label='Validation accuracy')
    plt.xlabel('Epoch number')
    plt.ylabel('Accuracy')
    plt.legend()
    f.savefig(project_dir.joinpath(training_figures_filepath).joinpath('Training_Accuracy.pdf'),
              bbox_inches='tight')

    return train_val_dict



                                    