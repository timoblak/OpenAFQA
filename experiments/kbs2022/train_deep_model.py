import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import numpy as np
import sys
from learning_functions import FingermarkDataset, load_data
from models import AFQAModel, save_model, r2_score_torch
from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":
    # ----- CUDA
    device_name = "cuda:0" #if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    print("Using device: " + device_name)

    # ----- Datasets
    METRIC = "all"
    OUTPUTS = 4
    IMSIZE = 512
    BATCH_SIZE = 8
    FCN = 512

    DATA_FOLDER = "cnn_dataset/"
    LABELS = DATA_FOLDER + "qualities.pkl"

    labels = load_data(DATA_FOLDER + "train/", LABELS, METRIC)
    
    #np.random.seed(123)
    np.random.shuffle(labels)

    db_split = 0.9
    training_data = labels[:int(db_split * len(labels))]
    validation_data = labels[int(db_split * len(labels)):]

    print("Training data size: " + str(len(training_data)) + ", Validation data size: " + str(len(validation_data)))

    # ----- Generators
    dataset = FingermarkDataset(DATA_FOLDER + "train/", training_data, imsize=IMSIZE)
    training_generator = data.DataLoader(dataset, **{
        'batch_size': BATCH_SIZE,
        'shuffle': True,
        'num_workers': 6
    })
    dataset_val = FingermarkDataset(DATA_FOLDER + "train/", validation_data, imsize=IMSIZE)
    validation_generator = data.DataLoader(dataset_val, **{
        'batch_size': BATCH_SIZE,
        'shuffle': False,
        'num_workers': 6
    })

    # ----- Hyperparameter initialization

    TAG = METRIC + "_densenet121"
    MODEL_LOCATION = "models/model_" + TAG + ".pt"
    MAX_EPOCHS = 20000
    LEARNING_RATE = 1e-4
    LOG_INTERVAL = 1
    RUNNING_MEAN = 100
    DEBUG = False
    PRETRAIN_EPOCHS = 0
    CONTINUE_TRAINING = False
    EARLY_STOPPING = 14  # number of epochs without change before training is stopped

    # ----- Net initialization

    net = AFQAModel(outputs=OUTPUTS, fcn=FCN, pretrained=True).to(device)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)#, weight_decay=0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=6, verbose=True, factor=0.1)
    writer = SummaryWriter(log_dir="runs/log_" + TAG)

    starting_epoch = 0
    if CONTINUE_TRAINING:
        print("Continuing with training... IMPLEMENT")
        #starting_epoch, net, optimizer, _ = load_model(MODEL_LOCATION, net, optimizer)

    # ----- Training config
    loss_l2 = nn.MSELoss()
    loss_l1 = nn.L1Loss()

    # ----- Main loop
    no_change_epoch = 0
    best_val_loss = None
    metrics = {
        "train_l1": [], "val_l1": [],
        "train_l2": [], "val_l2": [],
        "train_r2": [], "val_r2": [],
    }
    for epoch in range(starting_epoch, MAX_EPOCHS):
        # ----- Data
        epoch_metrics = {
            "train_l1": [], "val_l1": [],
            "train_l2": [], "val_l2": [],
            "train_r2": [], "val_r2": [],
        }

        # ##########  TRAINING  ##########
        net.train()
        for batch_idx, (data, true_labels) in enumerate(training_generator):

            # Transfer to GPU
            data, true_labels = data.to(device), true_labels.to(device)
            
            optimizer.zero_grad()

            # Predict
            pred_labels, _ = net(data)

            # Calculate loss and backpropagate
            l2_loss = loss_l2(true_labels, pred_labels)
            l1_loss = loss_l1(true_labels, pred_labels)
            r2_acc = r2_score_torch(true_labels, pred_labels)
            l2_loss.backward()

            # Update weights
            optimizer.step()

            # Save loss
            epoch_metrics["train_l2"].append(l2_loss.item())
            epoch_metrics["train_l1"].append(l1_loss.item())
            epoch_metrics["train_r2"].append(r2_acc.item())

            #if torch.any(torch.isnan(net.encoder.fc.weight.grad)):
            #    print("--------------- NAN GRADS!!!! ---------------")
            #    exit()

            if batch_idx % LOG_INTERVAL == 0 and not DEBUG:
                sys.stdout.write("\033[K")
                print('Train Epoch: {} Step: {} [{}/{} ({:.0f}%)]\tMSE Loss: {:,.6f}\tMAE Loss: {:,.6f}\tR2 Accuracy: {:,.6f}'.format(
                    epoch, batch_idx, batch_idx * len(data), len(training_generator.dataset), 100. * batch_idx / len(training_generator),
                    np.mean(epoch_metrics["train_l2"][-RUNNING_MEAN:]), np.mean(epoch_metrics["train_l1"][-RUNNING_MEAN:]), np.mean(epoch_metrics["train_r2"][-RUNNING_MEAN:])), end="\r")

        # Print last
        sys.stdout.write("\033[K")
        print('Train Epoch: {} Step: {} [(100%)]\tMSE Loss: {:.6f}\tMAE Loss: {:.6f}\tR2 Accuracy: {:,.6f}'.format(epoch, batch_idx, np.mean(epoch_metrics["train_l2"]), np.mean(epoch_metrics["train_l1"]), np.mean(epoch_metrics["train_r2"])))

        # Save the mean od losses for this epoch
        metrics["train_l1"].append(np.mean(epoch_metrics["train_l1"]))
        metrics["train_l2"].append(np.mean(epoch_metrics["train_l2"]))
        metrics["train_r2"].append(np.mean(epoch_metrics["train_r2"]))

        # ##########  VALIDATION  ##########
        net.eval()
        with torch.no_grad():
            for batch_idx, (data, true_labels) in enumerate(validation_generator):
                data, true_labels = data.to(device), true_labels.to(device)

                # Predict
                pred_labels, _ = net(data)

                # Calculate loss and backpropagate
                l2_loss = loss_l2(true_labels, pred_labels)
                l1_loss = loss_l1(true_labels, pred_labels)
                r2_acc = r2_score_torch(true_labels, pred_labels)

                epoch_metrics["val_l2"].append(l2_loss.item())
                epoch_metrics["val_l1"].append(l1_loss.item())
                epoch_metrics["val_r2"].append(r2_acc.item())

        writer.add_figure('val predictions vs. actuals', plot_classes_preds(data, pred_labels, true_labels, num=10), global_step=epoch)

        metrics["val_l1"].append(np.mean(epoch_metrics["val_l1"]))
        metrics["val_l2"].append(np.mean(epoch_metrics["val_l2"]))
        metrics["val_r2"].append(np.mean(epoch_metrics["val_r2"]))

        writer.add_scalar("Loss/MSE_train", metrics["train_l2"][-1], epoch)
        writer.add_scalar("Loss/MAE_train", metrics["train_l1"][-1], epoch)
        writer.add_scalar("Loss/R2_train", metrics["train_r2"][-1], epoch)

        writer.add_scalar("Loss/MSE_val", metrics["val_l2"][-1], epoch)
        writer.add_scalar("Loss/MAE_val", metrics["val_l1"][-1], epoch)
        writer.add_scalar("Loss/R2_val", metrics["val_r2"][-1], epoch)

        # Based on L2 loss, stop training
        early_stop_metric = "val_l2"
        scheduler.step(metrics[early_stop_metric][-1])

        if best_val_loss is None:
            print("Saving first model..")
            best_val_loss = metrics[early_stop_metric][-1]
            save_model(MODEL_LOCATION, epoch, net, optimizer, metrics)
        elif metrics[early_stop_metric][-1] < best_val_loss:
            print("New best loss (" + str(round(metrics[early_stop_metric][-1], 2)) + " < " + str(round(best_val_loss, 2)) + ") achieved. Saving model ..")
            best_val_loss = metrics[early_stop_metric][-1]
            no_change_epoch = 0
            save_model(MODEL_LOCATION, epoch, net, optimizer, metrics)
        else:
            no_change_epoch += 1

        print("------------------------------------------------------------------------")
        print('Validation Epoch: {} Step: {} [(100%)]\tVal MSE Loss: {:,.6f}\tVal MAE Loss: {:,.6f}\tVal R2 Accuracy: {:,.6f}'.format(epoch, batch_idx, metrics["val_l2"][-1], metrics["val_l1"][-1], metrics["val_r2"][-1]))
        print("========================================================================")

        if no_change_epoch >= EARLY_STOPPING:
            print("Stopping training, went " + str(EARLY_STOPPING) + " without improvements.")
            break

    writer.close()
    #training_set.close()
    #validation_set.close()
