from torch.autograd import Variable
import numpy as np
import torch


def train(net, dataloader, optim, loss_func, epoch):
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BATCH_REPORT_PERIOD = 5

    net.train()  # Put the network in train mode
    total_loss = 0
    batches = 0
    pred_store = []

    for batch_idx, data in enumerate(dataloader):
        data = Variable(data.to(DEVICE))
        target = Variable(data.to(DEVICE))
        batches += 1

        # Training loop
        data = data.float()  # convert to float32. Otherwise throws type mismatch
        target = target.float()  # convert to float32. Otherwise throws type mismatch

        optim.zero_grad()  # clear the gradient
        pred = net(data)  # make predictions
        # print(pred.shape)
        loss = loss_func(pred, target)  # calculate loss
        loss.backward()  # perform backpropagation

        optim.step()  # perform an optimization step

        total_loss += loss
        if batch_idx % BATCH_REPORT_PERIOD == 0:  # Report stats every x batches
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(dataloader.dataset),
                       100. * (batch_idx + 1) / len(dataloader), loss.item()), flush=True)
            pred_store.append(np.argmax(pred.detach().cpu().numpy(), axis=1))
        del data, target

    pred_store = np.array(pred_store)
    av_loss = total_loss / batches
    av_loss = av_loss.detach().cpu().numpy()
    print('\nTraining set: Average loss: {:.4f}'.format(av_loss, flush=True))

    return av_loss


def val(net, val_dataloader, optim, loss_func, epoch):
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.eval()  # Put the model in eval mode

    epsilon = 1e-9
    total_loss = 0
    batches = 0
    with torch.no_grad():  # So no gradients accumulate
        for batch_idx, data in enumerate(val_dataloader):
            batches += 1
            data = Variable(data.to(DEVICE))
            target = Variable(data.to(DEVICE))            # Eval steps

            data = data.float()  # convert to float32. Otherwise throws type mismatch
            target = target.float()  # convert to float32. Otherwise throws type mismatch

            pred = net(data)  # make prediction
            loss = loss_func(pred, target)  # calculate loss

            total_loss += loss
        av_loss = total_loss / (batches + epsilon)

    av_loss = av_loss.detach().cpu().numpy()
    print('Validation set: Average loss: {:.4f}'.format(av_loss, flush=True))
    print('\n')

    return av_loss


def predict(net, test_dataloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    store = list()

    net.eval()  # Put the model in eval mode
    batches = 0
    with torch.no_grad():  # So no gradients accumulate
        for batch_idx, data in enumerate(test_dataloader):
            batches += 1
            data = Variable(data.to(device))
            target = Variable(data.to(device))
            data = data.float()
            target = target.float()
            pred = net(data)  # make prediction

            store.append((pred, target))


    return store