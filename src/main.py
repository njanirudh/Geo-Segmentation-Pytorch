import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss
import time

from dataloader import SegDatasetLoader
from model.unet import UNET

def train(model, train_dl, valid_dl, loss_fn, optimizer, acc_fn, epochs=1):
    start = time.time()
    model.cuda()

    train_loss, valid_loss = [], []

    best_acc = 0.0

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)  # Set trainind mode = true
                dataloader = train_dl
            else:
                model.train(False)  # Set model to evaluate mode
                dataloader = valid_dl

            running_loss = 0.0
            running_acc = 0.0

            step = 0

            # iterate over data
            for x, y in dataloader:
                x = torch.from_numpy(x).cuda()
                y = torch.from_numpy(y).cuda()
                step += 1

                # forward pass
                if phase == 'train':
                    # zero the gradients
                    optimizer.zero_grad()
                    outputs = model(x)
                    loss = loss_fn(outputs, y)

                    # the backward pass frees the graph memory, so there is no
                    # need for torch.no_grad in this training pass
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()

                else:
                    with torch.no_grad():
                        outputs = model(x)
                        loss = loss_fn(outputs, y.long())

                # stats - whatever is the phase
                acc = acc_fn(outputs, y)

                running_acc  += acc * dataloader.batch_size
                running_loss += loss * dataloader.batch_size

                if step % 10 == 0:
                    # clear_output(wait=True)
                    print('Current step: {}  Loss: {}  Acc: {}  AllocMem (Mb): {}'.format(step, loss, acc, torch.cuda.memory_allocated( ) /1024 /1024))
                    # print(torch.cuda.memory_summary())

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_acc / len(dataloader.dataset)

            print('{} Loss: {:.4f} Acc: {}'.format(phase, epoch_loss, epoch_acc))

            train_loss.append(epoch_loss) if phase=='train' else valid_loss.append(epoch_loss)

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return train_loss, valid_loss

def acc_metric(predb, yb):
    return (predb.argmax(dim=1) == yb.cuda()).float().mean()

if __name__ == "__main__":


    dataset_pth = "/home/anirudh/NJ/Interview/Vision-Impulse/Dataset/"
    seg_dataset = SegDatasetLoader(dataset_pth, 1, None)


    unet = UNET(12,1)
    xb, yb = next(iter(seg_dataset))

    print(xb.shape, yb.shape)
    print()
    # xb = xb[np.newaxis, ...]
    # xb = torch.from_numpy(xb)
    #
    # pred = unet(xb)
    # print(pred.shape)

    loss_fn = BCEWithLogitsLoss()
    opt = torch.optim.Adam(unet.parameters(), 0.01)
    train_loss, val_loss = train(model=unet,
                                 train_dl=seg_dataset,
                                 valid_dl=seg_dataset,
                                 loss_fn=loss_fn,
                                 optimizer=opt,
                                 acc_fn=acc_metric,
                                 epochs=50)
