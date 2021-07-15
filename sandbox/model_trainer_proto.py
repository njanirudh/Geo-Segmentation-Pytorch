import time

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from src.model.unet import UNET
from src.seg_dataset import SegDataset


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
                model.train(True)  # Set training mode = true
                dataloader = train_dl
            else:
                model.train(False)  # Set model to evaluate mode
                dataloader = valid_dl

            running_loss = 0.0
            running_acc = 0.0

            step = 0

            # iterate over data
            for x, y in dataloader:
                x = x.cuda()
                y = y.cuda()

                # print(x.shape, y.shape)
                # print(np.unique(y))

                step += 1

                # forward pass
                if phase == 'train':
                    # zero the gradients
                    optimizer.zero_grad()
                    outputs = model(x)
                    # print(outputs.shape)
                    # xxx = np.argmax(outputs.detach().numpy(), axis=1)
                    # # print(np.unique(np.equal(xxx,y),return_counts=True))
                    # print(xxx.shape, outputs.shape, y.shape)
                    loss = loss_fn(outputs, y.long())
                    print(loss)

                    # print(loss.shape)
                    # the backward pass frees the graph memory, so there is no
                    # need for torch.no_grad in this training pass
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()

                else:
                    with torch.no_grad():
                        outputs = model(x)
                        loss = loss_fn(outputs, y.long())
                        # print(loss.sum())

                # stats - whatever is the phase
                acc = acc_fn(outputs, y)

                running_acc += acc * dataloader.batch_size
                running_loss += loss * dataloader.batch_size

                if step % 10 == 0:
                    # clear_output(wait=True)
                    print('Current step: {}  Loss: {}  Acc: {}  AllocMem (Mb): {}'.format(step, loss, acc,
                                                                                          torch.cuda.memory_allocated() / 1024 / 1024))
                    # print(torch.cuda.memory_summary())

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_acc / len(dataloader.dataset)

            # print('{} Loss: {:.4f} Acc: {}'.format(phase, epoch_loss, epoch_acc))

            train_loss.append(epoch_loss) if phase == 'train' else valid_loss.append(epoch_loss)

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    torch.save(model, "../seg_model_1.pkl")
    return train_loss, valid_loss


def acc_metric(predb, yb):
    return (predb.argmax(dim=1) == yb.cuda()).float().mean()


if __name__ == "__main__":
    dataset_pth = "/home/anirudh/NJ/Interview/Vision-Impulse/Dataset/"
    seg_dataset = SegDataset(dataset_pth)

    unet = UNET(12, 3)
    # summary(unet,(12,64,64),device='cpu')

    loss_fn = CrossEntropyLoss()
    opt = torch.optim.Adam(unet.parameters(), 0.0003)

    seg_dl = DataLoader(seg_dataset,
                        batch_size=15,
                        shuffle=True,
                        num_workers=2)

    train_loss, val_loss = train(model=unet,
                                 train_dl=seg_dl,
                                 valid_dl=seg_dl,
                                 loss_fn=loss_fn,
                                 optimizer=opt,
                                 acc_fn=acc_metric,
                                 epochs=150)
