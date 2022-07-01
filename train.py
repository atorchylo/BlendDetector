import os
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from metrics.utils import draw_confusion
from sklearn.metrics import accuracy_score, f1_score


from config import TRAIN


def get_time():
    """Get string with current time for logging purposes"""
    from datetime import datetime
    now = datetime.now()
    return now.strftime("%d-%m-%Y"), now.strftime("%H-%M-%S")


def train(model, optimizer, criterion, train_loader, valid_loader,
          num_epochs=TRAIN.epochs,
          valid_loss_min=np.inf,
          device=TRAIN.device,
          log_dir=TRAIN.log_dir
):

    # logging
    day, time = get_time()
    save_dir = os.path.join(log_dir, model.name, day, time)
    save_path = os.path.join(save_dir, 'checkpoint.pt')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    tb_writer = SummaryWriter(log_dir=save_dir)

    for e in range(num_epochs):
        train_loss = 0.0
        valid_loss = 0.0
        ###################
        # train the model #
        ###################
        model.train()
        for imgs, labels in tqdm(train_loader, desc="Training loop"):
            imgs = imgs.to(device)
            labels = labels.to(device)
            # forward
            out = model(imgs)
            loss = criterion(out, labels)
            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # logg
            train_loss += loss.item()

        ######################
        # evaluate the model #
        ######################
        model.eval()
        with torch.no_grad():
            targets, predictions = [], []
            for imgs, labels in tqdm(valid_loader, desc="Evaluation loop"):
                imgs = imgs.to(device)
                labels = labels.to(device)
                # forward
                out = model(imgs)
                loss = criterion(out, labels)
                # logging
                prediction = out.argmax(1).cpu().numpy()
                target = labels.cpu().numpy()
                predictions.extend(prediction)
                targets.extend(target)
                valid_loss += loss.item()

        # calculate average metrics per batch
        train_loss /= len(train_loader)
        valid_loss /= len(valid_loader.sampler)
        acc = accuracy_score(predictions, targets)
        f1_macro = f1_score(predictions, targets, average='macro')
        confusion_fig = draw_confusion(targets, predictions, model.num_cls)
        # Tensorboards Logging
        tb_writer.add_scalar('Train Loss', train_loss, e)
        tb_writer.add_scalar('Valid Loss', valid_loss, e)
        tb_writer.add_scalar('Valid ACC', acc, e)
        tb_writer.add_scalar('Valid F1', f1_macro, e)
        tb_writer.add_figure('Confusion', confusion_fig)

        # print values
        print(f'Epoch: {e + 1} TrainLoss: {train_loss:.3f} ValidLoss: {valid_loss:.3f} ACC: {acc:.3f} F1_MACRO: {f1_macro}')

        ######################
        #   save the model   #
        ######################
        if valid_loss <= valid_loss_min:
            torch.save({
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_loss_min': valid_loss
            }, save_path)
            valid_loss_min = valid_loss


def main():
    from src.resnet import ResNet18
    from torch import nn
    from torch.optim import Adam

    # set up the model
    model = ResNet18(TRAIN.in_ch, TRAIN.num_cls).to(TRAIN.device)
    optimizer = Adam(model.parameters(), lr=TRAIN.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # load the data
    from src.dataloaders import NormalizeSCARLET, get_dataloaders
    transform = NormalizeSCARLET(TRAIN.Q, TRAIN.S)
    train_loader, valid_loader = get_dataloaders(
        TRAIN.train_data_path,
        TRAIN.test_data_path,
        TRAIN.batch_size,
        transform,
        test_run=True
    )

    # train the model
    train(model, optimizer, criterion, train_loader, valid_loader)


if __name__ == "__main__":
    main()
