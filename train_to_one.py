import os
import time
import torch
import torch.optim as optim
import argparse
import numpy as np

from torch.utils.data import DataLoader
from models import BLSTMConversionModel, BLSTMResConversionModel
from datasets import VCDataset, collate_fn
from config import Hparams
from utils import draw_melspectrograms, masked_mse_loss, seed_everything

# set up device
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
print('Train on {}'.format(device))


def main():
    parser = argparse.ArgumentParser('bnf-VC trainer')
    parser.add_argument('--test_dir', type=str, help='test data save directory')
    parser.add_argument('--model_dir', type=str, help='model ckpt save directory')
    parser.add_argument('--data_dir', type=str, help='data directory containing the *_meta.csv')
    parser.add_argument('--use_res', type=bool, default=False, help='use residual connection or not')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    args = parser.parse_args()

    seed_everything(args.seed)

    train_meta_file = os.path.join(args.data_dir, 'train_meta.csv')
    dev_meta_file = os.path.join(args.data_dir, 'dev_meta.csv')
    test_meta_file = os.path.join(args.data_dir, 'test_meta.csv')

    # validate args
    if not os.path.isdir(args.data_dir):
        raise NotADirectoryError('{} is not a valid directory'.format(args.data_dir))
    else:
        if not os.path.isfile(train_meta_file):
            raise FileNotFoundError('{} is not a valid path'.format(train_meta_file))
        if not os.path.isfile(dev_meta_file):
            raise FileNotFoundError('{} is not a valid path'.format(dev_meta_file))
        if not os.path.isfile(test_meta_file):
            raise FileNotFoundError('{} is not a valid path'.format(test_meta_file))
    if not os.path.isdir(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.isdir(args.test_dir):
        os.makedirs(args.test_dir)

    # set up dataset loader
    hps = Hparams()
    train_set = VCDataset(args.data_dir, train_meta_file)
    dev_set = VCDataset(args.data_dir, dev_meta_file)
    test_set = VCDataset(args.data_dir, test_meta_file)
    train_dataloader = DataLoader(train_set, batch_size=hps.TrainToOne.train_batch_size,
                                  shuffle=hps.TrainToOne.shuffle,
                                  num_workers=hps.TrainToOne.num_workers,
                                  collate_fn=collate_fn)
    dev_dataloader = DataLoader(dev_set, batch_size=hps.TrainToOne.train_batch_size,
                                shuffle=hps.TrainToOne.shuffle,
                                num_workers=hps.TrainToOne.num_workers,
                                collate_fn=collate_fn)
    test_dataloader = DataLoader(test_set, batch_size=hps.TrainToOne.test_batch_size,
                                 shuffle=hps.TrainToOne.shuffle,
                                 num_workers=hps.TrainToOne.num_workers,
                                 collate_fn=collate_fn)
    # set up model
    if args.use_res:
        model = BLSTMResConversionModel(in_channels=hps.Audio.bn_dim + 2,
                                        out_channels=hps.Audio.num_mels,
                                        lstm_hidden=hps.BLSTMConversionModel.lstm_hidden,
                                        hidden_dim=hps.BLSTMResConversionModel.hidden_dim)
    else:
        model = BLSTMConversionModel(in_channels=hps.Audio.bn_dim + 2,
                                     out_channels=hps.Audio.num_mels,
                                     lstm_hidden=hps.BLSTMConversionModel.lstm_hidden)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=hps.TrainToOne.learning_rate)

    train_loss = []
    valid_loss = []
    # start training
    for epoch in range(hps.TrainToOne.epochs):
        # training
        model.train()
        running_loss = 0.
        for idx, batch in enumerate(train_dataloader):
            # run forward pass
            optimizer.zero_grad()
            inputs = torch.cat([batch['bnf'], batch['f0']], dim=2).to(device)
            outputs = model(inputs)
            target_mels = batch['mel'].to(device)
            lengths = batch['length'].to(device)
            # run backward pass
            loss = masked_mse_loss(outputs.transpose(0, 1),
                                   target_mels.transpose(0, 1),
                                   lengths)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if idx % 1 == 0:  # print every batch
                print('[%d, %5d] Training loss: %.5f' %
                      (epoch + 1, idx + 1, running_loss))
                train_loss.append(running_loss)
                running_loss = 0.0
        # save model parameters
        torch.save(model.state_dict(), os.path.join(args.model_dir, "bnf-vc-to-one-{}.pt".format(epoch)))
        # validation
        model.eval()
        dev_running_loss = 0.
        for dev_batch in dev_dataloader:
            dev_inputs = torch.cat([dev_batch['bnf'], dev_batch['f0']], dim=2).to(device)
            dev_outputs = model(dev_inputs)
            dev_target_mels = dev_batch['mel'].to(device)
            dev_lengths = dev_batch['length'].to(device)
            # run backward pass
            dev_loss = masked_mse_loss(dev_outputs.transpose(0, 1),
                                       dev_target_mels.transpose(0, 1),
                                       dev_lengths)
            dev_running_loss += dev_loss
        print('[%d] Validation loss: %.5f' %
              (epoch + 1, dev_running_loss / len(dev_dataloader)))
        valid_loss.append(dev_running_loss / len(dev_dataloader))

        # test
        for test_batch in test_dataloader:
            test_inputs = torch.cat([test_batch['bnf'], test_batch['f0']], dim=2).to(device)
            test_outputs = model(test_inputs).transpose(0, 1)
            test_target_mels = test_batch['mel'].transpose(0, 1)

            draw_melspectrograms(
                args.test_dir, step=epoch, mel_batch=test_outputs.cpu().detach().numpy(),
                mel_lengths=test_batch['length'].numpy(), ids=test_batch['fid'],
                prefix='predicted')
            draw_melspectrograms(
                args.test_dir, step=epoch, mel_batch=test_target_mels.numpy(),
                mel_lengths=test_batch['length'].numpy(), ids=test_batch['fid'],
                prefix='groundtruth')
            break  # only test one batch of data
    train_loss = np.array(train_loss)
    valid_loss = np.array(valid_loss)
    if args.use_res:
        np.save('./one_res_train_loss.npy', train_loss)
        np.save('./one_res_valid_loss.npy', valid_loss)
    else:
        np.save('./one_train_loss.npy', train_loss)
        np.save('./one_valid_loss.npy', valid_loss)


if __name__ == '__main__':
    main()
