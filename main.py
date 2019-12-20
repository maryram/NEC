import argparse
import json
import os
import sys
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import DetectModel
from data import SeqDataset, collate_fn
from math import log
import numpy as np
from sklearn import preprocessing
import json


base_path = '/media/external_3TB/3TB/rafie/'


def predict(args, config):

    global base_path

    #    path = os.path.dirname(os.path.abspath(__file__))
    #    print(path , args.logdir , args.test)
    checkpoint = torch.load(os.path.join(base_path + 'model-outputs/', args.logdir, args.test))

    data = [json.loads(d) for d in open(base_path + 'model-inputs/' + args.input, "rt").readlines()]
    dataset = SeqDataset(data)

    device = torch.device("cuda:{}".format(args.cuda) if args.cuda else "cpu")

    model = DetectModel(input_size=config['model']['input_size'], hidden_size=config['model']['hidden_size'],
                        rnn_layers=config['model']['rnn_layers'],
                        out_channels=config['model']['out_channels'], height=config['model']['height'],
                        cnn_layers=config['model']['cnn_layers'],
                        linear_hidden_size=config['model']['linear_hidden_size'],
                        linear_layers=config['model']['linear_layers'], output_size=config['model']['output_size'])
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
    h0 = torch.zeros(config['model']['rnn_layers'], args.batch_size, config['model']['hidden_size']).to(device)
    
    with open('data/'+ args.logdir.split('_')[0] +'-special-cascades.json', 'r') as f:
        special_cascades = json.load(f)

    true_acc = []
    false_acc = []
    pred_lens = []
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    all_len = []
    hiddens = []
    all_labels = []
    loss = []
    stopping_lens={}

    new_pred_lens = []

    with tqdm(total=len(dataset), desc="Sequences", leave=False) as pbar:
        for step, (seq_data, labels) in enumerate(dataloader):
            pbar.update(args.batch_size)

            sequences = []
            lens = []
            real_lens = []
            eids = []

            for seq in seq_data:
                sequences.append(seq['seq'])
                l = seq['len']
                real_lens.append(l)
                lens.append(l if l<=100 else 100)
                eids.append(seq['eid'])
        
            lens = np.array(lens)
            real_lens = np.array(real_lens)
            sequences = np.array(sequences)
            labels = np.array(labels)
            eids = np.array(eids)
            all_len.extend(real_lens)

            #### sort by lengths

            reverse_idx = np.argsort(-lens)

            sorted_length = lens[reverse_idx]
            sorted_real_lengths = real_lens[reverse_idx]
            sorted_sequnces = sequences[reverse_idx]
            sorted_labels = labels[reverse_idx]
            sorted_eids = eids[reverse_idx]
            sorted_length[0] = 100

            sequences = torch.tensor(sorted_sequnces, dtype=torch.float, requires_grad=False).to(device)
            labels = torch.tensor(sorted_labels, dtype=torch.long, requires_grad=False).to(device)

            sequences = preprocessing.normalize(sequences.view(args.batch_size * 100, config['model']['input_size']),
                                                norm='l2')
            sequences = torch.tensor(sequences, dtype=torch.float, requires_grad=False)
            sequences = sequences.view(args.batch_size, 100, config['model']['input_size'])

            # print(sorted_length)


            outs = []
            output, hidden = model(sequences, sorted_length, h0)
            hiddens.extend(hidden)
            all_labels.extend(sorted_labels)
            
            # print loss
            
            for i in range(output.shape[0]):
                loss_i = loss2(output[i, :, :], sorted_length[i], labels[i], 0.8, 0.3, 0.3)
                loss.append(loss_i)

            outs, pred_lens,real_pred_lens = stopping_rule(output, sorted_real_lengths)
            for i,pred_len in enumerate(pred_lens):
                # if str(sorted_eids[i]) in special_cascades['very-special-cascades']:
                    stopping_lens[str(sorted_eids[i])] = list([pred_lens[i],int(real_pred_lens[i])])

            i = -1

            for o, t in zip(outs, labels.tolist()):
                i += 1
                if sorted_real_lengths[i] > 0 :
                    new_pred_lens.append(pred_lens[i])
                    o = o.tolist()
                    if t == 1:
                        if o == 1:
                            tp += 1
                            stopping_lens[str(sorted_eids[i])].append('tp')
                        else:
                            fn += 1
                            stopping_lens[str(sorted_eids[i])].append('fn')
                    elif t == 0:
                        if o == 1:
                            fp += 1
                            stopping_lens[str(sorted_eids[i])].append('fp')
                        elif o == 0:
                            tn += 1
                            stopping_lens[str(sorted_eids[i])].append('tn')

    # print('number of actual fakes: ', tp + fn, '  number of actual reals: ', tn + fp)

    acc = (tp + tn) / (tp + tn + fn + fp)

    recall_f = -1
    recall_r = -1
    precision_f = -1
    precision_f = -1

    if tp + fn > 0:
        recall_f = (tp) / (tp + fn)
    if tn + fp > 0:
        recall_r = (tn) / (tn + fp)
    if tp + fp > 0:
        precision_f = (tp) / (tp + fp)
    if tn + fn > 0:
        precision_r = (tn) / (tn + fn)

    all_len = np.array(all_len)

#    print(len(hiddens))
#    with open('hiddens_weibo' , 'w') as file:
#        for h,l in zip(hiddens,all_labels):
#            l = 0 if l ==0 else 1
#            json_data = json.dumps([h.detach().numpy().tolist() ,l])
#            file.write(json_data+'\n')

    loss = torch.stack(loss)
    
    loss = loss.mean()
#    print(pred_lens)

    # print('Loss: ', loss.detach().numpy())
    # print(new_pred_lens)
    # print('Prediction length array size: ', len(new_pred_lens))
    # print('Average Prediction Length: ', np.mean(new_pred_lens), 'Variance of Prediction Lengths: ', np.var(new_pred_lens), '\n\n')
    print('Accuracy: ', acc)
    print('Recall_r: ', recall_r, 'Precision_r: ', precision_r, 'F_r: ', 2/(1/recall_r + 1/precision_r))

    print('Recall_f: ',recall_f, 'Precision_f: ',precision_f, 'F_f: ',2/(1/recall_f + 1/precision_f) if recall_f > 0 and precision_f > 0 else -1)
    print(tp, fp, tn, fn, loss.detach().numpy(), np.mean(new_pred_lens), np.var(new_pred_lens))

    with open(args.logdir+'.json','w') as pred_f:
        json.dump(stopping_lens,pred_f)



def train(args: argparse.Namespace, config: dict):

    global base_path

    data = [json.loads(d) for d in open(base_path + 'model-inputs/' + args.input, "rt").readlines()]
    dataset = SeqDataset(data)

    device = torch.device("cuda:{}".format(args.cuda) if args.cuda else "cpu")

    model = DetectModel(input_size=config['model']['input_size'], hidden_size=config['model']['hidden_size'],
                        rnn_layers=config['model']['rnn_layers'],
                        out_channels=config['model']['out_channels'], height=config['model']['height'],
                        cnn_layers=config['model']['cnn_layers'],
                        linear_hidden_size=config['model']['linear_hidden_size'],
                        linear_layers=config['model']['linear_layers'], output_size=config['model']['output_size'])
    model = model.to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
    h0 = torch.zeros(config['model']['rnn_layers'], args.batch_size, config['model']['hidden_size']).to(device)

    for epoch in tqdm(range(args.epoch), desc="Epochs"):
        with tqdm(total=len(dataset), desc="Sequences", leave=False) as pbar:
            for step, (seq_data, labels) in enumerate(dataloader):
                pbar.update(args.batch_size)

                model.zero_grad()
                h0.data.zero_()

                sequences = []
                lens = []

                for seq in seq_data:
                    sequences.append(seq['seq'])
                    # lens.append(seq['len'])
                    l = seq['len']
                    lens.append(l if l <= 100 else 100)

                lens = np.array(lens)
                sequences = np.array(sequences)
                labels = np.array(labels)

                ####sort by lengths
                reverse_idx = np.argsort(-lens)

                sorted_length = lens[reverse_idx]  # for descending order
                sorted_sequnces = sequences[reverse_idx]
                sorted_labels = labels[reverse_idx]
                sorted_length[0] = 100

                sequences = torch.tensor(sorted_sequnces, dtype=torch.float, requires_grad=False).to(device)
                labels = torch.tensor(sorted_labels, dtype=torch.long, requires_grad=False).to(device)

                sequences = preprocessing.normalize(
                    sequences.view(args.batch_size * 100, config['model']['input_size']), norm='l2')
                sequences = torch.tensor(sequences, dtype=torch.float, requires_grad=False)
                sequences = sequences.view(args.batch_size, 100, config['model']['input_size'])

                output, _ = model(sequences, sorted_length, h0)
                alpha = 0.8
                landa0 = 0.3
                landa1 = 0.3
                loss = []
                for i in range(output.shape[0]):
                    loss_i = loss2(output[i, :, :], sorted_length[i], labels[i], alpha, landa0, landa1)
                    loss.append(loss_i)
                loss = torch.stack(loss)

                loss = loss.mean()
                loss.backward(torch.ones_like(loss))
                optimizer.step()

                if step % 30 == 0:
                    tqdm.write("Step: {:,} Loss: {:,}".format(step, loss))

            if args.logdir is not None:
                path = os.path.join(base_path + 'model-outputs/', args.logdir)
                if not os.path.exists(path):
                    os.makedirs(path)

                state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                torch.save(state, os.path.join(path, "{}.ckpt".format(epoch + 1)))
                tqdm.write("[+] {}.ckpt saved".format(epoch + 1))


def loss1(outputs, len, label, alpha, landa0, landa1):
    criterion = nn.CrossEntropyLoss()
    beta = 0
    for t in range(len):
        p_label = outputs[t].argmax()
        if outputs[t][0].tolist() >= alpha or outputs[t][1].tolist() >= alpha:
            beta = t
            break

    o_pred = torch.Tensor([0])
    o_diff = torch.Tensor([0])
    n = 0
    for t in range(beta, len):
        n += 1

        o_pred += torch.log(outputs[t][label]) * label.tolist() + (1 - label.tolist()) * torch.log(outputs[t][label])
        o_diff -= torch.max(torch.Tensor([0]),
                            torch.log(torch.Tensor([alpha])) - torch.log(outputs[t][label])) * label.tolist() + (
                              1 - label.tolist()) * torch.max(torch.Tensor([0]),
                                                              torch.log(1 - outputs[t][label]) - torch.log(
                                                                  torch.Tensor([1 - alpha])))
    if n > 0:
        o_pred = o_pred / n
        o_diff = o_diff / n
    if beta > 0:
        loss = o_pred + o_diff * landa0 + torch.log(torch.Tensor([(beta + 1) / len])) * landa1
    else:
        loss = o_pred + o_diff * landa0 + torch.log(torch.Tensor([1])) * landa1
    return -loss


def loss2(outputs, len, label, alpha, landa0, landa1):
    criterion = nn.CrossEntropyLoss()
    beta = 0
    for t in range(len):
        p_label = outputs[t].argmax()
        if outputs[t][0].tolist() >= alpha or outputs[t][1].tolist() >= alpha:
            beta = t
            break

    o_pred = torch.Tensor([0])
    o_diff = torch.Tensor([0])
    n = 0
    for t in range(0, len):
        n += 1
        p = outputs[t][label]
        l = label.tolist()
        o_pred += -log(1-(t)/len) * (torch.log(p) * l + (1 - l) * torch.log(p))
    #     o_diff -= torch.max(torch.Tensor([0]),
    #                         torch.log(torch.Tensor([alpha])) - torch.log(outputs[t][label])) * label.tolist() + (
    #                           1 - label.tolist()) * torch.max(torch.Tensor([0]),
    #                                                           torch.log(1 - outputs[t][label]) - torch.log(
    #                                                               torch.Tensor([1 - alpha])))
    # if n > 0:
        o_pred = o_pred / n
        # o_diff = o_diff / n
    # if beta > 0:
    #     loss = o_pred + o_diff * landa0 + torch.log(torch.Tensor([(beta + 1) / len])) * landa1
    # else:
    #     loss = o_pred + o_diff * landa0 + torch.log(torch.Tensor([1])) * landa1
    return -o_pred


def stopping_rule(output, sorted_length):
    pred_lens = []
    real_pred_lens = []
    outs = []

    for i in range(output.shape[0]):
        if sorted_length[i] == 1:
            pred_lens.append(1)
            real_pred_lens.append(1)
            outs.append(output[i][0].argmax())
        else:
            early = False
            for t in range(1, min(sorted_length[i],100)):
                if output[i, t - 1].argmax() == output[i, t].argmax():
                    p_label = output[i][t].argmax()
                    p_t0 = output[i][t - 1][p_label]
                    p_t1 = output[i][t][p_label]
                    len_ratio = (t + 1) / sorted_length[i]
                    if p_t1 > p_t0 and (p_t1 - p_t0) <= 1 * len_ratio and p_t1 >= 0.7:
                        outs.append(p_label)
                        pred_lens.append((t + 1) / sorted_length[i])
                        real_pred_lens.append(t+1)
                        early = True
                        break

            if not early:
                outs.append(p_label.data)
                pred_lens.append(1)
                real_pred_lens.append(min(sorted_length[i],100))
    return outs, pred_lens,real_pred_lens


if __name__ == '__main__':

    # np.random.seed(0)
    # torch.manual_seed(0)

    #    print('with manual seed 0')

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="config file path", required=True)
    parser.add_argument('--cuda', type=int, default=None, help="GPU number (default: None=CPU)")
    parser.add_argument('--logdir', type=str, help="log directory")
    parser.add_argument('--test', type=str, default=None, help="checkpoint path for test")

    parser.add_argument('--input', type=str, help="input file path", required=True)
    parser.add_argument('--learning-rate', type=float, default=0.2, metavar="0.2", help="learning rate for model")
    parser.add_argument('--batch-size', type=int, default=32, metavar='32', help="batch size for learning")
    parser.add_argument('--epoch', type=int, default=10, metavar="10", help="the number of epochs")

    args = parser.parse_args()
    config_json = json.load(open(args.config, "rt"))

    if args.test:
        if args.logdir is None:
            print("[-] No log directory option")
            sys.exit(1)

        predict(args, config_json)
    else:
        train(args, config_json)
