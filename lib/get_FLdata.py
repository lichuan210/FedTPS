import pickle
import numpy as np
import networkx as nx
import torch
import matplotlib.pyplot as plt
import os
import metispy as metis

class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True, shuffle=False):
        """

        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        if shuffle:
            permutation = np.random.permutation(self.size)
            xs, ys = xs[permutation], ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def graph_partition(args):
    num_clients = args.num_client
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(f'./datasets/{args.dataset}', category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']

    # load static graph
    file_path = f'./datasets/{args.dataset}/adj_mx.pkl'
    try:
        _, _, adj_matrix = load_pickle(file_path)
    except:
        adj_matrix = load_pickle(file_path)
    adj_matrix = adj_matrix - np.identity(adj_matrix.shape[0])
    G = nx.from_numpy_array(adj_matrix)

    # graph partition
    try:
        part = torch.load(f'./datasets/{args.dataset}/partition_clients_{str(num_clients)}.pt')
    except:
        if num_clients == 1 :
            part = [0] * adj_matrix.shape[0]
        else:
            n_cuts, part = metis.part_graph(G, num_clients)
        torch.save(part, f'./datasets/{args.dataset}/partition_clients_{str(num_clients)}.pt')

    # load data for clients
    clients_data = []
    for client_id in range(num_clients):
        nodes = np.where(np.array(part) == client_id)[0]
        nodes = list(nodes)
        client_adj = adj_matrix[nodes][:, nodes]
        scaler = StandardScaler(mean=data["x_train"][:, :, nodes, :].mean(), std=data["x_train"][:, :, nodes, :].std())

        for category in ['train', 'val', 'test']:
            data['x_' + category][:, :, nodes, :] = scaler.transform(data['x_' + category][:, :, nodes, :])
            data['y_' + category][:, :, nodes, :] = scaler.transform(data['y_' + category][:, :, nodes, :])


        train_loader = DataLoader(data["x_train"][:, :, nodes, :], data["y_train"][:, :, nodes, :], args.batch_size, shuffle=True)
        val_loader = DataLoader(data["x_val"][:, :, nodes, :], data["y_val"][:, :, nodes, :], args.batch_size, shuffle=False)
        test_loader = DataLoader(data["x_test"][:, :, nodes, :], data["y_test"][:, :, nodes, :], args.batch_size, shuffle=False)

        clients_data.append({
            "adj":client_adj,
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader,
            "scaler":scaler
        })

    return clients_data




