import torch
import random

from lib.models import DCRNN,DCRNN_TP

class Server():
    def __init__(self, args, device):
        if args.mode == "fedtps":
            self.model = DCRNN_TP(num_nodes=args.num_nodes, input_dim=args.input_dim, output_dim=args.output_dim,
                                  horizon=args.horizon,
                                  rnn_units=args.rnn_units, num_layers=args.num_rnn_layers, pattern_num=args.pattern_num,
                                  pattern_dim=args.pattern_dim,
                                  cheb_k=args.cheb_k, cl_decay_steps=args.cl_decay_steps,
                                  use_curriculum_learning=args.use_curriculum_learning,
                                  ycov_dim=1).cuda().to(device)
        else:
            self.model = DCRNN(num_nodes=args.num_nodes, input_dim=args.input_dim, output_dim=args.output_dim,
                               horizon=args.horizon,
                               rnn_units=args.rnn_units, num_layers=args.num_rnn_layers,
                               cheb_k=args.cheb_k, cl_decay_steps=args.cl_decay_steps,
                               use_curriculum_learning=args.use_curriculum_learning,
                               ycov_dim=1).cuda().to(device)
        self.W = {key: value for key, value in self.model.named_parameters()}
        self.device = device

    def randomSample_clients(self, all_clients, frac):
        return random.sample(all_clients, int(len(all_clients) * frac))

    def aggregate_weights(self, selected_clients):
        # pass train_size, and weighted aggregate
        total_size = 0
        for client in selected_clients:
            total_size += client.num_nodes
        for k in self.W.keys():
            if k == "We1" or k == "We2":
                continue
            self.W[k].data = torch.div(torch.sum(torch.stack([torch.mul(client.W[k].data, client.num_nodes) for client in selected_clients]), dim=0), total_size).clone()


    def aggregate_bank_based_on_topk(self,selected_clients):
        total_size = 0
        for client in selected_clients:
            total_size += client.num_nodes

        cos_sim = [[None for j in range(len(selected_clients))] for i in range(len(selected_clients))]
        for i in range(len(selected_clients)):
            for j in range(len(selected_clients)):
                cos_sim[i][j] = torch.nn.functional.cosine_similarity(selected_clients[i].W["Patterns"].data[:, :, None],
                                                                      selected_clients[j].W["Patterns"].data.t()[None, :, :])
        cos_sim = torch.tensor([[data.cpu().detach().numpy() for data in item] for item in cos_sim])
        cos_sim = cos_sim.transpose(1, 2)
        cos_sim = torch.exp(10 * cos_sim)
        max_index = torch.argmax(cos_sim, dim=-1)
        max_values = cos_sim.max(dim=-1).values
        row_sums = max_values.sum(dim=-1).unsqueeze(-1)
        max_sim = max_values / row_sums

        clients_banks = []
        for i in range(max_sim.size(0)):
            bank = None
            for j in range(max_sim.size(1)):
                item = torch.div(torch.sum(torch.stack([torch.mul(selected_clients[m].W["Patterns"].data[max_index[i][j][m]] , selected_clients[m].num_nodes) for m in range(max_sim.size(2))]),dim=0),total_size).unsqueeze(dim=0)
                if bank is None:
                    bank = item
                else:
                    bank = torch.cat((bank,item))
            clients_banks.append(bank)

        for i in range(len(selected_clients)):
            selected_clients[i].W["Patterns"].data = clients_banks[i].clone()


    def aggregate_bank_based_on_sim(self,selected_clients):
        cos_sim = [[None for j in range(len(selected_clients))] for i in range(len(selected_clients))]
        for i in range(len(selected_clients)):
            for j in range(len(selected_clients)):
                cos_sim[i][j] = torch.nn.functional.cosine_similarity(selected_clients[i].W["Patterns"].data[:, :, None],
                                                                      selected_clients[j].W["Patterns"].data.t()[None, :, :])
        cos_sim = torch.tensor([[data.cpu().detach().numpy() for data in item] for item in cos_sim])
        cos_sim = cos_sim.transpose(1, 2)
        cos_sim = torch.exp(10 * cos_sim)
        max_index = torch.argmax(cos_sim, dim=-1)
        max_values = cos_sim.max(dim=-1).values
        row_sums = max_values.sum(dim=-1).unsqueeze(-1)
        max_sim = max_values / row_sums


        clients_banks = []
        for i in range(max_sim.size(0)):
            bank = None
            for j in range(max_sim.size(1)):
                item = torch.sum(torch.stack([torch.mul(selected_clients[m].W["Patterns"].data[max_index[i][j][m]] , max_sim[i][j][m]) for m in range(max_sim.size(2))]),dim=0).unsqueeze(dim=0)
                if bank is None:
                    bank = item
                else:
                    bank = torch.cat((bank,item))
            clients_banks.append(bank)

        for i in range(len(selected_clients)):
            selected_clients[i].W["Patterns"].data = clients_banks[i].clone()
