import copy
from lib.loss import masked_mae, masked_rmse, masked_mape, masked_mse
import numpy as np
import time
import torch
import torch.nn as nn
from lib.models import DCRNN,DCRNN_TP


class Client(): 
    def __init__(self,client_data, client_id, args, logger):
        self.logger = logger
        self.device = torch.device("cuda:{}".format(args.gpu)) if torch.cuda.is_available() else torch.device("cpu")
        self.args = args
        self.client_data = client_data
        self.client_id = client_id
        self.num_nodes = client_data["adj"].shape[1]

        if args.mode == "fedtps" :
            self.model = DCRNN_TP(num_nodes=self.num_nodes, input_dim=args.input_dim, output_dim=args.output_dim,
                             horizon=args.horizon,
                             rnn_units=args.rnn_units, num_layers=args.num_rnn_layers, pattern_num=args.pattern_num,
                             pattern_dim=args.pattern_dim,
                             cheb_k=args.cheb_k, cl_decay_steps=args.cl_decay_steps,
                             use_curriculum_learning=args.use_curriculum_learning, ycov_dim=1).cuda()
        else:
            self.model = DCRNN(num_nodes=self.num_nodes, input_dim=args.input_dim, output_dim=args.output_dim,
                             horizon=args.horizon,
                             rnn_units=args.rnn_units, num_layers=args.num_rnn_layers,
                             cheb_k=args.cheb_k, cl_decay_steps=args.cl_decay_steps,
                             use_curriculum_learning=args.use_curriculum_learning, ycov_dim=1).cuda()


        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, eps=args.epsilon)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.steps,
                                                                 gamma=args.lr_decay_ratio)

        self.W = {key: value for key, value in self.model.named_parameters()}
        self.dW = {key: torch.zeros_like(value) for key, value in self.model.named_parameters()}



    def local_train_sim(self, epochs):
        batches_seen = 0
        for epoch_num in range(epochs):
            start_time = time.time()
            self.model.train()
            data_iter = self.client_data['train_loader'].get_iterator()
            losses = []
            for x, y in data_iter:
                self.optimizer.zero_grad()
                x, y, ycov = self.prepare_x_y(x, y)
                output, h_att, query, pos, neg = self.model(x, ycov, y, batches_seen)
                y_pred = self.client_data["scaler"].inverse_transform(output)
                y_true = self.client_data["scaler"].inverse_transform(y)
                loss1 = masked_mae(y_pred, y_true)
                separate_loss = nn.TripletMarginLoss(margin=1.0)
                compact_loss = nn.MSELoss()
                loss2 = separate_loss(query, pos.detach(), neg.detach())
                loss3 = compact_loss(query, pos.detach())
                loss = loss1 + self.args.lamb * loss2 + self.args.lamb1 * loss3
                losses.append(loss.item())
                batches_seen += 1
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
            train_loss = np.mean(losses)
            self.lr_scheduler.step()

            end_time = time.time()
            message1 = '  Local Epoch [{}/{}] ({}) train_loss: {:.4f},  lr: {:.6f}, {:.1f}s'.format(
                epoch_num + 1,
                epochs, batches_seen, train_loss, self.optimizer.param_groups[0]['lr'],
                (end_time - start_time))

            val_loss, _, _, val_result = self.evaluate_sim('val')
            test_loss, _, _, test_result = self.evaluate_sim('test')
            message2 = "Val MAE: {:.4f}, Val MAPE: {:.4f} , Val RMSE: {:.4f}; Test MAE: {:.4f}, Test MAPE: {:.4f} , Test RMSE: {:.4f}".format(
                val_result["mae"],val_result["mape"],val_result["rmse"],test_result["mae"],test_result["mape"],test_result["rmse"],
            )

            self.logger.info("Client" + str(self.client_id) + message1 + "    " + message2)


    def evaluate_sim(self, mode):
        with torch.no_grad():
            self.model.eval()
            data_iter = self.client_data[f'{mode}_loader'].get_iterator()
            losses, ys_true, ys_pred = [], [], []
            for x, y in data_iter:
                x, y, ycov = self.prepare_x_y(x, y)
                output, h_att, query, pos, neg = self.model(x, ycov)
                y_pred = self.client_data["scaler"].inverse_transform(output)
                y_true = self.client_data["scaler"].inverse_transform(y)
                loss1 = masked_mae(y_pred, y_true)  # masked_mae_loss(y_pred, y_true)
                separate_loss = nn.TripletMarginLoss(margin=1.0)
                compact_loss = nn.MSELoss()
                loss2 = separate_loss(query, pos.detach(), neg.detach())
                loss3 = compact_loss(query, pos.detach())
                loss = loss1 + self.args.lamb * loss2 + self.args.lamb1 * loss3
                losses.append(loss.item())
                ys_true.append(y_true)
                ys_pred.append(y_pred)
            mean_loss = np.mean(losses)
            ys_true, ys_pred = torch.cat(ys_true, dim=0), torch.cat(ys_pred, dim=0)

            ys_true, ys_pred = ys_true.permute(1, 0, 2, 3), ys_pred.permute(1, 0, 2, 3)
            mae = masked_mae(ys_pred, ys_true).item()
            mape = masked_mape(ys_pred, ys_true).item()
            rmse = masked_rmse(ys_pred, ys_true).item()
            mae_3 = masked_mae(ys_pred[2:3], ys_true[2:3]).item()
            mape_3 = masked_mape(ys_pred[2:3], ys_true[2:3]).item()
            rmse_3 = masked_rmse(ys_pred[2:3], ys_true[2:3]).item()
            mae_6 = masked_mae(ys_pred[5:6], ys_true[5:6]).item()
            mape_6 = masked_mape(ys_pred[5:6], ys_true[5:6]).item()
            rmse_6 = masked_rmse(ys_pred[5:6], ys_true[5:6]).item()
            mae_12 = masked_mae(ys_pred[11:12], ys_true[11:12]).item()
            mape_12 = masked_mape(ys_pred[11:12], ys_true[11:12]).item()
            rmse_12 = masked_rmse(ys_pred[11:12], ys_true[11:12]).item()
            ys_true, ys_pred = ys_true.permute(1, 0, 2, 3), ys_pred.permute(1, 0, 2, 3)

            result = {"mae": mae,
                      "mape": mape,
                      "rmse": rmse,
                      "mae_3": mae_3,
                      "mape_3": mape_3,
                      "rmse_3": rmse_3,
                      "mae_6": mae_6,
                      "mape_6": mape_6,
                      "rmse_6": rmse_6,
                      "mae_12": mae_12,
                      "mape_12": mape_12,
                      "rmse_12": rmse_12}

            return mean_loss, ys_true, ys_pred, result

    def local_train(self, epochs):
        batches_seen = 0
        for epoch_num in range(epochs):
            start_time = time.time()
            self.model.train()
            data_iter = self.client_data['train_loader'].get_iterator()
            losses = []
            for x, y in data_iter:
                self.optimizer.zero_grad()
                x, y, ycov = self.prepare_x_y(x, y)
                output = self.model(x, ycov, y, batches_seen)
                y_pred = self.client_data["scaler"].inverse_transform(output)
                y_true = self.client_data["scaler"].inverse_transform(y)
                loss = masked_mae(y_pred, y_true)
                losses.append(loss.item())
                batches_seen += 1
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
            train_loss = np.mean(losses)
            self.lr_scheduler.step()

            end_time = time.time()
            message1 = '  Local Epoch [{}/{}] ({}) train_loss: {:.4f},  lr: {:.6f}, {:.1f}s'.format(
                epoch_num + 1,
                epochs, batches_seen, train_loss, self.optimizer.param_groups[0]['lr'],
                (end_time - start_time))

            val_loss, _, _, val_result = self.evaluate('val')
            test_loss, _, _, test_result = self.evaluate('test')
            message2 = "Val MAE: {:.4f}, Val MAPE: {:.4f} , Val RMSE: {:.4f}; Test MAE: {:.4f}, Test MAPE: {:.4f} , Test RMSE: {:.4f}".format(
                val_result["mae"], val_result["mape"], val_result["rmse"], test_result["mae"], test_result["mape"],
                test_result["rmse"],
            )

            self.logger.info("Client" + str(self.client_id) + message1 + "    " + message2)


    def evaluate(self, mode):
        with torch.no_grad():
            self.model.eval()
            data_iter = self.client_data[f'{mode}_loader'].get_iterator()
            losses, ys_true, ys_pred = [], [], []
            for x, y in data_iter:
                x, y, ycov = self.prepare_x_y(x, y)
                output = self.model(x, ycov)
                y_pred = self.client_data["scaler"].inverse_transform(output)
                y_true = self.client_data["scaler"].inverse_transform(y)
                loss = masked_mae(y_pred, y_true)
                losses.append(loss.item())
                ys_true.append(y_true)
                ys_pred.append(y_pred)
            mean_loss = np.mean(losses)
            ys_true, ys_pred = torch.cat(ys_true, dim=0), torch.cat(ys_pred, dim=0)

            ys_true, ys_pred = ys_true.permute(1, 0, 2, 3), ys_pred.permute(1, 0, 2, 3)
            mae = masked_mae(ys_pred, ys_true).item()
            mape = masked_mape(ys_pred, ys_true).item()
            rmse = masked_rmse(ys_pred, ys_true).item()
            mae_3 = masked_mae(ys_pred[2:3], ys_true[2:3]).item()
            mape_3 = masked_mape(ys_pred[2:3], ys_true[2:3]).item()
            rmse_3 = masked_rmse(ys_pred[2:3], ys_true[2:3]).item()
            mae_6 = masked_mae(ys_pred[5:6], ys_true[5:6]).item()
            mape_6 = masked_mape(ys_pred[5:6], ys_true[5:6]).item()
            rmse_6 = masked_rmse(ys_pred[5:6], ys_true[5:6]).item()
            mae_12 = masked_mae(ys_pred[11:12], ys_true[11:12]).item()
            mape_12 = masked_mape(ys_pred[11:12], ys_true[11:12]).item()
            rmse_12 = masked_rmse(ys_pred[11:12], ys_true[11:12]).item()
            ys_true, ys_pred = ys_true.permute(1, 0, 2, 3), ys_pred.permute(1, 0, 2, 3)

            result = {"mae": mae,
                      "mape": mape,
                      "rmse": rmse,
                      "mae_3": mae_3,
                      "mape_3": mape_3,
                      "rmse_3": rmse_3,
                      "mae_6": mae_6,
                      "mape_6": mape_6,
                      "rmse_6": rmse_6,
                      "mae_12": mae_12,
                      "mape_12": mape_12,
                      "rmse_12": rmse_12}

            return mean_loss, ys_true, ys_pred, result



    def prepare_x_y(self, x, y):
        """
        :param x: shape (batch_size, seq_len, num_sensor, input_dim)
        :param y: shape (batch_size, horizon, num_sensor, input_dim)
        :return1: x shape (seq_len, batch_size, num_sensor, input_dim)
                  y shape (horizon, batch_size, num_sensor, input_dim)
        :return2: x: shape (seq_len, batch_size, num_sensor * input_dim)
                  y: shape (horizon, batch_size, num_sensor * output_dim)
        """
        x0 = x[..., :self.args.input_dim]
        y0 = y[..., :self.args.output_dim]
        y1 = y[..., self.args.output_dim:]
        x0 = torch.from_numpy(x0).float()
        y0 = torch.from_numpy(y0).float()
        y1 = torch.from_numpy(y1).float()
        if y1.shape[-1] == 0:
            y1 = torch.zeros(y0.shape)
        return x0.to(self.device), y0.to(self.device), y1.to(self.device)  # x, y, y_cov

    #Federal Learning (FedAvg)
    def download_from_server(self,server):
        total_size_bytes = 0
        for k in server.W:
            if k == "We1" or k == "We2":
                continue
            self.W[k].data = server.W[k].data.clone()
