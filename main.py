import random
import torch
import argparse
import os
import time

from federated import *
from lib import get_FLdata
from lib.logger import get_logger
from lib.client import Client
from lib.server import Server

args = argparse.ArgumentParser(description='arguments')
# -------------------------------data-------------------------------------------#
args.add_argument('--dataset', type=str, choices=["PEMS03", "PEMS04", "PEMS08", "PEMS07"],
                    default='PEMS08', help='which dataset to run')
args.add_argument('--num_nodes', type=int, default=170, help='num_nodes')
args.add_argument('--seq_len', type=int, default=12, help='input sequence length')
args.add_argument('--horizon', type=int, default=12, help='output sequence length')
# -------------------------------model------------------------------------------#
args.add_argument('--input_dim', type=int, default=1, help='number of input channel')
args.add_argument('--output_dim', type=int, default=1, help='number of output channel')
args.add_argument('--num_rnn_layers', type=int, default=1, help='number of rnn layers')
args.add_argument('--rnn_units', type=int, default=64, help='number of rnn units')
args.add_argument('--pattern_num', type=int, default=20, help='number of meta-nodes/prototypes')
args.add_argument('--pattern_dim', type=int, default=64, help='dimension of meta-nodes/prototypes')
args.add_argument('--cheb_k', type=int, default=3, help='max diffusion step or Cheb K')
# -------------------------------train------------------------------------------#
args.add_argument("--loss", type=str, default='mask_mae_loss', help="mask_mae_loss")
args.add_argument('--lamb', type=float, default=0.01, help='lamb value for separate loss')
args.add_argument('--lamb1', type=float, default=0.01, help='lamb1 value for compact loss')
args.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
args.add_argument("--patience", type=int, default=20, help="patience used for early stop")
args.add_argument("--batch_size", type=int, default=128, help="size of the batches")
args.add_argument("--lr", type=float, default=0.01, help="base learning rate")
args.add_argument("--steps", type=eval, default=[50, 100], help="steps")
args.add_argument("--lr_decay_ratio", type=float, default=0.1, help="lr_decay_ratio")
args.add_argument("--epsilon", type=float, default=1e-3, help="optimizer epsilon")
args.add_argument("--max_grad_norm", type=int, default=5, help="max_grad_norm")
args.add_argument("--use_curriculum_learning", type=eval, choices=[True, False], default='True',
                    help="use_curriculum_learning")
args.add_argument("--cl_decay_steps", type=int, default=2000, help="cl_decay_steps")
args.add_argument('--test_every_n_epochs', type=int, default=5, help='test_every_n_epochs')
# -------------------------------static------------------------------------------#
args.add_argument('--gpu', type=int, default=0, help='which gpu to use')
args.add_argument('--seed', type=int, default=100, help='random seed.')
args.add_argument('--mode', type=str, default="fedavg")
args.add_argument('--num_client', type=int, default=4, help="number of clients")
args = args.parse_args()

if args.dataset == 'PEMS03':
    args.num_nodes = 358
elif args.dataset == 'PEMS04':
    args.num_nodes = 307
elif args.dataset == 'PEMS08':
    args.num_nodes = 170
elif args.dataset == 'PEMS07':
    args.num_nodes = 883
else:
    pass


# -------------------------------set logger-----------------------------------------#
timestring = time.strftime('%Y%m%d%H%M%S', time.localtime())
args.path = f'./save/{args.dataset}/{str(args.num_client)}/{args.mode}_{timestring}'
if not os.path.exists(args.path): os.makedirs(args.path)

cpu_num = 4
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)
device = torch.device("cuda:{}".format(args.gpu)) if torch.cuda.is_available() else torch.device("cpu")
# -------------------------------set seed-----------------------------------------#
random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

def main():
    logger = get_logger(args=args)
    logger.info(args)

    server = Server(args, device)

    clients_list = []
    clients_data = get_FLdata.graph_partition(args)
    for idx, data in enumerate(clients_data):
        client = Client(data, idx, args, logger)
        clients_list.append(client)

    if args.mode == "selftrain":
        run_selftrain(logger , clients_list, server, COMMUNICATION_ROUNDS=args.epochs)
    if args.mode == "fedavg":
        run_fedavg(logger , clients_list, server, COMMUNICATION_ROUNDS=args.epochs, local_epoch=1)
    if args.mode == "fedtps":
        run_fedtps(logger , clients_list, server, COMMUNICATION_ROUNDS=args.epochs, local_epoch=1)


if __name__ == '__main__':
    main()
