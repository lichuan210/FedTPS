import re
from collections import defaultdict
import numpy as np


def parse_log_file(file_path):
    pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}): (Client\d+)  "
        r"Local Epoch \[(\d+)/(\d+)\] \((\d+)\) train_loss: ([\d.]+),  lr: ([\d.]+), ([\d.]+)s    "
        r"Val MAE: ([\d.]+), Val MAPE: ([\d.]+) , Val RMSE: ([\d.]+); "
        r"Test MAE: ([\d.]+), Test MAPE: ([\d.]+) , Test RMSE: ([\d.]+)"
    )

    grouped_data = defaultdict(list)

    with open(file_path, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                timestamp, client, epoch_current, epoch_total, step, train_loss, lr, time, \
                val_mae, val_mape, val_rmse, test_mae, test_mape, test_rmse = match.groups()

                entry = {
                    "timestamp": timestamp,
                    "client": client,
                    "epoch_current": int(epoch_current),
                    "epoch_total": int(epoch_total),
                    "step": int(step),
                    "train_loss": float(train_loss),
                    "lr": float(lr),
                    "time": float(time),
                    "val_mae": float(val_mae),
                    "val_mape": float(val_mape),
                    "val_rmse": float(val_rmse),
                    "test_mae": float(test_mae),
                    "test_mape": float(test_mape),
                    "test_rmse": float(test_rmse),
                }
                grouped_data[client].append(entry)

    return grouped_data

def find_min_val_mae_entries(grouped_data):
    min_val_mae_entries = {}

    for client, entries in grouped_data.items():
        min_entry = min(entries, key=lambda x: x["val_mae"])
        min_val_mae_entries[client] = min_entry

    return min_val_mae_entries

# example
file_path = "./save/PEMS03/4/selftrain/PEMS07_client_num_4_selftrain.log"
grouped_data = parse_log_file(file_path)
min_val_mae_entries = find_min_val_mae_entries(grouped_data)

mean_test_mae = []
mean_test_mape = []
mean_test_rmse = []

for client, entry in min_val_mae_entries.items():
    mean_test_mae.append(entry["test_mae"])
    mean_test_mape.append(entry["test_mape"])
    mean_test_rmse.append(entry["test_rmse"])
    print(f"Client: {client}")
    print(f"Minimum Val MAE Entry: {entry}")
    print("-" * 50)

mean_test_mae = np.mean(mean_test_mae,axis=0)
mean_test_mape = np.mean(mean_test_mape,axis=0)
mean_test_rmse = np.mean(mean_test_rmse,axis=0)
print("Mean Test MAE: ", mean_test_mae)
print("Mean Test MAPE: ", mean_test_mape)
print("Mean Test RMSE: ", mean_test_rmse)
