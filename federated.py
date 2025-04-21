import numpy as np

def run_selftrain(logger , clients, server, COMMUNICATION_ROUNDS):
    # all clients are initialized with the same weights
    for client in clients:
        client.download_from_server(server)

    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        logger.info('communication round'+str(c_round)+"/"+str(COMMUNICATION_ROUNDS))
        for client in clients:
            client.local_train(epochs=1)


def run_fedavg(logger , clients, server, COMMUNICATION_ROUNDS, local_epoch, samp=None, frac=1.0):
    for client in clients:
        client.download_from_server(server)

    if samp is None:
        sampling_fn = server.randomSample_clients
        frac = 1.0

    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        logger.info(f"  > round {c_round}")
        if c_round == 1:
            selected_clients = clients
        else:
            selected_clients = sampling_fn(clients, frac)

        for client in selected_clients:
            client.local_train(local_epoch)

        server.aggregate_weights(selected_clients)
        for client in selected_clients:
            client.download_from_server(server)


def run_fedtps(logger , clients, server, COMMUNICATION_ROUNDS, local_epoch, samp=None, frac=1.0):
    for client in clients:
        client.download_from_server(server)

    if samp is None:
        sampling_fn = server.randomSample_clients
        frac = 1.0

    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        # if (c_round) % 50 == 0:
        logger.info(f"  > round {c_round}")
        if c_round == 1:
            selected_clients = clients
        else:
            selected_clients = sampling_fn(clients, frac)

        for client in selected_clients:
            client.local_train_sim(local_epoch)

        server.aggregate_bank_based_on_sim(selected_clients)
