
import flwr as fl

if __name__ == '__main__':

    # Starts the server and continues the training for 20 rounds.
    fl.server.start_server("[::]:8080", config={"num_rounds": 20})
