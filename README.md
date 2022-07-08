# Federated Learning Research Task

For this task of simulating the federated learning using [Flower](https://flower.dev/) in Python. I have used the FEMNIST Dataset from [LEAF CMU](https://leaf.cmu.edu/).

I first preprocess the data by downloading the data from the LEAF CMU's Github repo and run the following command:
```
git clone https://github.com/TalwalkarLab/leaf.git
cd leaf/data/femnist
./preprocess.sh -s niid --sf 0.05 -k 0 -t sample
```

The above commands will download the `data` in the data directory in `leaf/data/femnist`. Then update the path variable in the `preprocess.py` script to the train directory from the above data directory. Generally, the path should be `./leaf/data/femnist/data/train`.

After setting the path to the train directory run the preprocess.py file to generate the `FEMNIST.csv` file:
```
python3 preprocess.py
```

Now, we should have the `FEMNIST.csv` file and it's time to run the server and client to simulate the Federated Learning using the [Flower](https://flower.dev/) library and the preprocessed image data with 62 classes.

First, run the server as follows:
```
python3 server.py
```

Then, run atleast 2 clients as follows:
```
python3 client.py
```
