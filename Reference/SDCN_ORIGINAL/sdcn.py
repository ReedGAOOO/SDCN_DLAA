from __future__ import print_function, division
import argparse
import random
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
from Reference.SDCN_ORIGINAL.utils import load_data, load_graph
from Reference.SDCN_ORIGINAL.GNN import GNNLayer
from SDCN_ORIGINAL.evaluation import eva
import sys
import os
from datetime import datetime

# torch.cuda.set_device(1)


class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        # Store shapes in a dictionary without printing
        self.layer_shapes = {
            'autoencoder': {
                'Encoder Layer 1': enc_h1.shape,
                'Encoder Layer 2': enc_h2.shape,
                'Encoder Layer 3': enc_h3.shape,
                'Latent Space': z.shape,
                'Decoder Layer 1': dec_h1.shape,
                'Decoder Layer 2': dec_h2.shape,
                'Decoder Layer 3': dec_h3.shape,
                'Output Layer': x_bar.shape
            }
        }

        return x_bar, enc_h1, enc_h2, enc_h3, z


class SDCN(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, 
                n_input, n_z, n_clusters, v=1):
        super(SDCN, self).__init__()
        # Initialize epoch tracking variables
        self.current_epoch = 0
        self.last_logged_epoch = -1  # Initialize here instead of later

        # autoencoder for intra information
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)
        self.ae.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))

        # GCN for inter information
        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_3 = GNNLayer(n_enc_2, n_enc_3)
        self.gnn_4 = GNNLayer(n_enc_3, n_z)
        self.gnn_5 = GNNLayer(n_z, n_clusters)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v

    def forward(self, x, adj):
        # Get autoencoder outputs
        x_bar, tra1, tra2, tra3, z = self.ae(x)
        
        sigma = 0.5
        gcn_shapes = {}

        # GCN module forward
        h = self.gnn_1(x, adj)
        gcn_shapes['Layer 1'] = h.shape

        h = self.gnn_2((1 - sigma) * h + sigma * tra1, adj)
        gcn_shapes['Layer 2'] = h.shape

        h = self.gnn_3((1 - sigma) * h + sigma * tra2, adj)
        gcn_shapes['Layer 3'] = h.shape

        h = self.gnn_4((1 - sigma) * h + sigma * tra3, adj)
        gcn_shapes['Layer 4'] = h.shape

        h = self.gnn_5((1 - sigma) * h + sigma * z, adj, active=False)
        gcn_shapes['Layer 5'] = h.shape

        # Store GCN shapes
        self.ae.layer_shapes['gcn'] = gcn_shapes

        predict = F.softmax(h, dim=1)

        # Only print shapes once per epoch during training
        if self.training and self.current_epoch != self.last_logged_epoch:
            print(f"\nEpoch {self.current_epoch}")
            print("=" * 50)
            print("\nAutoencoder Architecture:")
            print("-" * 30)
            for layer_name, shape in self.ae.layer_shapes['autoencoder'].items():
                print(f"{layer_name}: {shape}")
            
            print("\nGCN Architecture:")
            print("-" * 30)
            for layer_name, shape in gcn_shapes.items():
                print(f"{layer_name}: {shape}")
            print()
            self.last_logged_epoch = self.current_epoch

        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, predict, z, gcn_shapes

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def train_sdcn(dataset):
    model = SDCN(500, 500, 2000, 2000, 500, 500,
                n_input=args.n_input,
                n_z=args.n_z,
                n_clusters=args.n_clusters,
                v=1.0).to(device)
    print(model)

    optimizer = Adam(model.parameters(), lr=args.lr)

    # KNN Graph
    adj = load_graph(args.name, args.k)
    adj = adj.cuda()

    # cluster parameter initiate
    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y
    with torch.no_grad():
        _, _, _, _, z = model.ae(data)

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    y_pred_last = y_pred
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    eva(y, y_pred, 'pae')

    # Create a list to store results
    results = []

    for epoch in range(200):
        # Update the current epoch
        model.current_epoch = epoch
        
        if epoch % 1 == 0:
            # Get current timestamp for this evaluation
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Evaluate the model
            _, tmp_q, pred, _, _ = model(data, adj)
            
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)

            res1 = tmp_q.cpu().numpy().argmax(1)  # Q
            res2 = pred.data.cpu().numpy().argmax(1)  # Z
            res3 = p.data.cpu().numpy().argmax(1)  # P

            # Get evaluation metrics for each round
            acc1, f1_1, nmi1, ari1 = eva(y, res1, f'{epoch}Q')
            acc2, f1_2, nmi2, ari2 = eva(y, res2, f'{epoch}Z')
            acc3, f1_3, nmi3, ari3 = eva(y, res3, f'{epoch}P')

            # Save clustering results for each round
            results.append([epoch, acc1, f1_1, acc2, f1_2, acc3, f1_3])

        # Correctly unpack all return values
        x_bar, q, pred, _, _ = model(data, adj)

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
        re_loss = F.mse_loss(x_bar, data)

        loss = 0.1 * kl_loss + 0.01 * ce_loss + re_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Correctly unpack all return values when getting final clustering results
    _, _, final_pred, _, _ = model(data, adj)  # Model's final prediction output (Z distribution)
    final_clusters = final_pred.data.cpu().numpy().argmax(1)  # Extract clustering label for each node

    # Convert to DataFrame and save as CSV file
    results_df = pd.DataFrame(results, columns=['Epoch', 'Acc_Q', 'F1_Q', 'Acc_Z', 'F1_Z', 'Acc_P', 'F1_P'])
    results_df.to_csv('training_results.csv', index=False)

    # Print final results
    print("Training complete. Results saved to 'training_results.csv'.")

    # Save final clustering results
    _, _, final_pred, _, _ = model(data, adj)
    final_clusters = final_pred.data.cpu().numpy().argmax(1)
    final_results_df = pd.DataFrame({'Node': np.arange(len(final_clusters)), 'Cluster': final_clusters})
    final_results_df.to_csv('final_cluster_results.csv', index=False)

    print("Final clustering results saved to 'final_cluster_results.csv'.")

class Logger(object):
    def __init__(self, filename="Default.log", terminal_mode="normal"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
        self.terminal_mode = terminal_mode

    def write(self, message):
        # Always write everything to log file
        self.log.write(message)
        
        # For terminal, only show important information
        if self.terminal_mode == "minimal":
            # Only print to terminal if message contains important keywords
            if any(key in message for key in [
                'acc', 'nmi', 'ari', 'f1',  # Metrics
                'Training complete',         # Important status
                'Final clustering',          # Final results
                'use cuda',                  # Hardware info
                'Epoch'                      # Epoch progress
            ]):
                # Skip layer shape information even in epoch headers
                if not any(shape in message for shape in [
                    'Layer', 'Shape', 'Architecture'
                ]):
                    self.terminal.write(message)
        else:
            # Normal mode - print everything
            self.terminal.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Create a log file with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'logs/sdcn_run_{timestamp}.txt'
    
    # Redirect stdout to both console and file, with minimal terminal output
    sys.stdout = Logger(log_filename, terminal_mode="minimal")
    
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='reut')
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_clusters', default=3, type=int)
    parser.add_argument('--n_z', default=10, type=int)
    parser.add_argument('--pretrain_path', type=str, default='pkl')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    args.pretrain_path = 'data/{}.pkl'.format(args.name)
    dataset = load_data(args.name)

    if args.name == 'usps':
        args.n_clusters = 10
        args.n_input = 256

    if args.name == 'hhar':
        args.k = 5
        args.n_clusters = 6
        args.n_input = 561

    if args.name == 'reut':
        args.lr = 1e-4
        args.n_clusters = 4
        args.n_input = 2000

    if args.name == 'acm':
        args.k = None
        args.n_clusters = 3
        args.n_input = 1870

    if args.name == 'dblp':
        args.k = None
        args.n_clusters = 4
        args.n_input = 334

    if args.name == 'cite':
        args.lr = 1e-4
        args.k = None
        args.n_clusters = 6
        args.n_input = 3703


    print(args)
    train_sdcn(dataset)
