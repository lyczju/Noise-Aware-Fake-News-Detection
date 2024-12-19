import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.utils import to_dense_adj
import time
import pdb

class GIBModel(torch.nn.Module):
    def __init__(self):
        super(GIBModel, self).__init__()
        self.n_features = 768
        self.n_hidden = 128
        self.n_classes = 2
        self.n_heads = 4
        self.conv1 = GATConv(self.n_features, self.n_hidden, heads=self.n_heads, dropout=0.2)
        self.conv2 = GATConv(self.n_hidden * self.n_heads, self.n_hidden, dropout=0.2)
        self.subgraph_clf_layer = nn.Linear(self.n_hidden, 2)
        self.subgraph_detect_layer = nn.Linear(self.n_hidden, 2)    # mask select
        self.fc1 = nn.Linear(self.n_hidden, self.n_hidden)
        self.fc2 = nn.Linear(self.n_hidden, self.n_classes)
        self.mse_loss = nn.MSELoss()
        self.ib_estimator_layer1 = nn.Linear(self.n_hidden * 2, self.n_hidden)
        self.ib_estimator_layer2 = nn.Linear(self.n_hidden, 1)

    def forward(self, graph_data):
        # TODO: sampling process and loss
        node_x, edge_index, batch = graph_data.x, graph_data.edge_index, graph_data.batch
        x = self.gnn(node_x, edge_index)
        if torch.isnan(x[0][0]):
            print("nan detected!")
        
        subgraph_clf = F.softmax(self.subgraph_clf_layer(x), dim=1)
        
        graph_embeddings, subgraph_embeddings, aggregate_loss = self.aggregate(x, edge_index, batch, subgraph_clf)
        mi_loss = self.mutual_information_estimation(graph_embeddings, subgraph_embeddings)

        # node_sig = self.determine_subgraph(node_x, edge_index, batch)
        # detect_loss = torch.mean(torch.abs(subgraph_clf[:, 0] - node_sig[:, 0]))
        detect_loss = 0

        output = F.relu(self.fc1(subgraph_embeddings))
        output = F.dropout(output, p=0.3, training=self.training)
        output = self.fc2(output)
        return output, aggregate_loss, detect_loss, mi_loss
        

    
    def aggregate(self, x, edge_index, batch, subgraph_clf):
        if torch.cuda.is_available():
            ones = torch.ones(2).cuda()
        else:
            ones = torch.ones(2)
        
        graph_embeddings = []
        subgraph_embeddings = []
        total_loss = 0
        
        dense_adj = to_dense_adj(edge_index)[0]
        
        graph_ids = torch.unique(batch)
        
        for graph_idx in graph_ids:
            mask = (batch == graph_idx)
            
            graph_x = x[mask]
            graph_adj = dense_adj[mask][:, mask]
            graph_subgraph_clf = subgraph_clf[mask]
            
            aggregated_adj = graph_subgraph_clf.t() @ graph_adj @ graph_subgraph_clf
            normalized_adj = F.normalize(aggregated_adj, p=1, dim=1, eps=1e-5)
            
            loss = self.mse_loss(normalized_adj.diagonal(), ones)
            total_loss += loss
            
            graph_embedding = torch.mean(graph_x, dim=0, keepdim=True)
            graph_embeddings.append(graph_embedding)
            
            aggregated_embeddings = graph_subgraph_clf.t() @ graph_x
            subgraph_embedding = aggregated_embeddings[0].unsqueeze(0)
            subgraph_embeddings.append(subgraph_embedding)
            
        graph_embeddings = torch.cat(graph_embeddings, dim=0)
        subgraph_embeddings = torch.cat(subgraph_embeddings, dim=0)
        total_loss = total_loss / len(graph_ids)
        
        return graph_embeddings, subgraph_embeddings, total_loss


    def mutual_information_estimation(self, graph_embeddings, subgraph_embeddings):
        shuffle_embeddings = graph_embeddings[torch.randperm(graph_embeddings.shape[0])]
        joint_embeddings = torch.cat([graph_embeddings, subgraph_embeddings], dim=-1)
        margin_embeddings = torch.cat([shuffle_embeddings, subgraph_embeddings], dim=-1)
        joint = F.relu(self.ib_estimator_layer2(F.relu(self.ib_estimator_layer1(joint_embeddings))))
        margin = F.relu(self.ib_estimator_layer2(F.relu(self.ib_estimator_layer1(margin_embeddings))))
        mi_est = torch.clamp(torch.log(torch.clamp(torch.mean(torch.exp(margin)),1,1e+25)),-100000,100000) - torch.mean(joint)
        return mi_est

    
    def determine_subgraph(self, x, edge_index, batch):
        # return node_sig: node_num * 2, one-hot flag of node significance

        if self.training:
            node_embeddings = self.aggregate_by_hop(x, edge_index, batch, 2)

        else:
            node_embeddings = self.gnn(x, edge_index)

        node_sig = F.softmax(self.subgraph_detect_layer(node_embeddings), dim=1)

        return node_sig
        
    def gnn(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return x
    
    def aggregate_by_hop(self, x, edge_index, batch, hop):
        # aggregate heighbor embeddings based on hop 
        # return embeddings of each node

        node_embeddings = torch.zeros([x.size(dim=0), self.n_hidden]) 
        if torch.cuda.is_available():
            node_embeddings = node_embeddings.cuda()
        graph_ids = torch.unique(batch)
        
        for graph_idx in graph_ids:
            start_time = time.time()
            mask = (batch == graph_idx)
            mask_idx = torch.nonzero(mask).squeeze().tolist()

            # build one graph for each node
            for node_global_idx in mask_idx:
                neighbor_edges = []
                subgraph_embeddings = torch.zeros(x.size())
                if torch.cuda.is_available():
                    subgraph_embeddings = subgraph_embeddings.cuda()
                subgraph_embeddings[node_global_idx] = x[node_global_idx]     
                queue = []
                queue_hop = []

                edges = edge_index[:, edge_index[0] == node_global_idx].t().tolist()
                queue.extend(edges)
                queue_hop.extend([hop] * len(edges))

                while len(queue) > 0:
                    neighbor_edge = queue.pop(0)
                    neighbor_edges.append(neighbor_edge)
                    neighbor_edges.append(neighbor_edge[::-1])

                    _node_global_idx = neighbor_edge[0]
                    subgraph_embeddings[_node_global_idx] = x[_node_global_idx]
                    _node_global_idx = neighbor_edge[1]
                    subgraph_embeddings[_node_global_idx] = x[_node_global_idx] 

                    edge_hop = queue_hop.pop(0) - 1

                    if edge_hop > 0:
                        _node_global_idx = neighbor_edge[1]
                        edges = edge_index[:, edge_index[0] == _node_global_idx].t().tolist()
                        edges = [e for e in edges if e not in neighbor_edges]
                        queue.extend(edges)
                        queue_hop.extend([edge_hop] * len(edges))

                neighbor_edges = torch.LongTensor(neighbor_edges).t()
                if torch.cuda.is_available():
                    neighbor_edges = neighbor_edges.cuda()
                
                
                x_embedding = self.gnn(subgraph_embeddings, neighbor_edges)
                node_embeddings[node_global_idx] = x_embedding[node_global_idx]
            print("time(seconds):", time.time() - start_time)

        return node_embeddings


    def mutual_information_estimation(self, graph_embeddings, subgraph_embeddings):
        shuffle_embeddings = graph_embeddings[torch.randperm(graph_embeddings.shape[0])]
        joint_embeddings = torch.cat([graph_embeddings, subgraph_embeddings], dim=-1)
        margin_embeddings = torch.cat([shuffle_embeddings, subgraph_embeddings], dim=-1)
        joint = F.relu(self.ib_estimator_layer2(F.relu(self.ib_estimator_layer1(joint_embeddings))))
        margin = F.relu(self.ib_estimator_layer2(F.relu(self.ib_estimator_layer1(margin_embeddings))))
        mi_est = torch.mean(joint) - torch.clamp(torch.log(torch.mean(torch.exp(margin))),-100000,100000)
        return mi_est
