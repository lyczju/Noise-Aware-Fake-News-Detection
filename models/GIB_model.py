import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.utils import to_dense_adj

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
        self.fc1 = nn.Linear(self.n_hidden, self.n_hidden)
        self.fc2 = nn.Linear(self.n_hidden, self.n_classes)
        self.mse_loss = nn.MSELoss()

    def forward(self, graph_data):
        # TODO: sampling process and loss
        x, edge_index, batch = graph_data.x, graph_data.edge_index, graph_data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        
        subgraph_clf = F.softmax(self.subgraph_clf_layer(x), dim=1)
        
        graph_embeddings, subgraph_embeddings, aggregate_loss = self.aggregate(x, edge_index, batch, subgraph_clf)
        output = F.relu(self.fc1(subgraph_embeddings))
        output = F.dropout(output, p=0.3, training=self.training)
        output = self.fc2(output)
        return output, aggregate_loss
        
        # relation_matrix = self.construct_relation_matrix(output)
        # ib_loss = self.calculate_ib_loss(relation_matrix)
        # node_mask = self.create_diagonal_mask(output)
        # output = node_mask @ output
        
        # output = F.relu(global_mean_pool(output, batch))
        # output = F.dropout(output, p=0.2, training=self.training)
        # output = self.fc(output)
        # # y = torch.sigmoid(output)
        # return output, ib_loss
    
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
    
    @staticmethod
    def construct_relation_matrix(node_features):
        node_features_normalized = F.normalize(node_features, p=2, dim=1)
        relation_matrix = torch.mm(node_features_normalized, node_features_normalized.t())
        return relation_matrix
    
    @staticmethod
    def calculate_ib_loss(relation_matrix):
        relation_matrix = F.normalize(relation_matrix, p=2, dim=1)
        k = torch.mm(relation_matrix, relation_matrix)
        ib_loss = torch.trace(k) / relation_matrix.size(0) + torch.mean(relation_matrix) ** 2 - 2 * torch.mean(relation_matrix) / relation_matrix.size(0)
        return ib_loss
    
    @staticmethod
    def create_diagonal_mask(node_features, threshold=0.9):
        node_importance = torch.norm(node_features, dim=1)
        
        mask_matrix = torch.eye(node_features.size(0), device=node_features.device)
        
        important_nodes_mask = node_importance > threshold * node_importance.mean()
        mask_matrix = mask_matrix * important_nodes_mask.float().diag()
        
        num_important_nodes = torch.sum(mask_matrix.diagonal())
        
        return mask_matrix