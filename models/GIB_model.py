import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATConv, global_mean_pool

class GIBModel(torch.nn.Module):
    def __init__(self):
        super(GIBModel, self).__init__()
        self.n_features = 768
        self.n_hidden = 128
        self.n_classes = 2
        self.n_heads = 4
        self.conv = GATConv(self.n_features, self.n_hidden, heads=self.n_heads, dropout=0.2)
        self.fc = nn.Linear(self.n_hidden * self.n_heads, self.n_classes)

    def forward(self, graph_data):
        output = F.relu(self.conv(graph_data.x, graph_data.edge_index))
        
        relation_matrix = self.construct_relation_matrix(output)
        ib_loss = self.calculate_ib_loss(relation_matrix)
        node_mask = self.create_diagonal_mask(output)
        output = node_mask @ output
        
        output = F.relu(global_mean_pool(output, graph_data.batch))
        output = F.dropout(output, p=0.2, training=self.training)
        output = self.fc(output)
        # y = torch.sigmoid(output)
        return output, ib_loss
    
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