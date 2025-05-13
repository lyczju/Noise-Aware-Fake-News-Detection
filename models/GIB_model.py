import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import time
import pdb
import math

class GIBModel(torch.nn.Module):
    def __init__(self):
        super(GIBModel, self).__init__()
        self.n_features = 768
        self.n_hidden = 256
        self.n_classes = 2
        self.n_heads = 4
        self.n_subgraphs = 10
        
        self.conv1 = GATConv(self.n_features, self.n_hidden, heads=self.n_heads, dropout=0.2)
        self.conv2 = GATConv(self.n_hidden * self.n_heads, self.n_hidden, dropout=0.2)
        # self.conv3 = GATConv(self.n_features, self.n_hidden, heads=self.n_heads, dropout=0.2)
        # self.conv4 = GATConv(self.n_hidden * self.n_heads, self.n_hidden, dropout=0.2)
        # self.subgraph_clf_layer = nn.Linear(self.n_hidden, 2)
        self.subgraph_clf_layers = nn.ModuleList([
            nn.Linear(self.n_hidden, 1) for _ in range(self.n_subgraphs)
        ])
        # self.subgraph_detect_layer = nn.Linear(self.n_hidden, 2)    # mask select
        self.fc1 = nn.Linear(self.n_hidden, self.n_hidden)
        self.fc2 = nn.Linear(self.n_hidden, self.n_classes)
        self.mse_loss = nn.MSELoss()
        self.ib_estimator_layer1 = nn.Linear(self.n_hidden * 2, self.n_hidden)
        self.ib_estimator_layer2 = nn.Linear(self.n_hidden, 1)
        self.batch_norm1 = nn.BatchNorm1d(self.n_hidden)
        self.batch_norm2 = nn.BatchNorm1d(1)
        self.alpha = 2
        self.beta = 1
        self.gamma = 0.5
        

    def forward(self, graph_data):
        x, edge_index, batch = graph_data.x, graph_data.edge_index, graph_data.batch
        x_emb = self.gnn(x, edge_index)
        
        # subgraph_clf = F.softmax(self.subgraph_clf_layer(x), dim=1)
        # graph_embeddings, subgraph_embeddings, aggregate_loss = self.aggregate(x, edge_index, batch, subgraph_clf)
        # mi_loss = self.mutual_information_estimation(graph_embeddings, subgraph_embeddings)
        
        subgraph_clfs = [
            torch.sigmoid(clf_layer(x_emb))
            for clf_layer in self.subgraph_clf_layers
        ]
        
        graph_embeddings_list = []
        subgraph_embeddings_list = []
        mi_losses = []
        all_node_masks = []
        
        for subgraph_clf in subgraph_clfs:
            g_emb, s_emb, node_masks = self.aggregate(x, x_emb, edge_index, batch, subgraph_clf)
            graph_embeddings_list.append(g_emb)
            subgraph_embeddings_list.append(s_emb)
            all_node_masks.append(node_masks)

            mi_loss = self.mutual_information_estimation(g_emb, s_emb)
            mi_losses.append(mi_loss)
        
                
        if self.training:
            # During training: use all subgraphs
            diversity_loss = 0
            n_pairs = 0
            for i in range(len(subgraph_embeddings_list)):
                for j in range(i + 1, len(subgraph_embeddings_list)):
                    # TODO: DPP?
                    sim = F.cosine_similarity(
                        subgraph_embeddings_list[i],
                        subgraph_embeddings_list[j],
                        dim=1
                    ).mean()
                    diversity_loss += sim
                    n_pairs += 1
            diversity_loss = diversity_loss / n_pairs
            
            coverage_loss = self.calculate_coverage_loss(all_node_masks)
            
            total_mi_loss = 0
            outputs = []
            for i in range(len(subgraph_embeddings_list)):
                output = F.relu(self.fc1(subgraph_embeddings_list[i]))
                output = F.dropout(output, p=0.3, training=self.training)
                output = self.fc2(output)
                outputs.append(output)
                total_mi_loss += mi_losses[i]
            
            mi_loss = total_mi_loss / len(subgraph_embeddings_list)
            output = torch.stack(outputs).mean(0)  # Average predictions
            # print(f'mi loss: {mi_loss}, diversity_loss: {diversity_loss}, coverage_loss: {coverage_loss}')
            return output, self.alpha * mi_loss + self.beta * diversity_loss + self.gamma * coverage_loss
        else:
            # During evaluation: use best subgraph (original logic)
            min_mi_idx = torch.argmin(torch.tensor(mi_losses))
            subgraph_embeddings = subgraph_embeddings_list[min_mi_idx]
            mi_loss = mi_losses[min_mi_idx]
            
            output = F.relu(self.fc1(subgraph_embeddings))
            output = F.dropout(output, p=0.3, training=self.training)
            output = self.fc2(output)
            
            return output, mi_loss
        

    
    def aggregate(self, x, x_emb, edge_index, batch, subgraph_clf, threshold=0.5):
        if torch.cuda.is_available():
            ones = torch.ones(2).cuda()
        else:
            ones = torch.ones(2)
        
        graph_embeddings = []
        subgraph_embeddings = []
        total_loss = 0
        node_masks = []
        
        # dense_adj = to_dense_adj(edge_index)[0]
        
        graph_ids = torch.unique(batch)
        
        for graph_idx in graph_ids:
            mask = (batch == graph_idx)
            
            init_x = x[mask]
            graph_x = x_emb[mask]
            # graph_adj = dense_adj[mask][:, mask]
            graph_subgraph_clf = subgraph_clf[mask]
            
            # current method
            node_probs = torch.sigmoid(graph_subgraph_clf.squeeze())
            node_masks.append(node_probs)
            # key_nodes_x = init_x * node_probs.unsqueeze(1)
            # key_nodes_adj = graph_adj * node_probs.unsqueeze(1) * node_probs.unsqueeze(0)
            # key_edge_index, _ = dense_to_sparse(key_nodes_adj)
            # graph_x_sub = self.subgraph_gnn(key_nodes_x, key_edge_index)
            # subgraph_embedding = torch.mean(graph_x_sub, dim=0, keepdim=True)
            # subgraph_embeddings.append(subgraph_embedding)
            
            # aggregated_adj = graph_subgraph_clf.t() @ graph_adj @ graph_subgraph_clf
            # normalized_adj = F.normalize(aggregated_adj, p=1, dim=1, eps=1e-5)
            
            # loss = self.mse_loss(normalized_adj.diagonal(), ones)
            # total_loss += loss
            
            graph_embedding = torch.mean(graph_x, dim=0, keepdim=True)
            graph_embeddings.append(graph_embedding)
            
            # previous method
            aggregated_embeddings = graph_subgraph_clf.t() @ graph_x
            subgraph_embedding = aggregated_embeddings[0].unsqueeze(0)
            subgraph_embeddings.append(subgraph_embedding)            
        
        graph_embeddings = torch.cat(graph_embeddings, dim=0)
        subgraph_embeddings = torch.cat(subgraph_embeddings, dim=0)
        # total_loss = total_loss / len(graph_ids)
        
        return graph_embeddings, subgraph_embeddings, node_masks


    def mutual_information_estimation(self, graph_embeddings, subgraph_embeddings):
        shuffle_embeddings = graph_embeddings[torch.randperm(graph_embeddings.shape[0])]
        joint_embeddings = torch.cat([graph_embeddings, subgraph_embeddings], dim=-1)
        margin_embeddings = torch.cat([shuffle_embeddings, subgraph_embeddings], dim=-1)
        joint = self.estimator(joint_embeddings)
        margin = self.estimator(margin_embeddings)
        mi_est = torch.clamp(torch.log(torch.clamp(torch.mean(torch.exp(margin)),1,1e+25)),-100000,100000) - torch.mean(joint)
        return mi_est
    
    
    def calculate_coverage_loss(self, all_node_masks):
        """Calculate coverage and overlap loss for each graph separately"""
        # all_node_masks: list of n_subgraphs tensors, each tensor has shape [n_graphs]
        n_graphs = len(all_node_masks[0])
        total_loss = 0
        
        # calculate loss for each graph
        for graph_idx in range(n_graphs):
            # collect masks from all classifiers
            # Stack masks for current graph: n_subgraphs x n_nodes
            graph_masks = torch.stack([masks[graph_idx] for masks in all_node_masks])
            if graph_masks.ndimension() == 1:
                graph_masks = graph_masks.unsqueeze(1)
            
            # Coverage term: encourage complete coverage of graph
            coverage = torch.mean(torch.max(graph_masks, dim=0)[0])
            coverage_term = 1 - coverage
            
            # Overlap term: penalize overlapping selections between subgraphs
            pairwise_overlaps = torch.matmul(graph_masks, graph_masks.transpose(0, 1))
            # Remove self-overlaps from diagonal and normalize
            overlap = (pairwise_overlaps.sum() - pairwise_overlaps.diagonal().sum()) / (2 * graph_masks.size(1) * self.n_subgraphs)
            
            total_loss += coverage_term + overlap
        
        return total_loss / n_graphs
        
    def gnn(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return x

    def estimator(self, x):
        x = F.relu(self.batch_norm1(self.ib_estimator_layer1(x)))
        x = F.relu(self.batch_norm2(self.ib_estimator_layer2(x)))
        return x
