# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:   iswc_model.py
   Description: this code is an implementation of our iswc model in PyTorch.
   Author:  Ruijie Wang (https://github.com/xjdwrj)
   date:    14 Nov. 2019
-------------------------------------------------
"""

import torch
import torch.nn as nn
import torch.nn.functional as funcs
from torch.utils.data import DataLoader
from tools.pickle_funcs import dump_data, load_data
from tools.dataset import MyDataset


class Model(nn.Module):

    def __init__(self, result_path, log_path, entity_dimension, relation_dimension, num_of_entities, num_of_relations, norm, device):
        super(Model, self).__init__()

        self.result_path = result_path
        self.log_path = log_path

        self.entity_dimension = entity_dimension
        self.relation_dimension = relation_dimension

        self.num_of_entities = num_of_entities
        self.num_of_relations = num_of_relations

        self.entity_embeddings = nn.Embedding(self.num_of_entities, self.entity_dimension)
        self.relation_embeddings = nn.Embedding(self.num_of_relations, self.relation_dimension)

        self.norm = norm
        self.device = device

        self.log_sigmoid = nn.LogSigmoid()

        sqrt_entity = self.entity_dimension ** 0.5
        sqrt_relation = self.relation_dimension ** 0.5

        self.entity_embeddings.weight.data = torch.FloatTensor(self.num_of_entities, self.entity_dimension).uniform_(-6. / sqrt_entity, 6. / sqrt_entity)
        self.entity_embeddings.weight.data = funcs.normalize(self.entity_embeddings.weight.data, self.norm, 1)
        self.relation_embeddings.weight.data = torch.FloatTensor(self.num_of_relations, self.relation_dimension).uniform_(-6. / sqrt_relation, 6. / sqrt_relation)
        self.relation_embeddings.weight.data = funcs.normalize(self.relation_embeddings.weight.data, self.norm, 1)

    def forward(self, entity_batch, head_batch, head_relation_batch, tail_relation_batch, tail_batch, negative_batch):
        entity_embeddings = self.entity_embeddings(entity_batch)  # (batch_size, entity_dim)
        negative_embeddings = self.entity_embeddings(negative_batch)  # (batch_size, negative_batch_size, entity_dim)
        object_embeddings = torch.cat((torch.unsqueeze(entity_embeddings, 1), negative_embeddings), 1)  # (batch_size, 1 + negative_batch_size, entity_dim)
        f_2_head, f_2_tail, f_1 = None, None, None
        if head_batch is not None:
            head_embeddings = self.entity_embeddings(head_batch)  # (batch_size, head_context_size, entity_dim)
            head_relation_embeddings = self.relation_embeddings(head_relation_batch)  # (batch_size, head_context_size, relation_dim)
            f_2_head = torch.norm(torch.unsqueeze(head_embeddings, 1) + torch.unsqueeze(head_relation_embeddings, 1) - torch.unsqueeze(object_embeddings, 2), self.norm, 3)
            # (batch_size, 1 + negative_batch_size, head_context_size)
            f_1_head = -1. * torch.sum(f_2_head, 2) / f_2_head.size()[2]
            # (batch_size, 1 + negative_batch_size)
            f_1 = f_1_head
        if tail_batch is not None:
            tail_relation_embeddings = self.relation_embeddings(tail_relation_batch)  # (batch_size, tail_context_size, relation_dim)
            tail_embeddings = self.entity_embeddings(tail_batch)  # (batch_size, tail_context_size, entity_dim)
            f_2_tail = torch.norm(torch.unsqueeze(object_embeddings, 2) + torch.unsqueeze(tail_relation_embeddings, 1) - torch.unsqueeze(tail_embeddings, 1), self.norm, 3)
            # (batch_size, 1 + negative_batch_size, tail_context_size)
            f_1_tail = -1. * torch.sum(f_2_tail, 2) / f_2_tail.size()[2]
            # (batch_size, 1 + negative_batch_size)
            f_1 = f_1_tail
        if head_batch is not None and tail_batch is not None:
            f_1_both = -1. * torch.sum(torch.cat((f_2_head, f_2_tail), 2), 2) / (f_2_head.size()[2] + f_2_tail.size()[2])
            # (batch_size, 1 + negative_batch_size)
            f_1 = f_1_both
        tmp_ones = torch.ones(f_1.size()[1]) * -1.
        tmp_ones[0] = 1.
        obj_func = torch.sum(self.log_sigmoid(f_1 * tmp_ones.to(self.device)), 1)  # (batch_size,)
        return torch.sum(obj_func, 0)

    def output(self):
        dump_data(self.entity_embeddings.weight.data.to("cpu"), self.result_path + "entity_embeddings.pickle",
                  self.log_path, "self.entity_embeddings.weight.data")
        dump_data(self.relation_embeddings.weight.data.to("cpu"), self.result_path + "relation_embeddings.pickle",
                  self.log_path, "self.relation_embeddings.weight.data")

    def input(self):
        self.entity_embeddings.weight.data = load_data(self.result_path + "entity_embeddings.pickle",
                  self.log_path, "self.entity_embeddings.weight.data")
        self.relation_embeddings.weight.data = load_data(self.result_path + "relation_embeddings.pickle",
                  self.log_path, "self.relation_embeddings.weight.data")

    def normalize(self):
        self.entity_embeddings.weight.data = funcs.normalize(self.entity_embeddings.weight.data, 2, 1)

    def validate(self, id_heads, id_relations, id_tails):
        head_embeddings = self.entity_embeddings(id_heads)  # (valid_size, entity_embedding_dim)
        relation_embeddings = self.relation_embeddings(id_relations)  # (valid_size, relation_embedding_dim)
        tail_embeddings = self.entity_embeddings(id_tails)  # (valid_size, entity_embedding_dim)

        target_loss = torch.norm(head_embeddings + relation_embeddings - tail_embeddings, self.norm, 1)  # (valid_size,)
        tmp_head_loss = torch.norm(
            torch.unsqueeze(self.entity_embeddings.weight.data, 1) + relation_embeddings - tail_embeddings, self.norm,
            2)  # (num_of_entities, valid_size)
        tmp_tail_loss = torch.norm(
            head_embeddings + relation_embeddings - torch.unsqueeze(self.entity_embeddings.weight.data, 1), self.norm,
            2)  # (num_of_entities, valid_size)

        rank_h = torch.nonzero(nn.functional.relu(target_loss - tmp_head_loss)).size()[0]
        rank_t = torch.nonzero(nn.functional.relu(target_loss - tmp_tail_loss)).size()[0]

        return (rank_h + rank_t + 2) / 2

    def test_calc(self, n_of_hit, test_result, train_triple_tensor, test_heads, test_relations, test_tails):
        test_head_embeddings = self.entity_embeddings(test_heads)  # (num_of_test_triples, entity_dim)
        test_relation_embeddings = self.relation_embeddings(test_relations)  # (num_of_test_triples, relation_dim)
        test_tail_embeddings = self.entity_embeddings(test_tails)  # (num_of_test_triples, entity_dim)

        target_loss = torch.norm(test_head_embeddings + test_relation_embeddings - test_tail_embeddings, self.norm, 1)  # (num_of_test_triples,)
        tmp_head_loss = torch.norm(torch.unsqueeze(self.entity_embeddings.weight.data, 1) + test_relation_embeddings - test_tail_embeddings, self.norm, 2)  # (num_of_entities, num_of_test_triples)
        tmp_tail_loss = torch.norm(test_head_embeddings + test_relation_embeddings - torch.unsqueeze(self.entity_embeddings.weight.data, 1), self.norm, 2)  # (num_of_entities, num_of_test_triples)

        better_heads = torch.nonzero(nn.functional.relu(target_loss - tmp_head_loss))  # (number of better heads, 2)
        better_tails = torch.nonzero(nn.functional.relu(target_loss - tmp_tail_loss))  # (number of better tails, 2)

        rank_h = better_heads.size()[0]
        rank_t = better_tails.size()[0]

        test_result[0] += (rank_h + rank_t + 2) / 2
        if rank_h + 1 <= n_of_hit * test_heads.size()[0]:
            test_result[1] += test_heads.size()[0]
        if rank_t + 1 <= n_of_hit * test_heads.size()[0]:
            test_result[1] += test_heads.size()[0]

        existing_heads = 0
        existing_tails = 0
        batch_num = 200
        dataset_h = MyDataset(rank_h)
        data_loader_h = DataLoader(dataset_h, batch_num, False)
        for batch in data_loader_h:
            existing_heads += torch.nonzero(torch.relu(-1 * torch.sum(torch.abs(torch.cat((torch.unsqueeze(better_heads[batch, 0], 1), torch.unsqueeze(test_relations[better_heads[batch, 1]], 1), torch.unsqueeze(test_tails[better_heads[batch, 1]], 1)), 1) - torch.unsqueeze(train_triple_tensor, 1)), 2) + 0.5)).size()[0]
        dataset_t = MyDataset(rank_t)
        data_loader_t = DataLoader(dataset_t, batch_num, False)
        for batch in data_loader_t:
            existing_tails += torch.nonzero(torch.relu(-1 * torch.sum(torch.abs(torch.cat((torch.unsqueeze(test_heads[better_tails[batch, 1]], 1), torch.unsqueeze(test_relations[better_tails[batch, 1]], 1), torch.unsqueeze(better_tails[batch, 0], 1)), 1) - torch.unsqueeze(train_triple_tensor, 1)), 2) + 0.5)).size()[0]

        filtered_rank_h = rank_h - existing_heads
        filtered_rank_t = rank_t - existing_tails

        test_result[2] += (filtered_rank_h + filtered_rank_t + 2) / 2
        if filtered_rank_h + 1 <= n_of_hit * test_heads.size()[0]:
            test_result[3] += test_heads.size()[0]
        if filtered_rank_t + 1 <= n_of_hit * test_heads.size()[0]:
            test_result[3] += test_heads.size()[0]