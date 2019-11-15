"""
-------------------------------------------------
   File Name:   online_batch_retrieve.py
   Description: this code is to sample entity context and negative entities for training.
   Author:  Ruijie Wang (https://github.com/xjdwrj)
   date:    14 Nov. 2019
-------------------------------------------------
"""

import torch
from tools.uniform_sampling import sampled_id_generation


class BatchProcess:
    def __init__(self, train_entities, head_entities, tail_entities, both_entities, head_context_head, head_context_relation, head_context_statistics,
                 tail_context_relation, tail_context_tail, tail_context_statistics,
                 head_context_size, tail_context_size, num_of_train_entities, negative_batch_size, device):

        self.head_context_head = head_context_head
        self.head_context_relation = head_context_relation
        self.head_context_statistics = head_context_statistics

        self.tail_context_relation = tail_context_relation
        self.tail_context_tail = tail_context_tail
        self.tail_context_statistics = tail_context_statistics

        self.head_context_size = head_context_size
        self.tail_context_size = tail_context_size
        self.negative_batch_size = negative_batch_size

        self.num_of_train_entities = num_of_train_entities

        self.train_entities = train_entities

        self.train_head_entities = head_entities
        self.train_tail_entities = tail_entities
        self.train_both_entities = both_entities

        self.device = device

    def batch_classification(self, entity_batch):
        head_entity_batch = []
        tail_entity_batch = []
        both_entity_batch = []
        for entity in entity_batch:
            if entity in self.train_head_entities:
                head_entity_batch.append(entity)
            if entity in self.train_tail_entities:
                tail_entity_batch.append(entity)
            if entity in self.train_both_entities:
                both_entity_batch.append(entity)
        return head_entity_batch, tail_entity_batch, both_entity_batch

    def head_context_process(self, head_batch):
        head_head = torch.LongTensor(len(head_batch), self.head_context_size)
        head_relation = torch.LongTensor(len(head_batch), self.head_context_size)
        for index in range(len(head_batch)):
            entity = head_batch[index]
            heads = self.head_context_head[entity]
            relations = self.head_context_relation[entity]
            num_of_head_context = self.head_context_statistics[entity]
            sampled_ids = sampled_id_generation(0, num_of_head_context, self.head_context_size)
            head_head[index] = torch.LongTensor([heads[_] for _ in sampled_ids])
            head_relation[index] = torch.LongTensor([relations[_] for _ in sampled_ids])
        return head_head, head_relation

    def tail_context_process(self, tail_batch):
        tail_relation = torch.LongTensor(len(tail_batch), self.tail_context_size)
        tail_tail = torch.LongTensor(len(tail_batch), self.tail_context_size)
        for index in range(len(tail_batch)):
            entity = tail_batch[index]
            relations = self.tail_context_relation[entity]
            tails = self.tail_context_tail[entity]
            num_of_tail_context = self.tail_context_statistics[entity]
            sampled_ids = sampled_id_generation(0, num_of_tail_context, self.tail_context_size)
            tail_relation[index] = torch.LongTensor([relations[_] for _ in sampled_ids])
            tail_tail[index] = torch.LongTensor([tails[_] for _ in sampled_ids])
        return tail_relation, tail_tail

    def negative_batch_generation(self, positive_batch):
        negative_batch = torch.LongTensor(len(positive_batch), self.negative_batch_size)
        for index in range(negative_batch.size()[0]):
            entity = positive_batch[index]
            negative_entities = []
            sampled_entities = {}
            sampled_entity_count = 0
            while len(negative_entities) < self.negative_batch_size and sampled_entity_count < self.num_of_train_entities:
                sampled_entity_id = sampled_id_generation(0, self.num_of_train_entities, 1)[0]
                while sampled_entity_id in sampled_entities:
                    sampled_entity_id = sampled_id_generation(0, self.num_of_train_entities, 1)[0]
                sampled_entities[sampled_entity_id] = None
                sampled_entity_count += 1
                if self.negative_or_not(entity, self.train_entities[sampled_entity_id]):
                    negative_entities.append(self.train_entities[sampled_entity_id])
            if len(negative_entities) == 0:
                negative_entities = [self.train_entities[tmp_id] for tmp_id in sampled_id_generation(0, self.num_of_train_entities, self.negative_batch_size)]
            if len(negative_entities) < self.negative_batch_size:
                sampled_indices = sampled_id_generation(0, len(negative_entities), self.negative_batch_size - len(negative_entities))
                for sampled_index in sampled_indices:
                    negative_entities.append(negative_entities[sampled_index])
            negative_batch[index] = torch.FloatTensor(negative_entities)
        return negative_batch

    def negative_or_not(self, entity, sampled_entity):
        is_negative = True
        original_head = self.head_context_head[entity]
        original_relation = self.head_context_relation[entity]
        original_num = self.head_context_statistics[entity]
        sampled_head = self.head_context_head[sampled_entity]
        sampled_relation = self.head_context_relation[sampled_entity]
        sampled_num = self.head_context_statistics[sampled_entity]
        if original_num > 0 and sampled_num > 0:
            if self.compare_context(original_head, original_relation, original_num, sampled_head, sampled_relation, sampled_num) == "intersection":
                is_negative = False
        if is_negative:
            original_relation = self.tail_context_relation[entity]
            original_tail = self.tail_context_tail[entity]
            original_num = self.tail_context_statistics[entity]
            sampled_relation = self.tail_context_relation[sampled_entity]
            sampled_tail = self.tail_context_tail[sampled_entity]
            sampled_num = self.tail_context_statistics[sampled_entity]
            if original_num > 0 and sampled_num > 0:
                if self.compare_context(original_relation, original_tail, original_num, sampled_relation, sampled_tail, sampled_num) == "intersection":
                    is_negative = False
        return is_negative

    def compare_context(self, original_first, original_second, original_num, sampled_first, sampled_second, sampled_num):
        o_f = torch.LongTensor(1, original_num)
        o_s = torch.LongTensor(1, original_num)
        s_f = torch.LongTensor(1, sampled_num)
        s_s = torch.LongTensor(1, sampled_num)
        for index in original_first:
            o_f[0, index] = original_first[index]
            o_s[0, index] = original_second[index]
        for index in sampled_first:
            s_f[0, index] = sampled_first[index]
            s_s[0, index] = sampled_second[index]
        o_f = o_f.to(self.device)
        o_s = o_s.to(self.device)
        s_f = s_f.to(self.device)
        s_s = s_s.to(self.device)
        o = torch.unsqueeze(torch.cat((o_f, o_s), 0), 2)  # (2, original_num, 1)
        s = torch.unsqueeze(torch.cat((s_f, s_s), 0), 1)  # (2, 1, sampled_num)
        tmp_compare = torch.sum(torch.abs(o - s), 0)  # (original_num, sampled_num)
        if torch.nonzero(tmp_compare).size()[0] < original_num * sampled_num:
            return "intersection"
        else:
            return "non-intersection"













