# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:   offline_batch_retrieve.py
   Description: this code is to retrieve offline entity contexts and negative entities for training.
   Author:  Ruijie Wang (https://github.com/xjdwrj)
   date:    14 Nov. 2019
-------------------------------------------------
"""

import torch
from tools.pickle_funcs import load_data
from tools.log_text import log_text

class OfflineBatchRetrieve:
    def __init__(self, names, dataset):
        self.dataset = dataset
        self.names = names
        self.name_index = {"train": 0, "validate": 1, "test": 2}
        self.input_path = "./datasets/%s/input/" % self.dataset
        self.output_path = "./datasets/%s/output/" % self.dataset
        self.log_path = "./logs/offline_batch_retrieve_on_%s.log" % self.dataset

        self.train_head_entities = {}
        self.train_tail_entities = {}
        self.train_both_entities = {}
        self.validate_head_entities = {}
        self.validate_tail_entities = {}
        self.validate_both_entities = {}
        self.test_head_entities = {}
        self.test_tail_entities = {}
        self.test_both_entities = {}
        self.head_entities = [self.train_head_entities, self.validate_head_entities, self.test_head_entities]
        self.tail_entities = [self.train_tail_entities, self.validate_tail_entities, self.test_tail_entities]
        self.both_entities = [self.train_both_entities, self.validate_both_entities, self.test_both_entities]

        self.train_context_head = {}
        self.train_context_head_relation = {}
        self.train_context_tail_relation = {}
        self.train_context_tail = {}
        self.validate_context_head = {}
        self.validate_context_head_relation = {}
        self.validate_context_tail_relation = {}
        self.validate_context_tail = {}
        self.test_context_head = {}
        self.test_context_head_relation = {}
        self.test_context_tail_relation = {}
        self.test_context_tail = {}
        self.context_heads = [self.train_context_head, self.validate_context_head, self.test_context_head]
        self.context_head_relations = [self.train_context_head_relation, self.validate_context_head_relation, self.test_context_head_relation]
        self.context_tail_relations = [self.train_context_tail_relation, self.validate_context_tail_relation, self.test_context_tail_relation]
        self.context_tails = [self.train_context_tail, self.validate_context_tail, self.test_context_tail]

        self.train_negatives = {}
        self.validate_negatives = {}
        self.test_negatives = {}
        self.negatives = [self.train_negatives, self.validate_negatives, self.test_negatives]

        self.read_data()

    def read_data(self):
        log_text(self.log_path, "...... Reading Data for Offline Batch Generation ......")
        for index in range(len(self.names)):
            name = self.names[index]
            self.read_dict(self.head_entities[index], load_data(self.output_path + "%s_head_entities.pickle" % name, self.log_path, "self.%s_head_entities" % name))
            self.read_dict(self.tail_entities[index], load_data(self.output_path + "%s_tail_entities.pickle" % name, self.log_path, "self.%s_tail_entities" % name))
            self.read_dict(self.both_entities[index], load_data(self.output_path + "%s_both_entities.pickle" % name, self.log_path, "self.%s_both_entities" % name))

            self.read_dict(self.context_heads[index], load_data(self.output_path + "%s_context_head.pickle" % name, self.log_path, "self.%s_context_head" % name))
            self.read_dict(self.context_head_relations[index], load_data(self.output_path + "%s_context_head_relation.pickle" % name, self.log_path, "self.%s_context_head_relation" % name))
            self.read_dict(self.context_tail_relations[index], load_data(self.output_path + "%s_context_tail_relation.pickle" % name, self.log_path, "self.%s_context_tail_relation" % name))
            self.read_dict(self.context_tails[index], load_data(self.output_path + "%s_context_tail.pickle" % name, self.log_path, "self.%s_context_tail" % name))

            self.read_dict(self.negatives[index], load_data(self.output_path + "%s_negatives.pickle" % name, self.log_path, "self.%s_negatives" % name))

    def re_read_context_and_negatives(self):
        log_text(self.log_path, "...... Reading Data for Offline Batch Generation ......")
        for index in range(len(self.names)):
            name = self.names[index]
            self.context_heads[index].clear()
            self.context_head_relations[index].clear()
            self.context_tail_relations[index].clear()
            self.context_tails[index].clear()
            self.read_dict(self.context_heads[index], load_data(self.output_path + "%s_context_head.pickle" % name, self.log_path, "self.%s_context_head" % name))
            self.read_dict(self.context_head_relations[index], load_data(self.output_path + "%s_context_head_relation.pickle" % name, self.log_path, "self.%s_context_head_relation" % name))
            self.read_dict(self.context_tail_relations[index], load_data(self.output_path + "%s_context_tail_relation.pickle" % name, self.log_path, "self.%s_context_tail_relation" % name))
            self.read_dict(self.context_tails[index], load_data(self.output_path + "%s_context_tail.pickle" % name, self.log_path, "self.%s_context_tail" % name))

            self.negatives[index].clear()
            self.read_dict(self.negatives[index], load_data(self.output_path + "%s_negatives.pickle" % name, self.log_path, "self.%s_negatives" % name))

    @staticmethod
    def read_dict(dict1, dict2):
        for key in dict2:
            dict1[key] = dict2[key]

    def batch_classification(self, name, entity_batch):
        index = self.name_index[name]
        head_entity_batch = []
        tail_entity_batch = []
        both_entity_batch = []
        for entity in entity_batch:
            if entity in self.head_entities[index]:
                head_entity_batch.append(entity)
            if entity in self.tail_entities[index]:
                tail_entity_batch.append(entity)
            if entity in self.both_entities[index]:
                both_entity_batch.append(entity)
        return head_entity_batch, tail_entity_batch, both_entity_batch

    def head_context_retrieve(self, name, entity_batch):
        index = self.name_index[name]
        head_batch = None
        head_relation_batch = None
        for entity in entity_batch:
            if head_batch is None:
                head_batch = self.context_heads[index][entity]
                head_relation_batch = self.context_head_relations[index][entity]
            else:
                head_batch = torch.cat((head_batch, self.context_heads[index][entity]), 0)
                head_relation_batch = torch.cat((head_relation_batch, self.context_head_relations[index][entity]), 0)
        return head_batch, head_relation_batch

    def tail_context_retrieve(self, name, entity_batch):
        index = self.name_index[name]
        tail_relation_batch = None
        tail_batch = None
        for entity in entity_batch:
            if tail_relation_batch is None:
                tail_relation_batch = self.context_tail_relations[index][entity]
                tail_batch = self.context_tails[index][entity]
            else:
                tail_relation_batch = torch.cat((tail_relation_batch, self.context_tail_relations[index][entity]), 0)
                tail_batch = torch.cat((tail_batch, self.context_tails[index][entity]), 0)
        return tail_relation_batch, tail_batch

    def negative_retrieves(self, name, entity_batch):
        index = self.name_index[name]
        negative_batch = None
        for entity in entity_batch:
            if negative_batch is None:
                negative_batch = self.negatives[index][entity]
            else:
                negative_batch = torch.cat((negative_batch, self.negatives[index][entity]), 0)
        return negative_batch



