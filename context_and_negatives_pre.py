# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:   context_and_negatives_pre.py
   Description: this code generates sampled entity contexts and negative entities for training.
   Author:  Ruijie Wang (https://github.com/xjdwrj)
   date:    14 Nov. 2019
-------------------------------------------------
"""

import torch
from tools.pickle_funcs import load_data, dump_data
from tools.uniform_sampling import sampled_id_generation
from tools.log_text import log_text


class ContextAndNegatives:
    def __init__(self, names, dataset, head_context_size, tail_context_size, negative_batch_size):
        self.print_results_for_validation = False
        self.dataset = dataset
        self.names = names
        self.input_path = "./datasets/%s/input/" % self.dataset
        self.output_path = "./datasets/%s/output/" % self.dataset
        self.log_path = "./logs/context_and_negatives_pre_on_%s.log" % self.dataset

        self.head_context_size = head_context_size
        self.tail_context_size = tail_context_size
        self.negative_batch_size = negative_batch_size

        self.statistics = None
        self.num_of_entities = None

        self.train_head_relation_to_tail = {}
        self.train_tail_relation_to_head = {}
        self.train_head_context_statistics = {}
        self.train_tail_context_statistics = {}
        self.validate_head_relation_to_tail = {}
        self.validate_tail_relation_to_head = {}
        self.validate_head_context_statistics = {}
        self.validate_tail_context_statistics = {}
        self.test_head_relation_to_tail = {}
        self.test_tail_relation_to_head = {}
        self.test_head_context_statistics = {}
        self.test_tail_context_statistics = {}
        self.head_relation_to_tails = [self.train_head_relation_to_tail, self.validate_head_relation_to_tail, self.test_head_relation_to_tail]
        self.tail_relation_to_heads = [self.train_tail_relation_to_head, self.validate_tail_relation_to_head, self.test_tail_relation_to_head]
        self.head_context_statistics = [self.train_head_context_statistics, self.validate_head_context_statistics, self.test_head_context_statistics]
        self.tail_context_statistics = [self.train_tail_context_statistics, self.validate_tail_context_statistics, self.test_tail_context_statistics]

        self.train_head_context_head = {}
        self.train_head_context_relation = {}
        self.train_tail_context_relation = {}
        self.train_tail_context_tail = {}
        self.validate_head_context_head = {}
        self.validate_head_context_relation = {}
        self.validate_tail_context_relation = {}
        self.validate_tail_context_tail = {}
        self.test_head_context_head = {}
        self.test_head_context_relation = {}
        self.test_tail_context_relation = {}
        self.test_tail_context_tail = {}
        self.head_context_heads = [self.train_head_context_head, self.validate_head_context_head, self.test_head_context_head]
        self.head_context_relations = [self.train_head_context_relation, self.validate_head_context_relation, self.test_head_context_relation]
        self.tail_context_relations = [self.train_tail_context_relation, self.validate_tail_context_relation, self.test_tail_context_relation]
        self.tail_context_tails = [self.train_tail_context_tail, self.validate_tail_context_tail, self.test_tail_context_tail]

        self.train_entities = {}  # entities included in train data
        self.validate_entities = {}  # entities included in validation data
        self.test_entities = {}  # entities included in test data
        self.entity_dicts = [self.train_entities, self.validate_entities, self.test_entities]

        self.train_head_entities = {}  # train entities which have head context
        self.train_tail_entities = {}  # train entities which have tail context
        self.train_both_entities = {}  # train entities which have both context
        self.validate_head_entities = {}  # validation entities which have head context
        self.validate_tail_entities = {}  # validation entities which have head context
        self.validate_both_entities = {}  # validation entities which have head context
        self.test_head_entities = {}  # test entities which have head context
        self.test_tail_entities = {}  # test entities which have head context
        self.test_both_entities = {}  # test entities which have head context

        self.train_entity_head = {}
        self.train_entity_head_relation = {}
        self.train_entity_tail_relation = {}
        self.train_entity_tail = {}
        self.validate_entity_head = {}
        self.validate_entity_head_relation = {}
        self.validate_entity_tail_relation = {}
        self.validate_entity_tail = {}
        self.test_entity_head = {}
        self.test_entity_head_relation = {}
        self.test_entity_tail_relation = {}
        self.test_entity_tail = {}
        self.entity_heads = [self.train_entity_head, self.validate_entity_head, self.test_entity_head]
        self.entity_head_relations = [self.train_entity_head_relation, self.validate_entity_head_relation, self.test_entity_head_relation]
        self.entity_tail_relations = [self.train_entity_tail_relation, self.validate_entity_tail_relation, self.test_entity_tail_relation]
        self.entity_tails = [self.train_entity_tail, self.validate_entity_tail, self.test_entity_tail]

        self.train_negatives = {}
        self.validate_negatives = {}
        self.test_negatives = {}
        self.negatives = [self.train_negatives, self.validate_negatives, self.test_negatives]

        self.run_funcs()

    def run_funcs(self):
        log_text(self.log_path, "...... Reading Data for Context and Negatives Sampling ......")
        self.read_data()

        log_text(self.log_path, "...... Entity Classification ......")
        self.entity_classification()

        log_text(self.log_path, "...... Context Sampling ......")
        self.context_sampling()

        log_text(self.log_path, "...... Negative Sampling ......")
        self.negative_sampling()

        if self.print_results_for_validation:
            log_text(self.log_path, "...... Result Validation ......")
            self.result_validation()

    def re_sampling(self):
        for index in range(3):
            self.entity_heads[index].clear()
            self.entity_head_relations[index].clear()
            self.entity_tail_relations[index].clear()
            self.entity_tails[index].clear()
            self.negatives[index].clear()

        log_text(self.log_path, "...... Context Sampling ......")
        self.context_sampling()

        log_text(self.log_path, "...... Negative Sampling ......")
        self.negative_sampling()

    def read_data(self):
        for index in range(len(self.names)):
            name = self.names[index]
            self.read_dict(self.head_relation_to_tails[index], load_data(self.output_path + "%s_head_relation_to_tail.pickle" % name, self.log_path, "self.%s_head_relation_to_tail" % name))
            self.read_dict(self.tail_relation_to_heads[index], load_data(self.output_path + "%s_tail_relation_to_head.pickle" % name, self.log_path, "self.%s_tail_relation_to_head" % name))
            self.read_dict(self.head_context_statistics[index], load_data(self.output_path + "%s_head_context_statistics.pickle" % name, self.log_path, "self.%s_head_context_statistics" % name))
            self.read_dict(self.tail_context_statistics[index], load_data(self.output_path + "%s_tail_context_statistics.pickle" % name, self.log_path, "self.%s_tail_context_statistics" % name))

            self.read_dict(self.head_context_heads[index], load_data(self.output_path + "%s_head_context_head.pickle" % name, self.log_path, "self.%s_head_context_head" % name))
            self.read_dict(self.head_context_relations[index], load_data(self.output_path + "%s_head_context_relation.pickle" % name, self.log_path, "self.%s_head_context_relation" % name))
            self.read_dict(self.tail_context_relations[index], load_data(self.output_path + "%s_tail_context_relation.pickle" % name, self.log_path, "self.%s_tail_context_relation" % name))
            self.read_dict(self.tail_context_tails[index], load_data(self.output_path + "%s_tail_context_tail.pickle" % name, self.log_path, "self.%s_tail_context_tail" % name))

        self.statistics = load_data(self.output_path + "statistics.pickle", self.log_path, "self.statistics")
        self.num_of_entities = [self.statistics["num_of_train_entities"], self.statistics["num_of_validate_entities"], self.statistics["num_of_test_entities"]]

    @staticmethod
    def read_dict(dict1, dict2):
        for key in dict2:
            dict1[key] = dict2[key]

    def entity_classification(self):
        counts = [0, 0, 0]
        dataset_classifications = [self.train_entities, self.validate_entities, self.test_entities]
        context_head_classifications = [self.train_head_entities, self.validate_head_entities, self.test_head_entities]
        context_tail_classifications = [self.train_tail_entities, self.validate_tail_entities, self.test_tail_entities]
        context_both_classifications = [self.train_both_entities, self.validate_both_entities, self.test_both_entities]
        for entity in range(self.statistics["num_of_entities"]):
            for index in range(len(self.names)):
                if entity in self.head_relation_to_tails[index] or entity in self.tail_relation_to_heads[index]:
                    dataset_classifications[index][counts[index]] = entity
                    counts[index] += 1
                    if self.head_context_statistics[index][entity] > 0 and self.tail_context_statistics[index][entity] == 0:
                        context_head_classifications[index][entity] = None
                    if self.head_context_statistics[index][entity] == 0 and self.tail_context_statistics[index][entity] > 0:
                        context_tail_classifications[index][entity] = None
                    if self.head_context_statistics[index][entity] > 0 and self.tail_context_statistics[index][entity] > 0:
                        context_both_classifications[index][entity] = None
        self.statistics["num_of_train_entities"] = counts[0]
        self.statistics["num_of_validate_entities"] = counts[1]
        self.statistics["num_of_test_entities"] = counts[2]
        for index in range(len(self.names)):
            dump_data(dataset_classifications[index], self.output_path + "%s_entities.pickle" % self.names[index], self.log_path, "")
            dump_data(context_head_classifications[index], self.output_path + "%s_head_entities.pickle" % self.names[index], self.log_path, "")
            dump_data(context_tail_classifications[index], self.output_path + "%s_tail_entities.pickle" % self.names[index], self.log_path, "")
            dump_data(context_both_classifications[index], self.output_path + "%s_both_entities.pickle" % self.names[index], self.log_path, "")
        dump_data(self.statistics, self.output_path + "statistics.pickle", self.log_path, "")

    def context_sampling(self):
        for index in range(len(self.names)):
            num_of_entity = self.num_of_entities[index]
            entity_dict = self.entity_dicts[index]
            head_context_statistic = self.head_context_statistics[index]
            tail_context_statistic = self.tail_context_statistics[index]
            head_context_head = self.head_context_heads[index]
            head_context_relation = self.head_context_relations[index]
            tail_context_relation = self.tail_context_relations[index]
            tail_context_tail = self.tail_context_tails[index]
            entity_head = self.entity_heads[index]
            entity_head_relation = self.entity_head_relations[index]
            entity_tail_relation = self.entity_tail_relations[index]
            entity_tail = self.entity_tails[index]
            for entity_id in range(num_of_entity):
                entity = entity_dict[entity_id]
                num_of_head_context = head_context_statistic[entity]
                num_of_tail_context = tail_context_statistic[entity]
                if num_of_head_context > 0:
                    heads = head_context_head[entity]
                    relations = head_context_relation[entity]
                    sampled_ids = sampled_id_generation(0, num_of_head_context, self.head_context_size)
                    entity_head[entity] = torch.unsqueeze(torch.LongTensor([heads[_] for _ in sampled_ids]), 0)
                    entity_head_relation[entity] = torch.unsqueeze(torch.LongTensor([relations[_] for _ in sampled_ids]), 0)
                if num_of_tail_context > 0:
                    relations = tail_context_relation[entity]
                    tails = tail_context_tail[entity]
                    sampled_ids = sampled_id_generation(0, num_of_tail_context, self.tail_context_size)
                    entity_tail_relation[entity] = torch.unsqueeze(torch.LongTensor([relations[_] for _ in sampled_ids]), 0)
                    entity_tail[entity] = torch.unsqueeze(torch.LongTensor([tails[_] for _ in sampled_ids]), 0)
            name = self.names[index]
            dump_data(entity_head, self.output_path + name + "_context_head.pickle", self.log_path, "")
            dump_data(entity_head_relation, self.output_path + name + "_context_head_relation.pickle", self.log_path, "")
            dump_data(entity_tail_relation, self.output_path + name + "_context_tail_relation.pickle", self.log_path, "")
            dump_data(entity_tail, self.output_path + name + "_context_tail.pickle", self.log_path, "")

    def negative_sampling(self):
        for index in range(len(self.names)):
            name = self.names[index]
            num_of_entity = self.num_of_entities[index]
            entity_dict = self.entity_dicts[index]
            negative = self.negatives[index]
            for entity_id in range(num_of_entity):
                entity = entity_dict[entity_id]
                negative_entities = []
                sampled_entities = {}
                sampled_entity_count = 0
                while len(negative_entities) < self.negative_batch_size and sampled_entity_count < num_of_entity:
                    sampled_entity = entity_dict[sampled_id_generation(0, num_of_entity, 1)[0]]
                    while sampled_entity in sampled_entities:
                        sampled_entity = entity_dict[sampled_id_generation(0, num_of_entity, 1)[0]]
                    sampled_entities[sampled_entity] = None
                    sampled_entity_count += 1
                    if self.negative_or_not(entity, sampled_entity):
                        negative_entities.append(sampled_entity)
                if len(negative_entities) == 0:
                    sampled_ids = sampled_id_generation(0, num_of_entity, self.negative_batch_size)
                    for sampled_id in sampled_ids:
                        negative_entities.append(entity_dict[sampled_id])
                if len(negative_entities) < self.negative_batch_size:
                    sampled_ids = sampled_id_generation(0, len(negative_entities), self.negative_batch_size - len(negative_entities))
                    for sampled_id in sampled_ids:
                        negative_entities.append(negative_entities[sampled_id])
                negative[entity] = torch.unsqueeze(torch.LongTensor(negative_entities), 0)
            dump_data(negative, self.output_path + "%s_negatives.pickle" % name, self.log_path, "")

    def negative_or_not(self, entity, sampled_entity):
        neighbors = [self.train_head_relation_to_tail, self.train_tail_relation_to_head]
        for neighbor in neighbors:
            if entity in neighbor and sampled_entity in neighbor:
                for tmp_relation in neighbor[entity]:
                    if tmp_relation in neighbor[sampled_entity]:
                        for tmp_entity in neighbor[entity][tmp_relation]:
                            if tmp_entity in neighbor[sampled_entity][tmp_relation]:
                                return False
        return True

    def result_validation(self):
        log_text(self.log_path, "...... Result of Entity Classification ......")
        for name in self.names:
            log_text(self.log_path, load_data(self.output_path + "%s_entities.pickle" % name, self.log_path, ""))
            log_text(self.log_path, load_data(self.output_path + "%s_head_entities.pickle" % name, self.log_path, ""))
            log_text(self.log_path, load_data(self.output_path + "%s_tail_entities.pickle" % name, self.log_path, ""))
            log_text(self.log_path, load_data(self.output_path + "%s_both_entities.pickle" % name, self.log_path, ""))

        log_text(self.log_path, "...... Result of Context Sampling ......")
        for name in self.names:
            log_text(self.log_path, load_data(self.output_path + "%s_context_head.pickle" % name, self.log_path, ""))
            log_text(self.log_path, load_data(self.output_path + "%s_context_head_relation.pickle" % name, self.log_path, ""))
            log_text(self.log_path, load_data(self.output_path + "%s_context_tail_relation.pickle" % name, self.log_path, ""))
            log_text(self.log_path, load_data(self.output_path + "%s_context_tail.pickle" % name, self.log_path, ""))

        log_text(self.log_path, "...... Result of Negative Sampling ......")
        for name in self.names:
            log_text(self.log_path, load_data(self.output_path + "%s_negatives.pickle" % name, self.output_path, ""))

        log_text(self.log_path, "...... Other Results ......")
        log_text(self.log_path, load_data(self.output_path + "statistics.pickle", self.log_path, "statistics"))

if __name__ == "__main__":
    contextAndNegatives = ContextAndNegatives()