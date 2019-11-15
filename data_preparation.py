# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:   data_preparation.py
   Description: this code takes the original knowledge graph data which should be in N-triple format as input
                and generates intermediate files required by downstream codes.
   Author:  Ruijie Wang (https://github.com/xjdwrj)
   date:    14 Nov. 2019
-------------------------------------------------
"""

import torch
from tools.pickle_funcs import load_data, dump_data
from tools.log_text import log_text


class DataPreparation:
    def __init__(self):
        self.print_results_for_validation = False

        self.dataset = "new_test"
        self.input_path = "./datasets/%s/input/" % self.dataset
        self.output_path = "./datasets/%s/output/" % self.dataset
        self.log_path = "./logs/data_preparation_on_%s.log" % self.dataset

        self.num_of_train_triples = 0
        self.num_of_validate_triples = 0
        self.num_of_test_triples = 0
        self.num_of_entities = 0
        self.num_of_relations = 0

        self.string_train_triples = {"heads": [], "relations": [], "tails": []}
        self.string_validate_triples = {"heads": [], "relations": [], "tails": []}
        self.string_test_triples = {"heads": [], "relations": [], "tails": []}

        self.id_train_triples = {"id_heads": [], "id_relations": [], "id_tails": []}
        self.id_validate_triples = {"id_heads": [], "id_relations": [], "id_tails": []}
        self.id_test_triples = {"id_heads": [], "id_relations": [], "id_tails": []}

        self.entity2id = {}  # {"entity_string": id}
        self.relation2id = {}  # {"relation_string": id}

        self.train_head_relation_to_tail = {}  # {head: {relation: [tails, ...]}}
        self.train_tail_relation_to_head = {}  # {tail: {relation: [heads, ...]}}
        self.validate_head_relation_to_tail = {}  # {head: {relation: [tails, ...]}}
        self.validate_tail_relation_to_head = {}  # {tail: {relation: [heads, ...]}}
        self.test_head_relation_to_tail = {}  # {head: {relation: [tails, ...]}}
        self.test_tail_relation_to_head = {}  # {tail: {relation: [heads, ...]}}

        self.train_head_context_head = {}  # {entity: {context_id: head}}
        self.train_head_context_relation = {}  # {entity: {context_id: relation}}
        self.train_head_context_statistics = {}  # {entity: ...}
        self.train_tail_context_relation = {}  # {entity: {context_id: relation}}
        self.train_tail_context_tail = {}  # {entity: {context_id: tail}}
        self.train_tail_context_statistics = {}  # {entity: ...}

        self.validate_head_context_head = {}  # {entity: {context_id: head}}
        self.validate_head_context_relation = {}  # {entity: {context_id: relation}}
        self.validate_head_context_statistics = {}  # {entity: ...}
        self.validate_tail_context_relation = {}  # {entity: {context_id: relation}}
        self.validate_tail_context_tail = {}  # {entity: {context_id: tail}}
        self.validate_tail_context_statistics = {}  # {entity: ...}

        self.test_head_context_head = {}  # {entity: {context_id: head}}
        self.test_head_context_relation = {}  # {entity: {context_id: relation}}
        self.test_head_context_statistics = {}  # {entity: ...}
        self.test_tail_context_relation = {}  # {entity: {context_id: relation}}
        self.test_tail_context_tail = {}  # {entity: {context_id: tail}}
        self.test_tail_context_statistics = {}  # {entity: ...}

        self.run_functions()

    def run_functions(self):
        log_text(self.log_path, "\r\n---------------------Start-------------------------")

        log_text(self.log_path, "...... Reading Data ......")
        self.read_dataset()

        log_text(self.log_path, "...... Head Relation to Tail and the Reverse ......")
        self.head_relation_to_tail_and_reverse()

        log_text(self.log_path, "...... Entity Context Extraction ......")
        self.context_process()

        log_text(self.log_path, "...... Other Operations ......")
        self.train_triple_tensor_generation()
        self.statistics()

        if self.print_results_for_validation:
            log_text(self.log_path, "...... Result Validation ......")
            self.result_validation()

        log_text(self.log_path, "---------------------End-------------------------")

    def read_dataset(self):
        names = ["train", "valid", "test"]
        string_triples = [self.string_train_triples, self.string_validate_triples, self.string_test_triples]
        id_triples = [self.id_train_triples, self.id_validate_triples, self.id_test_triples]
        num_of_triples = [0, 0, 0]
        for index in range(3):
            name = names[index]
            string_triple = string_triples[index]
            id_triple = id_triples[index]
            log_text(self.log_path, "reading file %s" % self.input_path + name + ".txt")
            with open(self.input_path + name + ".txt") as data_reader:
                tmp_line = data_reader.readline()
                while tmp_line and tmp_line not in ["\n", "\r\n", "\r"]:
                    tmp_head = tmp_line.split()[0]
                    tmp_relation = tmp_line.split()[1]
                    tmp_tail = tmp_line.split()[2]
                    string_triple["heads"].append(tmp_head)
                    string_triple["relations"].append(tmp_relation)
                    string_triple["tails"].append(tmp_tail)
                    id_triple["id_heads"].append(self.entity_id_generation(tmp_head))
                    id_triple["id_relations"].append(self.relation_id_generation(tmp_relation))
                    id_triple["id_tails"].append(self.entity_id_generation(tmp_tail))
                    num_of_triples[index] += 1
                    tmp_line = data_reader.readline()
                dump_data(string_triple, self.output_path + "string_%s_triples.pickle" % name, self.log_path, "string_%s_triples" % name)
                dump_data(id_triple, self.output_path + "id_%s_triples.pickle" % name, self.log_path, "id_%s_triples" % name)
        dump_data(self.entity2id, self.output_path + "entity2id.pickle", self.log_path, "self.entity2id")
        dump_data(self.relation2id, self.output_path + "relation2id.pickle", self.log_path, "self.relation2id")
        self.num_of_train_triples = num_of_triples[0]
        self.num_of_validate_triples = num_of_triples[1]
        self.num_of_test_triples = num_of_triples[2]

    def head_relation_to_tail_and_reverse(self):
        names = ["train", "valid", "test"]
        num_of_triples = [self.num_of_train_triples, self.num_of_validate_triples, self.num_of_test_triples]
        id_triples = [self.id_train_triples, self.id_validate_triples, self.id_test_triples]
        head_relation_to_tails = [self.train_head_relation_to_tail, self.validate_head_relation_to_tail, self.test_head_relation_to_tail]
        tail_relation_to_heads = [self.train_tail_relation_to_head, self.validate_tail_relation_to_head, self.test_tail_relation_to_head]
        for index in range(3):
            name = names[index]
            num_of_triple = num_of_triples[index]
            id_triple = id_triples[index]
            head_relation_to_tail = head_relation_to_tails[index]
            tail_relation_to_head = tail_relation_to_heads[index]
            for triple_id in range(num_of_triple):
                tmp_head = id_triple["id_heads"][triple_id]
                tmp_relation = id_triple["id_relations"][triple_id]
                tmp_tail = id_triple["id_tails"][triple_id]
                if tmp_head not in head_relation_to_tail:
                    head_relation_to_tail[tmp_head] = {tmp_relation: []}
                else:
                    if tmp_relation not in head_relation_to_tail[tmp_head]:
                        head_relation_to_tail[tmp_head][tmp_relation] = []
                head_relation_to_tail[tmp_head][tmp_relation].append(tmp_tail)
                if tmp_tail not in tail_relation_to_head:
                    tail_relation_to_head[tmp_tail] = {tmp_relation: []}
                else:
                    if tmp_relation not in tail_relation_to_head[tmp_tail]:
                        tail_relation_to_head[tmp_tail][tmp_relation] = []
                tail_relation_to_head[tmp_tail][tmp_relation].append(tmp_head)
            dump_data(head_relation_to_tail, self.output_path + "%s_head_relation_to_tail.pickle" % name, self.log_path, "head_relation_to_tail")
            dump_data(tail_relation_to_head, self.output_path + "%s_tail_relation_to_head.pickle" % name, self.log_path, "tail_relation_to_head")

    def statistics(self):
        log_text(self.log_path, "number of train triples: %d" % self.num_of_train_triples)
        log_text(self.log_path, "number of validate triples: %d" % self.num_of_validate_triples)
        log_text(self.log_path, "number of test triples: %d" % self.num_of_test_triples)
        log_text(self.log_path, "number of entities: %d" % self.num_of_entities)
        log_text(self.log_path, "number of relations: %d" % self.num_of_relations)
        statistics = {"num_of_train_triples": self.num_of_train_triples,
                      "num_of_validate_triples": self.num_of_validate_triples,
                      "num_of_test_triples": self.num_of_test_triples,
                      "num_of_entities": self.num_of_entities,
                      "num_of_relations": self.num_of_relations,
                      "num_of_train_entities": None,
                      "num_of_validate_entities": None,
                      "num_of_test_entities": None}
        dump_data(statistics, self.output_path + "statistics.pickle", self.log_path, "statistics")

    def context_process(self):
        names = ["train", "valid", "test"]
        head_relation_to_tails = [self.train_head_relation_to_tail, self.validate_head_relation_to_tail, self.test_head_relation_to_tail]
        tail_relation_to_heads = [self.train_tail_relation_to_head, self.validate_tail_relation_to_head, self.test_tail_relation_to_head]
        head_context_heads = [self.train_head_context_head, self.validate_head_context_head, self.test_head_context_head]
        head_context_relations = [self.train_head_context_relation, self.validate_head_context_relation, self.test_head_context_relation]
        head_context_statistics_es = [self.train_head_context_statistics, self.validate_head_context_statistics, self.test_head_context_statistics]
        tail_context_relations = [self.train_tail_context_relation, self.validate_tail_context_relation, self.test_tail_context_relation]
        tail_context_tails = [self.train_tail_context_tail, self.validate_tail_context_tail, self.test_tail_context_tail]
        tail_context_statistics_es = [self.train_tail_context_statistics, self.validate_tail_context_statistics, self.test_tail_context_statistics]
        for index in range(3):
            name = names[index]
            head_relation_to_tail = head_relation_to_tails[index]
            tail_relation_to_head = tail_relation_to_heads[index]
            head_context_head = head_context_heads[index]
            head_context_relation = head_context_relations[index]
            head_context_statistics = head_context_statistics_es[index]
            tail_context_relation = tail_context_relations[index]
            tail_context_tail = tail_context_tails[index]
            tail_context_statistics = tail_context_statistics_es[index]
            for entity in range(self.num_of_entities):
                num_of_head_context = 0
                head_context_head[entity] = {}
                head_context_relation[entity] = {}
                if entity in tail_relation_to_head:
                    for relation in tail_relation_to_head[entity]:
                        for head in tail_relation_to_head[entity][relation]:
                            head_context_head[entity][num_of_head_context] = head
                            head_context_relation[entity][num_of_head_context] = relation
                            num_of_head_context += 1
                head_context_statistics[entity] = num_of_head_context

                num_of_tail_context = 0
                tail_context_relation[entity] = {}
                tail_context_tail[entity] = {}
                if entity in head_relation_to_tail:
                    for relation in head_relation_to_tail[entity]:
                        for tail in head_relation_to_tail[entity][relation]:
                            tail_context_relation[entity][num_of_tail_context] = relation
                            tail_context_tail[entity][num_of_tail_context] = tail
                            num_of_tail_context += 1
                tail_context_statistics[entity] = num_of_tail_context

            dump_data(head_context_head, self.output_path + "%s_head_context_head.pickle" % name, self.log_path, "head_context_head")
            dump_data(head_context_relation, self.output_path + "%s_head_context_relation.pickle" % name, self.log_path, "head_context_relation")
            dump_data(head_context_statistics, self.output_path + "%s_head_context_statistics.pickle" % name, self.log_path, "head_context_statistics")
            dump_data(tail_context_relation, self.output_path + "%s_tail_context_relation.pickle" % name, self.log_path, "tail_context_relation")
            dump_data(tail_context_tail, self.output_path + "%s_tail_context_tail.pickle" % name, self.log_path, "tail_context_tail")
            dump_data(tail_context_statistics, self.output_path + "%s_tail_context_statistics.pickle" % name, self.log_path, "tail_context_statistics")

    def entity_id_generation(self, entity):
        if entity not in self.entity2id:
            self.entity2id[entity] = self.num_of_entities
            self.num_of_entities += 1
        return self.entity2id[entity]

    def relation_id_generation(self, relation):
        if relation not in self.relation2id:
            self.relation2id[relation] = self.num_of_relations
            self.num_of_relations += 1
        return self.relation2id[relation]

    def train_triple_tensor_generation(self):
        train_triple_tensor = torch.zeros(self.num_of_train_triples, 3)
        for index in range(self.num_of_train_triples):
            train_triple_tensor[index][0] = self.id_train_triples["id_heads"][index]
            train_triple_tensor[index][1] = self.id_train_triples["id_relations"][index]
            train_triple_tensor[index][2] = self.id_train_triples["id_tails"][index]
        dump_data(train_triple_tensor, self.output_path + "train_triple_tensor.pickle", self.log_path, "train_triple_tensor")

    def result_validation(self):
        names = ["train", "valid", "test"]
        log_text(self.log_path, "......Result of Reading Data......")
        for name in names:
            log_text(self.log_path, load_data(self.output_path + "string_%s_triples.pickle" % name, self.log_path, ""))
            log_text(self.log_path, load_data(self.output_path + "id_%s_triples.pickle" % name, self.log_path, ""))
        log_text(self.log_path, load_data(self.output_path + "entity2id.pickle", self.log_path, ""))
        log_text(self.log_path, load_data(self.output_path + "relation2id.pickle", self.log_path, ""))

        log_text(self.log_path, "......Result of Head Relation to Tail and Reserve......")
        for name in names:
            log_text(self.log_path, load_data(self.output_path + "%s_head_relation_to_tail.pickle" % name, self.log_path, ""))
            log_text(self.log_path, load_data(self.output_path + "%s_tail_relation_to_head.pickle" % name, self.log_path, ""))

        log_text(self.log_path, "......Result of Entity Context Extraction......")
        for name in names:
            log_text(self.log_path, load_data(self.output_path + "%s_head_context_head.pickle" % name, self.log_path, ""))
            log_text(self.log_path, load_data(self.output_path + "%s_head_context_relation.pickle" % name, self.log_path, ""))
            log_text(self.log_path, load_data(self.output_path + "%s_head_context_statistics.pickle" % name, self.log_path, ""))
            log_text(self.log_path, load_data(self.output_path + "%s_tail_context_relation.pickle" % name, self.log_path, ""))
            log_text(self.log_path, load_data(self.output_path + "%s_tail_context_tail.pickle" % name, self.log_path, ""))
            log_text(self.log_path, load_data(self.output_path + "%s_tail_context_statistics.pickle" % name, self.log_path, ""))

        log_text(self.log_path, "......Other Results......")
        log_text(self.log_path, load_data(self.output_path + "statistics.pickle", self.log_path, ""))
        log_text(self.log_path, load_data(self.output_path + "train_triple_tensor.pickle", self.log_path, ""))




if __name__ == "__main__":
    dataPrepare = DataPreparation()
