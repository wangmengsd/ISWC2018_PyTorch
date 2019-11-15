# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:   iswc_train_with_online_context_and_negatives_generation.py
   Description: this file is only for demonstrating the use of our online sampling method of context and negative entities (online_batch_generation.py).
                there is no bug of this code, and it is executable.
                but the code is not recommended due to low efficiency!
   Author:  Ruijie Wang (https://github.com/xjdwrj)
   date:    14 Nov. 2019
-------------------------------------------------
"""

import torch
import time
from tools.dataset import MyDataset
from torch.utils.data import DataLoader
from tools.log_text import log_text
from tools.pickle_funcs import load_data, dump_data
from online_batch_generation import BatchProcess
from iswc_model import Model


class Train:
    def __init__(self):
        self.dataset = "FB15k"
        self.input_path = "./datasets/%s/input/" % self.dataset
        self.output_path = "./datasets/%s/output/" % self.dataset
        self.result_path = "./datasets/%s/result/" % self.dataset
        self.log_path = "./logs/iswc_train_with_online_context_and_negatives_generation_on_%s.log" % self.dataset

        self.entity_dimension = 100
        self.relation_dimension = 100
        self.num_of_epochs = 1000
        self.batch_size = 256
        self.norm = 2
        self.learning_rate = 0.01
        self.head_context_size = 64
        self.tail_context_size = 32
        self.negative_batch_size = 64
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda:3"
        self.continue_learning = False
        self.output_freq = 20
        self.validation_batch_size = 128
        self.test_batch_size = 1  # must set to 1 if hit@n is to be computed precisely
        self.patience = 5
        self.early_stop_patience = 5
        self.n_of_hit = 10

        self.optimal_entity_embeddings = None
        self.optimal_relation_embeddings = None

        self.id_train_triples = None  # {"id_heads": [], "id_relations": [], "id_tails": []}
        self.id_validate_triples = None  # {"id_heads": [], "id_relations": [], "id_tails": []}
        self.id_test_triples = None  # {"id_heads": [], "id_relations": [], "id_tails": []}

        self.head_context_head = {}  # {entity: {context_id: head}}
        self.head_context_relation = {}  # {entity: {context_id: relation}}
        self.head_context_statistics = {}  # {entity: ...}

        self.tail_context_relation = {}  # {entity: {context_id: relation}}
        self.tail_context_tail = {}  # {entity: {context_id: tail}}
        self.tail_context_statistics = {}  # {entity: ...}

        self.statistics = None
        # {"num_of_train_triples": self.num_of_train_triples,
        #  "num_of_validate_triples": self.num_of_validate_triples,
        #  "num_of_test_triples": self.num_of_test_triples,
        #  "num_of_entities": self.num_of_entities,
        #  "num_of_relations": self.num_of_relations}
        self.num_of_entities = None
        self.num_of_relations = None
        self.num_of_validate_triples = None
        self.num_of_test_triples = None
        self.num_of_train_entities = None
        self.num_of_validate_entities = None
        self.num_of_test_entities = None

        self.train_entities = None
        self.train_head_entities = None
        self.train_tail_entities = None
        self.train_both_entities = None

        self.run_functions()

    def run_functions(self):
        log_text(self.log_path, "\r\n---------------------Start-------------------------")
        log_text(self.log_path, "dataset: %s" % self.dataset)
        log_text(self.log_path, "head_context_size: %d" % self.head_context_size)
        log_text(self.log_path, "tail_context_size: %d" % self.tail_context_size)
        log_text(self.log_path, "negative_batch_size: %d" % self.negative_batch_size)
        log_text(self.log_path, "number of epochs: %d" % self.num_of_epochs)
        log_text(self.log_path, "batch size: %d" % self.batch_size)
        log_text(self.log_path, "norm: %d" % self.norm)
        log_text(self.log_path, "learning rate: %f" % self.learning_rate)
        log_text(self.log_path, "device: %s" % self.device)
        log_text(self.log_path, "continue learning: %s" % self.continue_learning)
        log_text(self.log_path, "entity dimension: %d" % self.entity_dimension)
        log_text(self.log_path, "relation dimension: %d" % self.relation_dimension)
        log_text(self.log_path, "patience: %d" % self.patience)
        log_text(self.log_path, "early stop patience: %d" % self.early_stop_patience)
        log_text(self.log_path, "output frequency: %d" % self.output_freq)
        log_text(self.log_path, "validation batch size: %d" % self.validation_batch_size)
        log_text(self.log_path, "test batch size: %d" % self.test_batch_size)
        log_text(self.log_path, "hit@: %d" % self.n_of_hit)

        self.read_data()
        self.train()

        log_text(self.log_path, "---------------------End-------------------------")

    def read_data(self):
        self.id_train_triples = load_data(self.output_path + "id_train_triples.pickle", self.log_path, "self.id_train_triples")
        self.id_validate_triples = load_data(self.output_path + "id_valid_triples.pickle", self.log_path, "self.id_validate_triples")
        self.id_test_triples = load_data(self.output_path + "id_test_triples.pickle", self.log_path, "self.id_test_triples")
        self.statistics = load_data(self.output_path + "statistics.pickle", self.log_path, "self.statistics")
        self.num_of_entities, self.num_of_relations, self.num_of_validate_triples, self.num_of_test_triples = \
        self.statistics["num_of_entities"], self.statistics["num_of_relations"], self.statistics[
            "num_of_validate_triples"], self.statistics["num_of_test_triples"]
        self.num_of_train_entities, self.num_of_validate_entities, self.num_of_test_entities = \
        self.statistics["num_of_train_entities"], self.statistics["num_of_validate_entities"], self.statistics["num_of_test_entities"]
        self.head_context_head = load_data(self.output_path + "train_head_context_head.pickle", self.log_path, "self.head_context_head")
        self.head_context_relation = load_data(self.output_path + "train_head_context_relation.pickle", self.log_path, "self.head_context_relation")
        self.head_context_statistics = load_data(self.output_path + "train_head_context_statistics.pickle", self.log_path, "self.head_context_statistics")
        self.tail_context_relation = load_data(self.output_path + "train_tail_context_relation.pickle", self.log_path, "self.tail_context_relation")
        self.tail_context_tail = load_data(self.output_path + "train_tail_context_tail.pickle", self.log_path, "self.tail_context_tail")
        self.tail_context_statistics = load_data(self.output_path + "train_tail_context_statistics.pickle", self.log_path, "self.tail_context_statistics")

        self.train_entities = load_data(self.output_path + "train_entities.pickle", self.log_path, "self.train_entities")
        self.train_head_entities = load_data(self.output_path + "train_head_entities.pickle", self.log_path, "self.train_head_entities")
        self.train_tail_entities = load_data(self.output_path + "train_tail_entities.pickle", self.log_path, "self.train_tail_entities")
        self.train_both_entities = load_data(self.output_path + "train_both_entities.pickle", self.log_path, "self.train_both_entities")

    def train(self):
        entity_set = MyDataset(self.num_of_train_entities)
        entity_loader = DataLoader(entity_set, self.batch_size, True)
        batch_process = BatchProcess(
            self.train_entities,
            self.train_head_entities, self.train_tail_entities, self.train_both_entities,
            self.head_context_head, self.head_context_relation, self.head_context_statistics,
            self.tail_context_relation, self.tail_context_tail, self.tail_context_statistics,
            self.head_context_size, self.tail_context_size, self.num_of_train_entities, self.negative_batch_size, self.device)
        model = Model(self.result_path, self.log_path, self.entity_dimension, self.relation_dimension, self.num_of_entities, self.num_of_relations, self.norm, self.device)
        if self.continue_learning:
            model.input()
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), self.learning_rate)
        current_mean_rank = self.validate(model)
        log_text(self.log_path, "initial mean rank (validation): %f" % current_mean_rank)
        optimal_mean_rank = current_mean_rank
        self.optimal_entity_embeddings = model.entity_embeddings.weight.data.clone()
        self.optimal_relation_embeddings = model.relation_embeddings.weight.data.clone()

        patience_count = 0
        for epoch in range(self.num_of_epochs):
            epoch_loss = 0.
            count = 0
            for entity_id_batch in entity_loader:
                if count % 200 == 0:
                    print "%d batches processed " % count + time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(time.time()))
                count += 1
                model.normalize()
                optimizer.zero_grad()
                entity_id_batch = entity_id_batch.tolist()
                entity_batch = [self.train_entities[entity_id] for entity_id in entity_id_batch]
                head_loss, tail_loss, both_loss, batch_loss = 0., 0., 0., 0.
                head_batch, tail_batch, both_batch = batch_process.batch_classification(entity_batch)
                if len(head_batch) > 0:
                    head_head, head_relation = batch_process.head_context_process(head_batch)
                    negative_head_batch = batch_process.negative_batch_generation(head_batch)
                    head_batch = torch.LongTensor(head_batch)
                    head_loss = -1. * model(head_batch.to(self.device),
                                            head_head.to(self.device), head_relation.to(self.device),
                                            None, None,
                                            negative_head_batch.to(self.device))
                if len(tail_batch) > 0:
                    tail_relation, tail_tail = batch_process.tail_context_process(tail_batch)
                    negative_tail_batch = batch_process.negative_batch_generation(tail_batch)
                    tail_batch = torch.LongTensor(tail_batch)
                    tail_loss = -1. * model(tail_batch.to(self.device),
                                            None, None,
                                            tail_relation.to(self.device), tail_tail.to(self.device),
                                            negative_tail_batch.to(self.device))
                if len(both_batch) > 0:
                    both_head, both_head_relation = batch_process.head_context_process(both_batch)
                    both_tail_relation, both_tail = batch_process.tail_context_process(both_batch)
                    negative_both_batch = batch_process.negative_batch_generation(both_batch)
                    both_batch = torch.LongTensor(both_batch)
                    both_loss = -1. * model(both_batch.to(self.device),
                                            both_head.to(self.device), both_head_relation.to(self.device),
                                            both_tail_relation.to(self.device), both_tail.to(self.device),
                                            negative_both_batch.to(self.device))

                batch_loss += head_loss + tail_loss + both_loss
                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss
            log_text(self.log_path, "\r\nepoch " + str(epoch) + ": , loss: " + str(epoch_loss))
            current_mean_rank = self.validate(model)
            if current_mean_rank < optimal_mean_rank:
                log_text(self.log_path, "optimal average raw mean rank: " + str(optimal_mean_rank) + " -> " + str(current_mean_rank))
                patience_count = 0
                optimal_mean_rank = current_mean_rank
                self.optimal_entity_embeddings = model.entity_embeddings.weight.data.clone()
                self.optimal_relation_embeddings = model.relation_embeddings.weight.data.clone()
            else:
                patience_count += 1
                log_text(self.log_path, "early stop patience: " + str(self.early_stop_patience) + ", patience count: " + str(patience_count) + ", current rank: " + str(current_mean_rank) + ", best rank: " + str(optimal_mean_rank))
                if patience_count == self.patience:
                    if self.early_stop_patience == 1:
                        dump_data(self.optimal_entity_embeddings.to("cpu"), self.result_path + "optimal_entity_embedding.pickle", self.log_path, "self.optimal_entity_embeddings")
                        dump_data(self.optimal_relation_embeddings.to("cpu"), self.result_path + "optimal_relation_embedding.pickle", self.log_path, "self.optimal_relation_embeddings")
                        break
                    log_text(self.log_path, "learning rate: " + str(self.learning_rate) + " -> " + str(self.learning_rate / 2))
                    self.learning_rate = self.learning_rate / 2
                    model.entity_embeddings.weight.data = self.optimal_entity_embeddings.clone()
                    model.relation_embeddings.weight.data = self.optimal_relation_embeddings.clone()
                    optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
                    patience_count = 0
                    self.early_stop_patience -= 1
            if epoch % self.output_freq == 0:
                model.output()
                dump_data(self.optimal_entity_embeddings.to("cpu"), self.result_path + "optimal_entity_embedding.pickle", self.log_path, "self.optimal_entity_embeddings")
                dump_data(self.optimal_relation_embeddings.to("cpu"), self.result_path + "optimal_relation_embedding.pickle", self.log_path, "self.optimal_relation_embeddings")
        self.test(model)

    def validate(self, model):
        mean_rank = 0
        valid_dataset = MyDataset(self.num_of_validate_triples)
        valid_dataloader = DataLoader(valid_dataset, self.validation_batch_size, False)
        for valid_batch in valid_dataloader:
            mean_rank += model.validate(torch.tensor([self.id_validate_triples["id_heads"][index] for index in valid_batch]).to(self.device),
                                        torch.tensor([self.id_validate_triples["id_relations"][index] for index in valid_batch]).to(self.device),
                                        torch.tensor([self.id_validate_triples["id_tails"][index] for index in valid_batch]).to(self.device))
        return mean_rank/self.num_of_validate_triples

    def test(self, model):
        train_triple_tensor = load_data(self.output_path + "train_triple_tensor.pickle", self.log_path, "train_triple_tensor").to(self.device)
        test_dataset = MyDataset(self.num_of_test_triples)
        test_dataloader = DataLoader(test_dataset, self.test_batch_size, False)
        test_result = torch.zeros(4).to(self.device)  # [mean_rank, hit_n, filtered_mean_rank, filtered_hit_n]
        log_text(self.log_path, "number of test triples: %d" % self.num_of_test_triples)
        count = 0
        for test_batch in test_dataloader:
            if count % 1000 == 0:
                print "%d test triples processed" % count
            count += self.test_batch_size
            model.test_calc(self.n_of_hit, test_result, train_triple_tensor,
                                torch.tensor([self.id_test_triples["id_heads"][index] for index in test_batch]).to(self.device),
                                torch.tensor([self.id_test_triples["id_relations"][index] for index in test_batch]).to(self.device),
                                torch.tensor([self.id_test_triples["id_tails"][index] for index in test_batch]).to(self.device))
        log_text(self.log_path, "raw mean rank: %f" % (test_result[0].item() / float(self.num_of_test_triples)))
        log_text(self.log_path, "raw hit@%d: %f%%" % (self.n_of_hit, 100. * test_result[1].item() / float(2. * self.num_of_test_triples)))
        log_text(self.log_path, "filtered mean rank: %f" % (test_result[2].item() / float(self.num_of_test_triples)))
        log_text(self.log_path, "filtered hit@%d: %f%%" % (self.n_of_hit, 100. * test_result[3].item() / float(2. * self.num_of_test_triples)))


if __name__ == "__main__":
    train = Train()