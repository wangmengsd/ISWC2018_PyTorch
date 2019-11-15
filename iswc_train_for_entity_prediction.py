# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:   iswc_train_for_entity_prediction.py
   Description: this code is to learn the embedding representation which could be utilized to predict the entity according to the context.
   Author:  Ruijie Wang (https://github.com/xjdwrj)
   date:    15 Nov. 2019
-------------------------------------------------
"""

import torch
from tools.print_gpu_status import PrintGPUStatus
from tools.log_text import log_text
from tools.pickle_funcs import load_data, dump_data
from tools.dataset import MyDataset
from torch.utils.data import DataLoader
from context_and_negatives_pre import ContextAndNegatives
from offline_batch_retrieve import OfflineBatchRetrieve
from iswc_model import Model


class Train:
    def __init__(self):
        self.dataset = "FB15k"
        self.input_path = "./datasets/%s/input/" % self.dataset
        self.output_path = "./datasets/%s/output/" % self.dataset
        self.result_path = "./datasets/%s/result/" % self.dataset
        self.log_path = "./logs/iswc_train_for_entity_prediction_on_%s.log" % self.dataset

        self.names = ["train", "valid", "test"]

        self.head_context_size = 128  # the number of sampled head context
        self.tail_context_size = 64  # the number of sampled tail context
        self.negative_batch_size = 300  # the number of sampled negative entities
        self.re_sampling_freq = 10  # re-sampling the context and negatives every self.re_sampling_freq epochs

        self.num_of_epochs = 1000
        self.batch_size = 128
        self.learning_rate = 0.002
        self.norm = 2
        self.entity_dimension = 100
        self.relation_dimension = 100
        self.continue_learning = False  # continue learning based on existing embedding vectors
        self.output_freq = 10
        self.n_of_hit = 10
        self.patience = 5
        self.early_stop_patience = 5
        self.validation_batch_size = 128
        self.test_batch_size = 128
        self.validation_freq = 1  # validation frequency

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda:1"

        self.context_and_negatives = None
        self.offline_batch_retrieve = None

        self.optimal_entity_embeddings = None
        self.optimal_relation_embeddings = None

        self.statistics = None
        # {"num_of_train_triples": self.num_of_train_triples,
        #  "num_of_validate_triples": self.num_of_validate_triples,
        #  "num_of_test_triples": self.num_of_test_triples,
        #  "num_of_entities": self.num_of_entities,
        #  "num_of_relations": self.num_of_relations,
        #  "num_of_train_entities": self.num_of_train_entities,
        #  "num_of_validate_entities": self.num_of_validate_entities,
        #  "num_of_test_entities": self.num_of_test_entities}
        self.num_of_entities = None
        self.num_of_relations = None
        self.num_of_train_entities = None
        self.num_of_validate_entities = None
        self.num_of_test_entities = None

        self.train_entities = None
        self.validate_entities = None
        self.test_entities = None

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

        log_text(self.log_path, "...... Context and Negatives Preparation ......")
        self.prepare_context_and_negatives()

        log_text(self.log_path, "...... Reading Data for ISWC Training ......")
        self.read_data()

        log_text(self.log_path, "...... ISWC Training ......")
        self.train()

        log_text(self.log_path, "---------------------End-------------------------")

    def prepare_context_and_negatives(self):
        self.context_and_negatives = ContextAndNegatives(self.names, self.dataset, self.head_context_size, self.tail_context_size, self.negative_batch_size)

    def read_data(self):
        self.statistics = load_data(self.output_path + "statistics.pickle", self.log_path, "self.statistics")
        self.num_of_entities, self.num_of_relations = self.statistics["num_of_entities"], self.statistics["num_of_relations"]
        self.num_of_train_entities, self.num_of_validate_entities, self.num_of_test_entities = \
            self.statistics["num_of_train_entities"], self.statistics["num_of_validate_entities"], self.statistics["num_of_test_entities"]
        self.train_entities = load_data(self.output_path + "train_entities.pickle", self.log_path, "self.train_entities")
        self.validate_entities = load_data(self.output_path + "valid_entities.pickle", self.log_path, "self.validate_entities")
        self.test_entities = load_data(self.output_path + "test_entities.pickle", self.log_path, "self.test_entities")

    def train(self):
        model = Model(self.result_path, self.log_path, self.entity_dimension, self.relation_dimension, self.num_of_entities, self.num_of_relations, self.norm, self.device)
        if self.continue_learning:
            model.input()
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), self.learning_rate)
        PrintGPUStatus.print_gpu_status("after the initialization of model")

        self.offline_batch_retrieve = OfflineBatchRetrieve(self.names, self.dataset)

        current_validate_loss = self.validate(model)
        log_text(self.log_path, "initial loss (validation): %f" % current_validate_loss)
        optimal_validate_loss = current_validate_loss
        self.optimal_entity_embeddings = model.entity_embeddings.weight.data.clone()
        self.optimal_relation_embeddings = model.relation_embeddings.weight.data.clone()

        entity_set = MyDataset(self.num_of_train_entities)
        entity_loader = DataLoader(entity_set, self.batch_size, True)
        patience_count = 0
        for epoch in range(self.num_of_epochs):
            epoch_loss = 0.
            if epoch != 0 and epoch % self.re_sampling_freq == 0:
                self.context_and_negatives.re_sampling()
                self.offline_batch_retrieve.re_read_context_and_negatives()
            for entity_id_batch in entity_loader:
                model.normalize()
                optimizer.zero_grad()
                entity_batch = [self.train_entities[entity_id.item()] for entity_id in entity_id_batch]
                head_batch, tail_batch, both_batch = self.offline_batch_retrieve.batch_classification("train", entity_batch)
                batch_loss = self.loss_compute("train", model, head_batch, tail_batch, both_batch)
                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss
            log_text(self.log_path, "\r\nepoch " + str(epoch) + ": , loss: " + str(epoch_loss))
            if epoch % self.validation_freq == 0:
                current_validate_loss = self.validate(model)
                if current_validate_loss < optimal_validate_loss:
                    log_text(self.log_path, "optimal validate loss: " + str(optimal_validate_loss) + " -> " + str(current_validate_loss))
                    patience_count = 0
                    optimal_validate_loss = current_validate_loss
                    self.optimal_entity_embeddings = model.entity_embeddings.weight.data.clone()
                    self.optimal_relation_embeddings = model.relation_embeddings.weight.data.clone()
                else:
                    patience_count += 1
                    log_text(self.log_path, "early stop patience: " + str(self.early_stop_patience) + ", patience count: " + str(patience_count) + ", current validate loss: " + str(current_validate_loss) + ", optimal validate loss: " + str(optimal_validate_loss))
                    if patience_count == self.patience:
                        if self.early_stop_patience == 1:
                            dump_data(self.optimal_entity_embeddings.to("cpu"),
                                      self.result_path + "optimal_entity_embedding.pickle", self.log_path,
                                      "self.optimal_entity_embeddings")
                            dump_data(self.optimal_relation_embeddings.to("cpu"),
                                      self.result_path + "optimal_relation_embedding.pickle", self.log_path,
                                      "self.optimal_relation_embeddings")
                            break
                        log_text(self.log_path, "learning rate: " + str(self.learning_rate) + " -> " + str(self.learning_rate / 2))
                        self.learning_rate = self.learning_rate / 2
                        model.entity_embeddings.weight.data = self.optimal_entity_embeddings.clone()
                        model.relation_embeddings.weight.data = self.optimal_relation_embeddings.clone()
                        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
                        patience_count = 0
                        self.early_stop_patience -= 1
            if (epoch + 1) % self.output_freq == 0:
                model.output()
                dump_data(self.optimal_entity_embeddings.to("cpu"), self.result_path + "optimal_entity_embedding.pickle", self.log_path, "self.optimal_entity_embeddings")
                dump_data(self.optimal_relation_embeddings.to("cpu"), self.result_path + "optimal_relation_embedding.pickle", self.log_path, "self.optimal_relation_embeddings")
        print "test loss: %f" % self.test(model)

    def loss_compute(self, name, model, head_batch, tail_batch, both_batch):
        head_loss, tail_loss, both_loss = 0., 0., 0.
        if len(head_batch) > 0:
            head_head, head_relation = self.offline_batch_retrieve.head_context_retrieve(name, head_batch)
            negative_head_batch = self.offline_batch_retrieve.negative_retrieves(name, head_batch)
            head_batch = torch.LongTensor(head_batch)
            head_loss = -1. * model(head_batch.to(self.device),
                                    head_head.to(self.device), head_relation.to(self.device),
                                    None, None,
                                    negative_head_batch.to(self.device))
        if len(tail_batch) > 0:
            tail_relation, tail_tail = self.offline_batch_retrieve.tail_context_retrieve(name, tail_batch)
            negative_tail_batch = self.offline_batch_retrieve.negative_retrieves(name, tail_batch)
            tail_batch = torch.LongTensor(tail_batch)
            tail_loss = -1. * model(tail_batch.to(self.device),
                                    None, None,
                                    tail_relation.to(self.device), tail_tail.to(self.device),
                                    negative_tail_batch.to(self.device))
        if len(both_batch) > 0:
            both_head, both_head_relation = self.offline_batch_retrieve.head_context_retrieve(name, both_batch)
            both_tail_relation, both_tail = self.offline_batch_retrieve.tail_context_retrieve(name, both_batch)
            negative_both_batch = self.offline_batch_retrieve.negative_retrieves(name, both_batch)
            both_batch = torch.LongTensor(both_batch)
            both_loss = -1. * model(both_batch.to(self.device),
                                    both_head.to(self.device), both_head_relation.to(self.device),
                                    both_tail_relation.to(self.device), both_tail.to(self.device),
                                    negative_both_batch.to(self.device))
        return head_loss + tail_loss + both_loss

    def validate(self, model):
        validate_entity_set = MyDataset(self.num_of_validate_entities)
        validate_entity_loader = DataLoader(validate_entity_set, self.validation_batch_size, False)
        for entity_id_batch in validate_entity_loader:
            entity_batch = [self.validate_entities[entity_id.item()] for entity_id in entity_id_batch]
            head_batch, tail_batch, both_batch = self.offline_batch_retrieve.batch_classification("validate", entity_batch)
            return float(self.loss_compute("validate", model, head_batch, tail_batch, both_batch))

    def test(self, model):
        test_entity_set = MyDataset(self.num_of_test_entities)
        test_entity_loader = DataLoader(test_entity_set, self.test_batch_size, False)
        for entity_id_batch in test_entity_loader:
            entity_batch = [self.test_entities[entity_id.item()] for entity_id in entity_id_batch]
            head_batch, tail_batch, both_batch = self.offline_batch_retrieve.batch_classification("test", entity_batch)
            return float(self.loss_compute("test", model, head_batch, tail_batch, both_batch))


if __name__ == "__main__":
    train = Train()