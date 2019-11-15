"""
-------------------------------------------------
   File Name:   tsne_process.py
   Description: this code is to generate 2-dimensional tsne vectors based on learned embedding representations for visualization
   Author:  Ruijie Wang (https://github.com/xjdwrj)
   date:    15 Nov. 2019
-------------------------------------------------
"""

import torch
from tools.pickle_funcs import load_data
from tools.tsne import tsne


class ResultValidation:
    def __init__(self):
        self.dataset = "FB15k"
        self.result_path = "./datasets/%s/result/" % self.dataset
        self.log_path = "./logs/tsne_process_on_%s.log" % self.dataset
        tmp_embeddings = load_data(self.result_path + "entity_embeddings.pickle", self.log_path, "self.entity_embeddings.weight.data")
        self.num_of_entities = tmp_embeddings.size()[0]
        self.entity_dimension = tmp_embeddings.size()[1]
        self.entity_embeddings = torch.nn.Embedding(self.num_of_entities, self.entity_dimension)
        self.entity_embeddings.weight.data = tmp_embeddings

        self.run_funcs()

    def run_funcs(self):
        test_entity_embeddings = self.entity_embeddings.weight.data
        tsne_embeddings = tsne(test_entity_embeddings.detach().numpy(), 2, self.entity_dimension, 50.0)
        print len(tsne_embeddings)
        result = ""
        for x in range(len(tsne_embeddings)):
            result += str(tsne_embeddings[x][0]) + "\t" + str(tsne_embeddings[x][1]) + "\n"
        with open(self.result_path + "test_entity_vectors.txt", "w") as f:
            f.write(result)


if __name__ == "__main__":
    result_validation = ResultValidation()


