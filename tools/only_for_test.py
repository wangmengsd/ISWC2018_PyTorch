import pickle, torch
from tools.uniform_sampling import sampled_id_generation

ids = sampled_id_generation(0, 3, 1)
print ids[0]
