import numpy as np

f1_ft_1 = [0.1474, 0.19850000000000004, 0.1682, 0.2346, 0.18249999999999997]
f1_ft_2 = [0.4664, 0.4638, 0.4814, 0.4851, 0.4471]
f1_ft_4 = [0.989, 0.9718, 0.961, 0.9885, 0.9907]
f1_ft_8 = [0.9935, 0.994, 0.9927, 0.9943, 0.9913]

f1_pt_1 = [0.9437, 0.9508, 0.9614, 0.9482, 0.9418]
f1_pt_2 = [0.9887, 0.9815, 0.9781, 0.9861, 0.9872]
f1_pt_4 = [0.9921, 0.9949, 0.9928, 0.9937, 0.9928]
f1_pt_8 = [0.9935, 0.9946, 0.9964, 0.9951, 0.9955000000000002]


arr_mean = np.mean(f1_pt_8)
arr_std = np.std(f1_pt_8)
print("mean: ", arr_mean)
print("std: ", arr_std)