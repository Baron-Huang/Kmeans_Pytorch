import numpy as np
import torch
from sklearn import datasets
import random

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

class K_means_torch():
    def __init__(self, k_nums = 3, sel_dis = 'l1', train_iters = 50, p = 1):
        super(K_means_torch, self).__init__()
        self.k_nums = k_nums
        self.sel_dis = sel_dis
        self.train_iters = train_iters
        self.p = p

    def l2_distance(self, x, y):
        return torch.sqrt((x - y).permute(1, 0) @ (x - y))

    def l1_distance(self, x, y):
        return torch.sum(torch.abs(x - y))

    def lmax_distance(self, x, y):
        return torch.max(torch.abs(x - y))

    def lp_distance(self, x, y, p):
        lp_sum = 0
        for i in range(int(x.shape[0])):
            lp_sum += (x[i] - y[i]) ** p
        lp_sum = torch.abs(lp_sum) ** (1 / p)
        return lp_sum

    def init_cluster_centre(self, x, k_num):
        y_shape = x.shape[1]

        clus_center = torch.zeros((1, y_shape))
        for i in range(k_num):
            clus_center_k = torch.zeros((1, 1))
            for j in range(y_shape):
                clus_center_inter = random.uniform(torch.max(x[:, j]), torch.min(x[:, j]))
                clus_center_inter = torch.reshape(clus_center_inter, (1, 1))
                clus_center_k = torch.cat((clus_center_k, clus_center_inter), dim=1)
            clus_center_k = clus_center_k[:, 1:]
            clus_center = torch.cat((clus_center, clus_center_k))

        clus_center = clus_center[1:, :]

        #clus_center = torch.randn((k_num, y_shape))

        return clus_center

    def assign_data_point(self, x, init_cluster_cen):
        assigned_set = {}
        for i in range(init_cluster_cen.shape[0]):
            assigned_set[str(i)] = []

        for i in range(x.shape[0]):
            cont_dis = torch.zeros((1, 1))
            for j in range(init_cluster_cen.shape[0]):
                if self.sel_dis == 'l2':
                    dis_value = self.l2_distance(x[i, :].reshape(4, 1), init_cluster_cen[j, :].reshape(4, 1))
                elif self.sel_dis == 'l1':
                    dis_value = self.l1_distance(x[i, :].reshape(4, 1), init_cluster_cen[j, :].reshape(4, 1))
                elif self.sel_dis == 'lp':
                    dis_value = \
                        self.lp_distance(x[i, :].reshape(4, 1), init_cluster_cen[j, :].reshape(4, 1), p=self.p)
                elif self.sel_dis == 'lmax':
                    dis_value = self.lmax_distance(x[i, :].reshape(4, 1), init_cluster_cen[j, :].reshape(4, 1))
                else:
                    pass
                cont_dis = torch.cat((cont_dis, dis_value.reshape(1, 1)))
            cont_dis = cont_dis[1:, :]
            max_id = torch.argmin(cont_dis).cpu().numpy()
            assigned_set[str(max_id)].append(i)
        return assigned_set

    def upgrade_cluster_centre(self, x, assigned_set):
        new_centre = torch.zeros((1, x.shape[1]))
        for i in range(len(assigned_set)):
            new_inter = torch.mean(x[assigned_set[str(i)], :], dim=0)
            new_centre = torch.cat((new_centre, new_inter.reshape(1, x.shape[1])))
        new_centre = new_centre[1:, :]
        return new_centre

    def forward(self, x):
        k = self.k_nums
        clus_center = self.init_cluster_centre(x, self.k_nums)
        for train_i in range(self.train_iters):
            assiged_set = self.assign_data_point(x, clus_center)
            new_centre = self.upgrade_cluster_centre(x, assiged_set)
            if torch.mean(new_centre) == torch.mean(clus_center):
                break
            else:
                clus_center = new_centre
        print('train_i:', train_i)
        return assiged_set

def l2_distance(x, y):
    return torch.sqrt((x - y).permute(1, 0) @ (x - y))

def l1_distance(x, y):
    return torch.sum(torch.abs(x - y))

def lmax_distance(x, y):
    return torch.max(torch.abs(x - y))

def lp_distance(x, y, p):
    lp_sum = 0
    for i in range(int(x.shape[0])):
        lp_sum += (x[i] - y[i])**p
    lp_sum = torch.abs(lp_sum)**(1/p)
    return lp_sum

def init_cluster_centre(x, k_num):
    y_shape = x.shape[1]
    clus_center = torch.zeros((1, y_shape))
    for i in range(k_num):
        clus_center_k = torch.zeros((1, 1))
        for j in range(y_shape):
            clus_center_inter = random.uniform(torch.max(x[:, j]), torch.min(x[:, j]))
            clus_center_inter = torch.reshape(clus_center_inter, (1, 1))
            clus_center_k = torch.cat((clus_center_k, clus_center_inter), dim=1)
        clus_center_k = clus_center_k[:, 1:]
        clus_center = torch.cat((clus_center, clus_center_k))
    clus_center = clus_center[1:, :]

    return clus_center

def assign_data_point(x, init_cluster_cen):
    assigned_set = {}
    for i in range(init_cluster_cen.shape[0]):
        assigned_set[str(i)] = []

    for i in range(x.shape[0]):
        cont_dis = torch.zeros((1, 1))
        for j in range(init_cluster_cen.shape[0]):
            dis_value = l2_distance(x[i, :].reshape(4, 1), init_cluster_cen[j, :].reshape(4, 1))
            cont_dis = torch.cat((cont_dis, dis_value))
        cont_dis = cont_dis[1:, :]
        max_id = torch.argmax(cont_dis).cpu().numpy()
        assigned_set[str(max_id)].append(i)
    return assigned_set

def upgrade_cluster_centre(x, assigned_set):
    new_centre = torch.zeros((1, x.shape[1]))
    for i in range(len(assigned_set)):
        new_inter = torch.mean(iris_feature[s[str(i)], :], dim=0)
        new_centre = torch.cat((new_centre, new_inter.reshape(1, x.shape[1])))
    new_centre = new_centre[1:, :]
    return new_centre



if __name__ == '__main__':
    iris_dataset = datasets.load_iris()
    print(iris_dataset)
    iris_feature = iris_dataset.data
    iris_label = iris_dataset.target
    iris_label = torch.tensor(iris_label)
    iris_feature = torch.tensor(iris_feature)
    test_x = iris_feature[0].reshape((4, 1))
    test_y = iris_feature[1].reshape((4, 1))
    x_0_max = torch.max(iris_feature[:, 0])
    x_0_min = torch.min(iris_feature[:, 0])
    print(iris_feature.shape)
    print(iris_label.shape)
    x = torch.ones((4, 1))
    y = torch.ones((4, 1)) + 1


    print(l1_distance(test_x, test_y))
    print(l2_distance(test_x, test_y))
    print(lmax_distance(test_x, test_y))
    print(lp_distance(test_x, test_y, 2000))

    print(x_0_max)
    print(x_0_min)

    print(random.uniform(x_0_max, x_0_min))
    print(random.uniform(x_0_max, x_0_min))

    x_c = init_cluster_centre(iris_feature, k_num=3)
    print(x_c)

    #print(iris_feature[[0, 1, 2, 3], :])

    s = assign_data_point(iris_feature, x_c)
    new_c = upgrade_cluster_centre(iris_feature, s)
    print(new_c)

    kmenas_model = K_means_torch()
    iris_feature[:, 0] = (iris_feature[:, 0] - torch.mean(iris_feature[:, 0]))\
                         / torch.std(iris_feature[:, 0])
    iris_feature[:, 1] = (iris_feature[:, 1] - torch.mean(iris_feature[:, 1])) \
                         / torch.std(iris_feature[:, 1])
    iris_feature[:, 2] = (iris_feature[:, 2] - torch.mean(iris_feature[:, 0])) \
                         / torch.std(iris_feature[:, 1])
    iris_feature[:, 3] = (iris_feature[:, 3] - torch.mean(iris_feature[:, 3])) \
                         / torch.std(iris_feature[:, 3])
    predict_matrix = kmenas_model.forward(iris_feature)
    print(predict_matrix['0'])
    print(predict_matrix['1'])
    print(predict_matrix['2'])
    print(len(predict_matrix['0']))
    print(len(predict_matrix['1']))
    print(len(predict_matrix['2']))


    pre_label = torch.zeros((iris_feature.shape[0], 1))
    pre_label[predict_matrix['0'], :] = torch.zeros((1, 1))
    pre_label[predict_matrix['1'], :] = torch.zeros((1, 1)) + 1
    pre_label[predict_matrix['2'], :] = torch.zeros((1, 1)) + 2

    count = 0
    for i in range(pre_label.shape[0]):
        if pre_label[i] == iris_label[i]:
            count += 1
    print(count / pre_label.shape[0])


