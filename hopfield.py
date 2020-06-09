# -*- coding: UTF-8 -*-
# hopfield.py
import numpy as np

class Hopfield_Network():

    def __init__(self):
        self.theta = np.zeros([5 * 5, 1])  # 閾値の初期値は全て0としておく
        self.weight = np.zeros([5 * 5, 5 * 5])  # 重みパラメータ初期化


    def train(self, train_list):
        N = len(train_list)
        for i in range(N):
            train_flatten = np.ravel(train_list[i]).reshape([5 * 5, 1])  # 一次元化
            self.weight += np.dot(train_flatten, train_flatten.T) / N
        np.fill_diagonal(self.weight, 0)  # 強制的に対角成分は、w_i_i = 0 とする


    def potential_energy(self, input_flatten):
        # potential energyを計算して返す
        V = -1 / 2 * input_flatten.T @ self.weight @ input_flatten + np.dot(self.theta.T, input_flatten)
        return V


    # 画像を想起
    def recollect(self, input, loop_num=1000):
        # 入力は5x5
        input_flatten = np.ravel(input)  # 入力を一次元に平坦化
        energy = self.potential_energy(input_flatten)  # エネルギーを計算
        for i in range(loop_num):
            input_flatten = np.sign(np.dot(self.weight, input_flatten) - self.theta)
            new_energy = self.potential_energy(input_flatten)
            # エネルギーが前と変わらなくなったら(収束したら)ループから抜ける
            if new_energy == energy:
                break
            else:
                energy = new_energy
        
        recollected_img = np.reshape(input_flatten, (5, 5))   # 二次元配列に戻す
        return recollected_img
