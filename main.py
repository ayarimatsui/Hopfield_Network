# 知能機械情報学 レポート課題
# main.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
from hopfield import *


# 記憶する元画像をn個作り出す
def create_images(n):
    images = []
    for i in range(n):
        # ランダムに5x5の２値画像を生成
        def gen_img():
            image = np.random.rand(5, 5)
            image[image > 0.5] = 1
            image[image <= 0.5] = -1
            if np.sum([np.array_equal(image, images[j]) for j in range(i - 1)]) >= 1:
                image = gen_img()
            else:
                pass
            return image
        img = gen_img()
        images.append(img)
    return images


# 類似度と正解したかどうか(正解->1, 不正解->0)を返す
def check(answer, recollected):
    match = (answer == recollected)  # 配列の中で各要素が一致した数を返す
    similarity = np.sum(match) / 25 * 100
    # 全ての要素が一致しているときは正解、そうでないときは不正解
    if similarity >= 100:
        complete_match = 1
    else:
        complete_match = 0

    return similarity, complete_match


# 画像にノイズを加える
# 2つめの引数は与えるノイズの割合
def add_noise(img, noise_rate):
    index = random.sample([i for i in range(25)], noise_rate)
    noise_added_img = np.ravel(np.copy(img))   # 一次元化
    noise_added_img[index] = -1 * noise_added_img[index]   # ランダムに選ばれた要素を反転
    noise_added_img = np.reshape(noise_added_img, (5, 5))
    return noise_added_img


# データを二次元化してaxにプロット
def plot_data(ax, data):
    dim = int(np.sqrt(len(data)))
    assert dim * dim == len(data)
    img = (data.reshape(dim, dim) + 1) / 2
    ax.imshow(img, cmap=cm.Greys_r, interpolation='nearest')


# 元画像、初期値の画像(ノイズが加わったもの)、想起された画像を描画
def plot(original, noise_added, recollected, figsize=(5, 7)):
    # 画像データの一次元化
    for i in range(len(original)):
        original[i] = np.ravel(original[i])
        noise_added[i] = np.ravel(noise_added[i])
        recollected[i] = np.ravel(recollected[i])

    fig, axes = plt.subplots(len(original), 3, figsize=figsize)
    for i, axrow in enumerate(axes):
        if i == 0:
            axrow[0].set_title('original')
            axrow[1].set_title('input')
            axrow[2].set_title('output')
        plot_data(axrow[0], original[i])
        plot_data(axrow[1], noise_added[i])
        plot_data(axrow[2], recollected[i])

        for ax in axrow:
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
    return fig


# 1種類の 5x5の2値(+1/-1)画像を覚えさせ，元画像にノイズ(5~20%の一通り)を加えた画像を初期値として想起する実験
def main1():
    num_of_trials = 100  # 試行回数は100回とする
    noise = int(25 * 0.12)   # 12%のノイズを加える
    similarity_sum = 0
    correct_sum = 0
    # 画像描画用
    original_list = []
    noise_added_list = []
    recollected_list = []
    for i in range(num_of_trials):
        hopfield = Hopfield_Network()
        image = create_images(1)
        hopfield.train(image)   # 元画像を記憶させる
        noise_added_img = add_noise(image, noise)
        recollected_img = hopfield.recollect(noise_added_img)
        similarity, correct = check(image, recollected_img)
        similarity_sum += similarity / num_of_trials
        correct_sum += correct / num_of_trials
        # 最後の5回の試行の時のみ、元画像、ノイズを加えた画像、想起された画像を表示したいのでリストに保存しておく
        if i >= num_of_trials - 5:
            original_list.append(image)
            noise_added_list.append(noise_added_img)
            recollected_list.append(recollected_img)
    # 描画
    fig = plot(original_list, noise_added_list, recollected_list)
    plt.text(-15, -27, 'Number of Types: 1   Noise Rate: 12%', fontsize=13)
    plt.text(-10, 6.5, 'similarity: {:.1f},     accuracy: {:.1f}'.format(similarity_sum, correct_sum * 100))
    plt.savefig("figures/figure1.png")
    print("類似度 : {:.1f},   正答率 : {:.1f}".format(similarity_sum, correct_sum * 100))


# main1()と同条件で画像の種類を6程度まで徐々に増やして想起性能を調べる
def main2():
    num_of_images = [n for n in range(2, 7)]  # 画像の種類は2~6種類
    num_of_trials = 100  # 試行回数は100回とする
    noise = int(25 * 0.12)   # 12%のノイズを加える
    for n in num_of_images:
        similarity_sum = 0
        correct_sum = 0
        # 画像描画用
        original_list = []
        noise_added_list = []
        recollected_list = []
        for i in range(num_of_trials):
            hopfield = Hopfield_Network()
            image = create_images(n)  # n個の画像を生成
            hopfield.train(image)   # 元画像を記憶させる
            for j in range(n):
                target_image = image[j]
                noise_added_img = add_noise(target_image, noise)
                recollected_img = hopfield.recollect(noise_added_img)
                similarity, correct = check(target_image, recollected_img)
                similarity_sum += similarity / (n * num_of_trials)
                correct_sum += correct / (n * num_of_trials)
                # 最後の試行の時のみ、元画像、ノイズを加えた画像、想起された画像を表示したいのでリストに保存しておく
                if i == num_of_trials - 1:
                    original_list.append(target_image)
                    noise_added_list.append(noise_added_img)
                    recollected_list.append(recollected_img)
        # 最後の試行の時のみ、元画像、ノイズを加えた画像、想起された画像を表示
        fig = plot(original_list, noise_added_list, recollected_list, figsize=(5, n+2))
        plt.text(-10-n, 6.8, 'Number of Types: {}   Noise Rate: 12%'.format(n), fontsize=13)
        plt.text(-6, 5.5, 'similarity: {:.1f},     accuracy: {:.1f}'.format(similarity_sum, correct_sum * 100))
        plt.savefig("figures/figure2_{}.png".format(n))
        print("画像の種類 : {},   類似度 : {:.1f},   正答率 : {:.1f}".format(n, similarity_sum, correct_sum * 100))


# 画像が1~6種類(2種類と4種類)の場合について，ノイズを0%から100%まで徐々に増やして想起性能を調べる
def main3():
    num_of_images = [n for n in range(1, 7)]  # 画像の種類は1~6種類
    num_of_trials = 100  # 試行回数は100回とする
    noise_list = [i for i in range(0, 101, 4)]   # 0~100%まで4%ずつノイズを増やす
    total_sim_list = []
    total_acc_list = []
    for n in num_of_images:
        sim_list = []
        acc_list = []
        for noise_percentage in noise_list:
            similarity_sum = 0
            correct_sum = 0
            for i in range(num_of_trials):
                hopfield = Hopfield_Network()
                image = create_images(n)  # n個の画像を生成
                hopfield.train(image)   # 元画像を記憶させる
                for j in range(n):
                    target_image = image[j]
                    noise = int(25 * noise_percentage / 100)   # ノイズを加える
                    noise_added_img = add_noise(target_image, noise)
                    recollected_img = hopfield.recollect(noise_added_img)
                    similarity, correct = check(target_image, recollected_img)
                    similarity_sum += similarity / (n * num_of_trials)
                    correct_sum += correct / (n * num_of_trials)
            
            sim_list.append(similarity_sum)
            acc_list.append(correct_sum * 100)

        total_sim_list.append(sim_list)
        total_acc_list.append(acc_list)
    
    # グラフの描画
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    for i in range(len(num_of_images)):
        axes.flat[i].plot(noise_list, total_sim_list[i], label='similarity')
        axes.flat[i].plot(noise_list, total_acc_list[i], label='accuracy')
        axes.flat[i].set_xlabel("noise rate (%)")
        axes.flat[i].set_xlim(0, 100)
        axes.flat[i].set_ylabel("similarity and accuracy (%)")
        axes.flat[i].set_ylim(0, 101)
        axes.flat[i].legend(loc="best")
        axes.flat[i].set_title("Similarity, Accuracy and Noise (image types: {})".format(i+1))
    # 余白を設定
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    # グラフを保存
    plt.savefig("figures/figure3.png")
    


if __name__ == '__main__':
    #main1()
    #main2()
    main3()