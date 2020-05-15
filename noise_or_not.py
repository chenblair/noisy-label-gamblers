import os
import numpy as np
import matplotlib.pyplot as plt
import json
import math

def get_path(a):
    # 50 epochs
    file_name = "results/noise_or_not/mnist_gmblers_symmetric_{:.2f}_9.9_10_noise.json".format(a)
    # Early Stopping
    # file_name = "results/noise_or_not/mnist_gmblers_symmetric_{:.2f}_9.9_1_noise.json".format(a)
    return file_name

plt.style.use('seaborn-deep')

for noise_rate in [0.2, 0.5, 0.8]:
    data = json.load(open(get_path(1 - noise_rate), 'r'))
    train_loss_clean = [math.log(i + 1e-10) for i in data["train_loss_clean"]]
    train_loss_corrupt = [math.log(i + 1e-10) for i in data["train_loss_corrupt"]]
    bins = np.linspace(min(train_loss_clean + train_loss_corrupt), max(train_loss_clean + train_loss_corrupt), 40)
    plt.hist([train_loss_clean, train_loss_corrupt], bins, label=['Clean', 'Corrupt'], weights=([1.0 / len(train_loss_clean) for i in train_loss_clean], [1.0 / len(train_loss_corrupt) for i in train_loss_corrupt]))
    plt.xlabel("Log Training Loss", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.savefig("analysis/train_loss_{:.1f}_50_epochs.png".format(1 - noise_rate))
    plt.clf()
    
    bins = np.linspace(0, 1, 20)
    weights=([1.0 / len(data["rejection_clean"]) for i in data["rejection_clean"]], [1.0 / len(data["rejection_corrupt"]) for i in data["rejection_corrupt"]])
    plt.hist([data["rejection_clean"], data["rejection_corrupt"]], bins, label=['Clean', 'Corrupt'], weights=weights)
    plt.xlabel("Rejection Score", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.savefig("analysis/rejection_{:.1f}_50_epochs.png".format(1 - noise_rate))
    plt.clf()