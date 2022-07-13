from sklearn.metrics import confusion_matrix
from btk.plot_utils import get_rgb_image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def accuracy(targets, predictions):
    return (targets == predictions).sum()/len(targets)


def draw_confusion(targets, predictions, num_cls, title=None):
    confusion = confusion_matrix(targets, predictions)
    fig = plt.figure()
    sns.heatmap(confusion, fmt='g', annot=True, cbar=False,
                xticklabels=range(0, num_cls),
                yticklabels=range(0, num_cls))
    plt.xlabel('# predicted objects')
    plt.ylabel('# true objects')
    if title is not None:
        plt.title(title)
    return fig


def draw_probability_spread(targets, probabilities, focus_on_n=4):
    fig, ax = plt.subplots(7, 1, figsize=(7, 20), sharex=True, gridspec_kw={'hspace': 1})
    for idx in range(7):
        ax[idx].hist(probabilities[targets == focus_on_n][:, idx], bins=100, range=(0, 1))
        ax[idx].set_ylabel('Count')
        ax[idx].set_xlabel('Probability')
        ax[idx].set_title(f'Probability of predicting {idx} when the true class is {focus_on_n}')
    return fig


def visualize_missclassified_example(targets, num_predicted, probabilities, dataset):
    idx = np.random.choice(np.where((targets != num_predicted))[0])
    batch, nums = dataset[idx // 4]
    img, num = batch[idx % 4], nums[idx % 4]
    img = get_rgb_image(img.numpy()[1:4])
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].imshow(img)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title(f"gri bands of {num} galaxies")
    ax[1].bar(range(7), probabilities[idx], tick_label=range(7), color='deepskyblue')
    ax[1].set_ylim(0, 1.05)
    ax[1].set_ylabel("Probability")
    ax[1].set_xlabel("Number of galaxies")
    ax[1].set_title(f"predicted probability distribution")
    return fig


def draw_average_prob_distribution(targets, num_predicted, probabilities):
    fig, ax = plt.subplots(7, 7, figsize=(10, 10), gridspec_kw={'wspace': 0, 'hspace': 0})
    for i in range(7):
        for j in range(7):
            selected_prob = probabilities[(targets == i) & (num_predicted == j)]

            if selected_prob.shape[0] > 0:
                ax[i, j].bar(range(7), selected_prob.mean(0), yerr=selected_prob.std(0), tick_label=range(7),
                             color='deepskyblue')

            ax[i, j].tick_params(axis="x", direction="in", pad=-15)
            ax[i, j].set_xlim(-0.74, 6.739)
            ax[i, j].set_xticks(range(7))
            ax[i, j].set_ylim(0, 1.05)
            ax[i, j].set_yticks([])
            ax[i, j].xaxis.set_ticks_position('none')

    fig.suptitle('Average probability distribution', fontsize=20)
    fig.supxlabel('# predicted objects', fontsize=20)
    fig.supylabel('# true objects', fontsize=20)
    plt.tight_layout()
    return fig


def draw_thresholding_graph(targets, num_predicted, probabilities):
    correctly_classified = []
    fraction_classified = []
    thresh_values = np.linspace(0, 1, 1000)
    for thresh in thresh_values:
        idx = probabilities.max(1) > thresh
        num_classified = targets[idx].shape[0]
        fraction_correctly_classified = (num_predicted[idx] == targets[
            idx]).sum() / num_classified if num_classified != 0 else 1
        correctly_classified.append(fraction_correctly_classified)
        fraction_classified.append(targets[idx].shape[0] / targets.shape[0])

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].plot(thresh_values, np.array(correctly_classified))
    ax[0].set_xlabel('Probability threshold value')
    ax[0].set_title('Precision value (fraction of correctly classified images)')
    ax[1].plot(thresh_values, np.array(fraction_classified))
    ax[1].set_title('Fraction of images that pass the threshold')
    ax[1].set_xlabel('Probability threshold value')
    return fig