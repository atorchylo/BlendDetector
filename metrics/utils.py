from sklearn.metrics import confusion_matrix
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