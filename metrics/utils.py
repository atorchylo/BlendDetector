from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def draw_confusion(targets, predictions, num_cls):
    confusion = confusion_matrix(targets, predictions)
    fig = plt.figure()
    sns.heatmap(confusion, annot=True, cbar=False,
                xticklabels=range(1, num_cls + 1),
                yticklabels=range(1, num_cls + 1))
    plt.ylabel('Predicted class')
    plt.xlabel('Actual class')
    return fig