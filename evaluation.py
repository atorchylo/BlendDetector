from tqdm import tqdm
import os
import sep
import torch
import numpy as np


from src.resnet import ResNet
from metrics.utils import accuracy, draw_confusion, draw_probability_spread, \
    visualize_missclassified_example, draw_average_prob_distribution, draw_thresholding_graph
from config import DATA, TRAIN
from src.dataloaders import GetLuptitudes, BlendDataset, collate_fn
from torch.utils.data import DataLoader


def sep_detection(dataloader, sigma_noise=1.5):
    """Performs detection with SEP"""
    number_galaxies = []
    for imgs, _ in tqdm(dataloader, desc="running SE predictions"):
        for img in imgs:
            np_img = img.numpy()
            avg_image = np.mean(np_img, axis=0)

            # run Source Extractor
            bkg = sep.Background(avg_image)
            catalog = sep.extract(
              avg_image, sigma_noise, err=bkg.globalrms, segmentation_map=False
            )
            number_galaxies.append(len(catalog))
    return np.array(number_galaxies)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


def model_detection(dataloader, model, device):
    probabilities, targets = [], []
    model.eval()
    with torch.no_grad():
        for imgs, nums in tqdm(dataloader, desc="running model predictions"):
            input = GetLuptitudes()(imgs).to(device)
            output = model(input).cpu().numpy()
            prob = softmax(output)
            probabilities.extend(prob)
            targets.extend(nums)
    return np.array(probabilities), np.array(targets)


def run_evaluation(args):
    # get data
    dataset = BlendDataset(args.data_path, transform=None)
    dataloader = DataLoader(dataset, batch_size=args.batch_size//dataset.file_batch, collate_fn=collate_fn, shuffle=False)

    # initialize
    device = torch.device(args.device)
    data = torch.load(args.model_path, map_location=device)
    model = ResNet(TRAIN.in_ch, TRAIN.num_cls, TRAIN.num_layers)
    model.load_state_dict(data['model_state_dict'])
    model = model.to(device)

    # run detection
    model_probabilities, num_galaxies = model_detection(dataloader, model, device)
    model_predictions = model_probabilities.argmax(1)
    sep_predictions = sep_detection(dataloader, sigma_noise=1.5)

    # visualize
    acc1 = accuracy(num_galaxies, model_predictions)
    print(f"[Accuracy] Our Model: {acc1:.3f}")
    acc2 = accuracy(num_galaxies, sep_predictions)
    print(f"[Accuracy] Source Extractor: {acc2:.3f}")
    print("[Confusion] Saving to " + str(os.path.join(args.save_path, 'model_confusion.png')))
    fig1 = draw_confusion(num_galaxies, model_predictions, TRAIN.num_cls, "ResNet detection")
    fig1.savefig(os.path.join(args.save_path, 'model_confusion.png'))
    print("[Confusion] Saving to " + str(os.path.join(args.save_path, 'sep_confusion.png')))
    fig2 = draw_confusion(num_galaxies, sep_predictions, TRAIN.num_cls, "Source Extractor detection")
    fig2.savefig(os.path.join(args.save_path, 'sep_confusion.png'))
    print("[Prob Spread] Saving to " + str(os.path.join(args.save_path, 'probability_spread.png')))
    fig3 = draw_probability_spread(num_galaxies, model_probabilities)
    fig3.savefig(os.path.join(args.save_path, 'probability_spread.png'))
    print("[Prob Spread] Saving to " + str(os.path.join(args.save_path, 'probability_spread.png')))
    fig3 = draw_probability_spread(num_galaxies, model_probabilities)
    fig3.savefig(os.path.join(args.save_path, 'probability_spread.png'))
    print("[Prob Average] Saving to " + str(os.path.join(args.save_path, 'probability_average.png')))
    fig4 = draw_average_prob_distribution(num_galaxies, model_predictions, model_probabilities)
    fig4.savefig(os.path.join(args.save_path, f'probability_average.png'))
    print("[Prob Thresholding] Saving to " + str(os.path.join(args.save_path, 'probability_thresh.png')))
    fig5 = draw_thresholding_graph(num_galaxies, model_predictions, model_probabilities)
    fig5.savefig(os.path.join(args.save_path, f'probability_thresh.png'))
    print("[Example Distribution] Saving to " + str(os.path.join(args.save_path, 'example_<x>.png')))
    for i in range(1, 10):
        fig = visualize_missclassified_example(num_galaxies, model_predictions, model_probabilities, dataset)
        fig.savefig(os.path.join(args.save_path, f'example_{i}.png'))



if __name__ == '__main__':
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    parser = ArgumentParser(description=(
        """
        Performs evaluation of a Resnet model and compares that with the source extractor performance 
        """), formatter_class=RawDescriptionHelpFormatter)

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model to evaluate")

    parser.add_argument("--data_path", type=str, default=TRAIN.test_data_path,
                        help="Path to the data to use for evaluation")

    parser.add_argument("--save_path", type=str, default=TRAIN.figures_path,
                        help="Where to save resultant visualizations")

    parser.add_argument("--device", type=str, default=TRAIN.device,
                        help="Device for computation: either 'cuda' or 'cpu'.  Default: cuda")

    parser.add_argument("--batch_size", type=int, default=TRAIN.batch_size,
                        help="Batch size for the input to the neural network: Default: 16")


    args = parser.parse_args()
    run_evaluation(args)