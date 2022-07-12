from tqdm import tqdm
import os
import sep
import torch
import numpy as np
import matplotlib.pyplot as plt

from generate_dataset import get_generator
from src.resnet import ResNet
from src.dataloaders import GetLuptitudes
from metrics.utils import draw_confusion, accuracy
from config import DATA, TRAIN


def process_batch(batch):
    imgs = batch['blend_images']
    isolated = batch['isolated_images']
    meta = batch['blend_list']
    numGal = np.array([len(table) for table in meta])
    # add the blends with 0 galaxies by randomly erasing galaxy fluxes
    idx_0_galaxy = np.random.rand(imgs.shape[0]) < 1/(DATA.max_number + 1)
    imgs[idx_0_galaxy] = imgs[idx_0_galaxy] - isolated[idx_0_galaxy].sum(axis=1)
    numGal[idx_0_galaxy] = 0
    return imgs, numGal


def sep_detection(imgs, sigma_noise=1.5):
    """Performs detection with SEP"""
    number_galaxies = []
    for idx in range(imgs.shape[0]):
        image = imgs[idx]
        avg_image = np.mean(image, axis=0)

        # run Source Extractor
        bkg = sep.Background(avg_image)
        catalog = sep.extract(
          avg_image, sigma_noise, err=bkg.globalrms, segmentation_map=False
        )
        number_galaxies.append(len(catalog))
    return np.array(number_galaxies)


def model_detection(imgs, model, device, batch_size=16):
    predictions = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, imgs.shape[0], batch_size), desc='running model predictions'):
            batch = imgs[i: i+batch_size]
            luptitudes = GetLuptitudes()(batch)
            input = torch.from_numpy(luptitudes).float().to(device)
            output = model(input).argmax(1).cpu().numpy()
            predictions.extend(output)
    return np.array(predictions)


def accuracy_maxshift_plot(model, device, batch_size):
    model_accuracy = []
    sep_accuracy = []
    max_shifts = np.linspace(0, DATA.max_shift*2, 20)
    for shift in max_shifts:
        generator = get_generator(max_shift=shift, batch_size=1)
        batch = next(generator)
        imgs, num_galaxies = process_batch(batch)
        model_predictions = model_detection(imgs, model, device, batch_size=batch_size)
        sep_predictions = sep_detection(imgs, sigma_noise=1.5)
        model_accuracy.append(accuracy(num_galaxies, model_predictions))
        sep_accuracy.append(accuracy(num_galaxies, sep_predictions))
    fig, ax = plt.subplots()
    ax.plot(max_shifts, model_accuracy, 'o-', label="our model")
    ax.plot(max_shifts, sep_accuracy, 'o-', label="source extractor")
    ax.set_xlabel('Max separation between galaxies (arcsec)')
    ax.set_ylabel('Accuracy of the model')
    return fig


def run_evaluation(args):
    # generate data
    generator = get_generator(batch_size=args.n_blends)
    batch = next(generator)
    imgs, num_galaxies = process_batch(batch)

    # initialize
    device = torch.device(args.device)
    data = torch.load(args.model_path, map_location=device)
    model = ResNet(TRAIN.in_ch, TRAIN.num_cls, TRAIN.num_layers)
    model.load_state_dict(data['model_state_dict'])

    # run detection
    model_predictions = model_detection(imgs, model, device, batch_size=args.batch_size)
    sep_predictions = sep_detection(imgs, sigma_noise=1.5)

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

    # visualize performance of the model for more blended sources
    print("[Acc/maxshift] Saving to " + str(os.path.join(args.save_path, 'acc_vs_max_shift.png')))
    fig3 = accuracy_maxshift_plot(model, device, batch_size=args.n_blends)
    fig3.savefig(os.path.join(args.save_path, 'acc_vs_max_shift.png'))


if __name__ == '__main__':
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    parser = ArgumentParser(description=(
        """
        Performs evaluation of a Resnet model and compares that with the source extractor performance 
        """), formatter_class=RawDescriptionHelpFormatter)

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model to evaluate")

    parser.add_argument("--device", type=str, default='cuda',
                        help="Device for computation: either 'cuda' or 'cpu'.  Default: cuda")

    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for the input to the neural network: Default: 16")

    parser.add_argument("--n_blends", type=int, default=1000,
                        help="Totall number of blends to generate: Default: 1000")

    parser.add_argument("--save_path", type=str, default=None,
                        help="Where to save resultant visualizations")

    args = parser.parse_args()
    run_evaluation(args)