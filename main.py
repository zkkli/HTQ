import argparse
import numpy as np
import torchvision.models as models
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sensitivity import *
from pareto_frontier import *
from utils import *


def get_args_parser():
    parser = argparse.ArgumentParser(description="PSAQ-ViT", add_help=False)
    parser.add_argument("--model", default="resnet18",
                        help="model")
    parser.add_argument('--dataset', default="imagenet",
                        choices=['cifar10', 'imagenet'],
                        help='dataset')
    parser.add_argument('--dataset_path', default="/Path/to/Dataset/",
                        help='path to dataset')
    return parser


def main():
    # build model and dataset
    print('Load the model and training data ...')
    #model = models.resnet18().cuda()
    model = getattr(models, args.model)(pretrained=False).cuda()
    train_loader = getTrainData(dataset=args.dataset, path=args.dataset_path)

    # compute sensitivity
    print('Compute connection sensitivity ...')
    scores, names = snip(model, train_loader)

    '''
    # show the sensitivity
    plt.figure(figsize=(12, 4))
    plt.axes(yscale='log')
    plt.plot(scores, '-o')
    plt.xticks(np.arange(len(names)), names, rotation=40)
    plt.tight_layout()
    plt.title('resnet18')
    plt.ylabel('sensitivity')
    plt.show()
    '''

    # compute the model complexity
    print('Compute the model complexity ...')
    parameters = compute_parameters(model)  # Model size
    MACs = compute_MACs(model)  # BitOps

    # pareto frontier
    print('Generate the pareto frontier ...')
    model = getattr(models, args.model)(pretrained=True).cuda()
    sizes, BOPs, sens = pareto_frontier(model, scores, parameters, MACs)

    # draw the 3D figure
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(np.array(sizes), np.array(BOPs), np.array(sens), c='#DC143C', alpha=0.2, linewidths=1.5)
    ax.set_zlabel('Total Perturbation')
    ax.set_ylabel('BOPs(G)')
    ax.set_xlabel('Model Size(MB)')
    ax.set_title('resnet18')
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.xaxis._axinfo['grid']['linestyle'] = '--'
    ax.yaxis._axinfo['grid']['linestyle'] = '--'
    ax.zaxis._axinfo['grid']['linestyle'] = '--'
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, 0.65, 1]))
    ax.view_init(elev=15, azim=-50)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser('HTQ', parents=[get_args_parser()])
    args = parser.parse_args()
    main()