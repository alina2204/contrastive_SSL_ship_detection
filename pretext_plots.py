import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.offsetbox as osb
from matplotlib import rcParams as rcp

""" 
utils to do scatter plots
Code taken from https://github.com/lightly-ai/lightly tutorials """

# for resizing images to thumbnails
import torchvision.transforms.functional as functional
import torch


def create_filenames_embeddings(model, dataloader_test):
    embeddings = []
    filenames = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # disable gradients for faster calculations
    model.eval()
    with torch.no_grad():
        for i, (x, _, fnames) in enumerate(dataloader_test):
            # move the images to the gpu
            x = x.to(device)
            # embed the images with the pre-trained backbone
            y = model.backbone(x)
            y = y.squeeze()
            # store the embeddings and filenames in lists
            embeddings.append(y)
            filenames = filenames + list(fnames)
    # concatenate the embeddings and convert to numpy
    embeddings = torch.cat(embeddings, dim=0)
    embeddings = embeddings.cpu().numpy()
    return filenames, embeddings


def get_scatter_plot_with_thumbnails(embeddings_2d, path_to_data, filenames, ENABLE_VIS, save_dir):
    """Creates a scatter plot with image overlays.
    """
    # initialize empty figure and add subplot
    fig = plt.figure()
    fig.suptitle('Scatter Plot of the Sentinel-2 Dataset')
    ax = fig.add_subplot(1, 1, 1)
    # shuffle images and find out which images to show
    shown_images_idx = []
    shown_images = np.array([[1., 1.]])
    iterator = [i for i in range(embeddings_2d.shape[0])]
    np.random.shuffle(iterator)
    for i in iterator:
        # only show image if it is sufficiently far away from the others
        dist = np.sum((embeddings_2d[i] - shown_images) ** 2, 1)
        if np.min(dist) < 2e-3:
            continue
        shown_images = np.r_[shown_images, [embeddings_2d[i]]]
        shown_images_idx.append(i)

    # plot image overlays
    for idx in shown_images_idx:
        thumbnail_size = int(rcp['figure.figsize'][0] * 2.)
        path = os.path.join(path_to_data, filenames[idx])
        img = Image.open(path)
        # img = tifffile.imread(path).astype('float32')
        # h,w,_ = img.shape
        # output_rgb = np.zeros((h, w, 3))
        # i = 0
        # Bands = [1, 2, 3]
        # for CHAN in Bands:
        #     output_rgb[:, :, i] = img[:,:,CHAN]
        #     i += 1
        img = functional.resize(img, thumbnail_size)
        img_box = osb.AnnotationBbox(
            osb.OffsetImage(img, cmap=plt.cm.gray_r),
            embeddings_2d[idx],
            pad=0.2,
        )
        ax.add_artist(img_box)

    # set aspect ratio
    ratio = 1. / ax.get_data_ratio()
    ax.set_aspect(ratio, adjustable='box')
    if ENABLE_VIS is None:
        fig.show()
    else:
        fig.savefig(save_dir+'scatter_plot.png')


def get_image_as_np_array(filename: str):
    """Loads the image with filename and returns it as a numpy array.

    """
    img = Image.open(filename)
    return np.asarray(img)


def get_image_as_np_array_with_frame(filename: str, w: int = 5):
    """Returns an image as a numpy array with a black frame of width w.

    """
    img = get_image_as_np_array(filename)
    ny, nx, _ = img.shape
    # create an empty image with padding for the frame
    framed_img = np.zeros((w + ny + w, w + nx + w, 3))
    framed_img = framed_img.astype(np.uint8)
    # put the original image in the middle of the new one
    framed_img[w:-w, w:-w] = img
    return framed_img


def plot_nearest_neighbors_3x3(example_image: str, i: int, path_to_data, filenames, embeddings, ENABLE_VIS, save_dir):
    """Plots the example image and its eight nearest neighbors.

    """
    n_subplots = 9
    # initialize empty figure
    fig = plt.figure()
    fig.suptitle(f"Nearest Neighbor Plot {i + 1}")
    #
    matchers = [example_image]
    matching = [s for s in filenames if any(xs in s for xs in matchers)]
    example_idx = filenames.index(matching[0])
    # get distances to the cluster center
    distances = embeddings - embeddings[example_idx]
    distances = np.power(distances, 2).sum(-1).squeeze()
    # sort indices by distance to the center
    nearest_neighbors = np.argsort(distances)[:n_subplots]
    # show images
    for plot_offset, plot_idx in enumerate(nearest_neighbors):
        ax = fig.add_subplot(3, 3, plot_offset + 1)
        # get the corresponding filename
        fname = os.path.join(path_to_data, filenames[plot_idx])
        if plot_offset == 0:
            ax.set_title(f"Example Image")
            plt.imshow(get_image_as_np_array_with_frame(fname))
        else:
            plt.imshow(get_image_as_np_array(fname))
        # let's disable the axis
        plt.axis("off")
    if ENABLE_VIS is None:
        plt.show()
    else:
        plt.savefig(save_dir + 'KNN_{n}.png'.format(n=i))
