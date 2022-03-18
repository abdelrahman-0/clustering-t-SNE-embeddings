import argparse
import os
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde
from skimage.segmentation import watershed, find_boundaries
from sklearn import preprocessing
from scipy.spatial import ConvexHull
from mpl_toolkits.axes_grid1 import make_axes_locatable


def parse_args():
    '''
    Parse command line arguments and return them.
    '''
    parser = argparse.ArgumentParser(description='Unsupervised clustering of high-dimensional points using t-SNE and the watershed algorithm.')
    parser.add_argument('-path', help='path to .csv data file. Each column in the .csv file should contain one attribute.', required=True)
    return parser.parse_args()

def get_data(path: str):
    '''
    Read .csv file and return it as a pandas DataFrame.
    '''
    assert path.endswith('.csv'), 'Invalid file format. Only .csv files are supported.'
    assert os.path.exists(path), 'File not found.'
    data = pd.read_csv(path)
    return data

def embed_data(data: pd.DataFrame, **kwargs):
    '''
    Embed high-dimensional data into a 2D space using t-SNE.
    The perplexity value can significantly affect the number of regions in the final plot.
    '''
    normalized_data = preprocessing.StandardScaler().fit_transform(data)
    model = TSNE(n_components=2, learning_rate='auto', init='pca', **kwargs)
    embedded_data = model.fit_transform(normalized_data)
    # plt.scatter(embedded_data[:, 0], embedded_data[:, 1], c='red', s=1)
    # plt.show()
    return embedded_data

def get_positions_from_data(data: np.array, resolution=500):
    '''
    Returns a mesh grid created from the minimum and maximum points of the data.
    '''
    xmin, xmax = data[:, 0].min(), data[:, 0].max()
    ymin, ymax = data[:, 1].min(), data[:, 1].max()
    X, Y = np.meshgrid(np.linspace(xmin, xmax, resolution), np.linspace(ymin, ymax, resolution))
    positions = np.array([X.ravel(), Y.ravel()])
    return positions, X.shape

def get_density_matrix(data: np.array):
    '''
    Returns a (resolution, resolution) 2D matrix containing the evaluations
    of a KDE model that was fitted using the data.
    '''
    kde_model = gaussian_kde(data.T)
    positions, shape = get_positions_from_data(data)
    density_matrix = np.reshape(kde_model(positions).T, shape)
    return positions, density_matrix

def get_cluster_labels(density_matrix: np.array):
    '''
    Cluster the points in the density matrix using the watershed algorithm.
    '''
    inverted_matrix = -density_matrix
    return watershed(inverted_matrix)

def cluster_data(data: np.array):
    '''
    Cluster embedded data points.
    '''
    assert len(data.shape)==2, 'Incorrect embedding shape.'
    assert data.shape[1]==2, 'Expected 2D embeddings to plot.'
    assert data.shape[0]>=1, 'Empty embeddings list.'
    positions, density_matrix = get_density_matrix(data)
    cluster_labels = get_cluster_labels(density_matrix)
    return positions, density_matrix, cluster_labels

def get_image_from_plot():
    '''
    Get RGB image from plot.
    '''
    plt.gca().invert_yaxis()
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.gca().margins(0)
    plt.gcf().canvas.draw()
    image = np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(plt.gcf().canvas.get_width_height()[::-1] + (3,))
    return image


def plot_convex_hull(positions, image, points):
    '''
    Mask out points outside the convex hull our 2D embeddings (for visualization purposes).
    '''
    convex_hull = ConvexHull(points)
    points_path = matplotlib.path.Path(points[convex_hull.vertices])
    mask = points_path.contains_points(positions.T).reshape(image.shape[:2])
    image[mask == False] = [1,1,1]
    plt.imshow(image)
    minx = points[:, 0].min()
    maxx = points[:, 0].max()
    miny = points[:, 1].min()
    maxy = points[:, 1].max()
    h, w = image.shape[:2]
    # Plot outline of hull
    for simplex in convex_hull.simplices:
        plt.plot((points[simplex, 0]-minx)*w/(maxx-minx), (points[simplex, 1]-miny)*h/(maxy-miny), c='black')
    return get_image_from_plot()

def show_clustered_data(data_path: str, positions: np.array, embedded_data: np.array, density_matrix: np.array, clustered_data: np.array):
    '''
    Plot and save embedded data as 2D points.
    '''
    plot_name = os.path.splitext(data_path)[0] + '.png'
    watershed_boundaries = find_boundaries(clustered_data, mode='inner')
    im = plt.imshow(density_matrix, cmap='plasma')
    overlayed_img = im.cmap(im.norm(density_matrix))[:, :, :3]
    overlayed_img[watershed_boundaries==1] = [0,0,0]
    final_plot = plot_convex_hull(positions, overlayed_img, embedded_data)
    plt.imsave(plot_name, final_plot)
    print('Plot saved at', plot_name, sep=' ')
    return