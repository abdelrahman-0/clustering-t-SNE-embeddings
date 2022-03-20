import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, Isomap
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
from skimage.segmentation import watershed, find_boundaries
from sklearn import preprocessing

METHODS=['pca', 'tsne', 'isomap']
DENSITY_THRESHOLD = 4e-5
PERPLEXITY = 30
PAD_RATIO = 0.6

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
    dataframe = pd.read_csv(path)
    return dataframe


def apply_tsne(data: pd.DataFrame):
    '''
    Embed data using t-SNE.
    '''
    model = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=PERPLEXITY)
    embedded_data = model.fit_transform(data)
    return embedded_data


def apply_pca(data: pd.DataFrame):
    '''
    Embed data using PCA.
    '''
    model = PCA(n_components=2)
    embedded_data = model.fit_transform(data)
    return embedded_data


def apply_isomap(data: pd.DataFrame):
    '''
    Embed data using Isomap.
    '''
    model = Isomap(n_components=2, n_neighbors=15)
    embedded_data = model.fit_transform(data)
    return embedded_data


def embed_data(data: pd.DataFrame, method='tsne'):
    '''
    Embed high-dimensional data into a 2D space using t-SNE.
    The perplexity value can significantly affect the number of regions in the final plot.
    '''
    assert method in METHODS, 'embedding method not supported.'
    normalized_data = preprocessing.StandardScaler().fit_transform(data)
    if method == 'tsne':
        embedded_data = apply_tsne(normalized_data)
    elif method == 'pca':
        embedded_data = apply_pca(normalized_data)
    elif method == 'isomap':
        embedded_data = apply_isomap(normalized_data)
    return embedded_data


def get_positions_from_data(data: np.array, resolution=1000):
    '''
    Returns a mesh grid created from the minimum and maximum points of the data.
    '''
    xmin, xmax = data[:, 0].min(), data[:, 0].max()
    ymin, ymax = data[:, 1].min(), data[:, 1].max()
    xrange = xmax - xmin
    yrange = ymax - ymin
    xmin -= PAD_RATIO * xrange
    xmax += PAD_RATIO * xrange
    ymin -= PAD_RATIO * yrange
    ymax += PAD_RATIO * yrange
    X, Y = np.meshgrid(np.linspace(xmin, xmax, resolution), np.linspace(ymin, ymax, resolution))
    positions = np.array([X.ravel(), Y.ravel()])
    embedded_data_indices = np.zeros(data.shape)
    embedded_data_indices[:, 0] = (data[:, 0] - xmin) * resolution / ((2 * PAD_RATIO + 1) * xrange)
    embedded_data_indices[:, 1] = (data[:, 1] - ymin) * resolution / ((2 * PAD_RATIO + 1) * yrange)
    return positions, X.shape, embedded_data_indices.astype(int)


def get_density_matrix(data: np.array):
    '''
    Returns a (resolution, resolution) 2D matrix containing the evaluations
    of a KDE model that was fitted using the data.
    '''
    kde_model = gaussian_kde(data.T)
    positions, shape, embedded_data_indices = get_positions_from_data(data)
    density_matrix = np.reshape(kde_model(positions).T, shape)
    return positions, density_matrix, embedded_data_indices


def get_cluster_mask(density_matrix: np.array):
    '''
    Cluster the points in the density matrix using the watershed algorithm.
    '''
    inverted_matrix = -density_matrix
    mask = density_matrix > DENSITY_THRESHOLD
    cluster_mask = watershed(inverted_matrix, mask=mask)
    unique_clusters = np.unique(cluster_mask)
    print('Found {} unique clusters'.format(len(unique_clusters)-1 if 0 in unique_clusters else len(unique_clusters)))
    return cluster_mask


def cluster_data(data: np.array):
    '''
    Cluster embedded data points.
    '''
    assert len(data.shape)==2, 'Incorrect embedding shape.'
    assert data.shape[1]==2, 'Expected 2D embeddings to plot.'
    assert data.shape[0]>=1, 'Empty embeddings list.'
    positions, density_matrix, embedded_data_indices = get_density_matrix(data)
    cluster_mask = get_cluster_mask(density_matrix)
    return positions, density_matrix, cluster_mask, embedded_data_indices


def set_white_pixels_transparent(image):
    '''
    Set all white pixels as transparent pixels.
    '''
    plt.imshow(image[::-1], cmap='viridis', aspect='auto')
    h, w, _ = image.shape
    new_image = np.concatenate([image, np.full((h, w, 1), 1, dtype=image.dtype)], axis=-1)
    white = np.all(image == [1, 1, 1], axis=-1)
    new_image[white, -1] = 0
    return new_image


def get_cluster_labels(embedded_data_indices: np.array, clustered_data: np.array):
    '''
    Returns clusters labels that each 2D embedding belongs to.
    '''
    cluster_labels = clustered_data[embedded_data_indices[:, 0], embedded_data_indices[:, 1]]
    return cluster_labels


def export_clustered_data(data_path: str, embedded_data_indices: np.array, density_matrix: np.array, clustered_data: np.array):
    '''
    Plot and save embedded data as 2D points.
    '''
    plot_name = os.path.splitext(data_path)[0] + '_densitymap.png'
    cluster_csv_name = os.path.splitext(data_path)[0] + '_cluster_labels.csv'
    watershed_boundaries = find_boundaries(clustered_data, mode='thick')
    im = plt.imshow(density_matrix, cmap='Spectral_r', aspect='auto')
    final_plot = im.cmap(im.norm(density_matrix))[:, :, :3]
    final_plot[clustered_data == 0] = [1,1,1]
    final_plot[watershed_boundaries==1] = [0,0,0]
    final_plot = set_white_pixels_transparent(final_plot)
    plt.imsave(plot_name, final_plot[::-1])
    print('Plot saved at', plot_name, sep=' ')
    cluster_labels = get_cluster_labels(embedded_data_indices, clustered_data)
    cluster_labels_df = pd.DataFrame(cluster_labels, columns=['cluster_labels'])
    cluster_labels_df.to_csv(cluster_csv_name, index=False)
    print('Cluster memberships saved at', cluster_csv_name, sep=' ')
    return