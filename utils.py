import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde
from skimage.segmentation import watershed
from sklearn import preprocessing

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
    Embed high-dimensional data into a 2D space using t-SNE
    '''
    normalized_data = preprocessing.StandardScaler().fit_transform(data)
    model = TSNE(n_components=2, learning_rate='auto', init='pca', **kwargs)
    embedded_data = model.fit_transform(normalized_data)
    return embedded_data

def get_density_matrix(data: np.array, resolution=100):
    '''
    Returns a (resolution, resolution) 2D matrix containing the evaluations
    of a KDE model that was fitted using the data.
    '''
    xmin, xmax = data[:, 0].min(), data[:, 0].max()
    ymin, ymax = data[:, 1].min(), data[:, 1].max()
    kde_model = gaussian_kde(data.T)
    X, Y = np.meshgrid(np.linspace(xmin, xmax, resolution), np.linspace(ymin, ymax, resolution))
    positions = np.array([X.ravel(), Y.ravel()])
    estimated_data = np.reshape(kde_model(positions).T, X.shape)
    return estimated_data

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
    density_matrix = get_density_matrix(data, resolution=1000)
    cluster_labels = get_cluster_labels(density_matrix)
    return cluster_labels

def show_clustered_data(data: np.array, save=False):
    '''
    Plot and save embedded data as 2D points.
    '''
    plot_name = 'test.png'
    plt.imshow(data, cmap='plasma')
    if save:
        plt.savefig(plot_name)
        print('Saved plot at:', plot_name, sep=' ')
    plt.show()