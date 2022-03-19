from utils import *

if __name__ == '__main__':
    # Parse argumetns
    args = parse_args()

    # Read high-dimensional data
    data = get_data(args.path)

    # Embed data into 2D space.
    embedded_data = embed_data(data, method='tsne')

    # Cluster data
    positions, density_matrix, clustered_data = cluster_data(embedded_data)

    # Plot and save results
    show_clustered_data(args.path, density_matrix, clustered_data)