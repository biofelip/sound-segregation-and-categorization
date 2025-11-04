import cuml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.patches as patches
import pandas as pd

def generate_clusters_gpu( distances, u_map_ncomp, u_map_nneigh, u_map_min_dist, 
                      cluster_min_cluster_size, cluster_epsilon,
                     features,save_plot=False, save_pkl=False,
                    plot_embedding=False, plot_clusters=False):
    RANDOM_SEED = 20210105
    # labels_to_exclude = ['boat_sound', 'boat_noise', 'water_movement', 'boat_operations',
    #                      'electronic_noise', 'interference', 'voice', 'out_of_water', 'deployment']
    # if features.shape[0] == 0:
    #     features = ds.encode_clap(labels_to_exclude=labels_to_exclude, max_duration=3)
    original_features = features.copy()

    # Cluster the features
    if 'label' in features.columns:
        features = features.drop(columns=['label'])
    #features = features.loc[features.duration > 0.3]

    features['max_freq'] = features['max_freq'] / 12000
    features['min_freq'] = features['min_freq'] / 12000
    features['bandwidth'] = features['bandwidth'] / 12000
    features['duration'] = features['duration'] / 10

    features = features.drop(columns=['max_freq', 'min_freq', 'bandwidth', 'duration'])

    # Dimension reduction

    
    umap_box = cuml.manifold.UMAP(n_components=2, n_neighbors=u_map_nneigh, 
                         min_dist=u_map_min_dist, metric='euclidean',random_state=RANDOM_SEED)
    umap_box.fit(features)
    embedding = umap_box.transform(features).to_numpy()

    # Plot the embedding
    if plot_embedding:
        ax = sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1],
                            s=10, alpha=0.9,
                            legend=False)
        plt.xlabel('UMAP x')
        plt.ylabel('UMAP y')
        plt.savefig('umap2d.png')

    # Clustering
    #np.random.seed(RANDOM_SEED)
    hdbscan_model = cuml.cluster.hdbscan.HDBSCAN(cluster_selection_epsilon=cluster_epsilon,
                                    min_cluster_size=cluster_min_cluster_size, metric='euclidean',
                                    )
    clusterer = hdbscan_model.fit(embedding)
    clusters = clusterer.labels_

    # # Plot the clusters withou convex hull (original)
    # noise_mask = clusters == -1
    # clusters_array = np.arange(len(np.unique(clusters)) - 1)

    # ax = sns.scatterplot(x=embedding[noise_mask, 0], y=embedding[noise_mask, 1],
    #                      s=20, alpha=0.9,
    #                      legend=False, color='gray')
    # g = sns.scatterplot(x=embedding[~noise_mask, 0], y=embedding[~noise_mask, 1], s=45,
    #                     hue=clusters[~noise_mask].astype(str), hue_order=clusters_array.astype(str),
    #                     legend=True, ax=ax)
    # # Plot the cluster number
    # for c in clusters_array:
    #     embeddings_c = embedding[clusters == c]
    #     x, y = embeddings_c.mean(axis=0)
    #     plt.text(x, y, str(c))
    # plt.xlabel('UMAP x')
    # plt.ylabel('UMAP y')
    # g.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
    # plt.savefig('clusters.png')
    # plt.show()

    # Plot the clusters with convex hull
    noise_mask = clusters == -1
    clusters_array = np.arange(len(np.unique(clusters)) - 1)
    variances=embedding.var(axis=0)
    top_2=variances.argsort()[-2:][::-1]
    embedding=embedding[:,top_2]
    if plot_clusters:
        ax = sns.scatterplot(x=embedding[noise_mask, 0], y=embedding[noise_mask, 1],
                            s=20, alpha=0.9,
                            legend=False, color='gray')
        g = sns.scatterplot(x=embedding[~noise_mask, 0], y=embedding[~noise_mask, 1], s=45,
                            hue=clusters[~noise_mask].astype(str), hue_order=clusters_array.astype(str),
                            legend=True, ax=ax)

        # Add convex hulls around clusters
        for c in clusters_array:
            embeddings_c = embedding[clusters == c]
            
            # Only draw convex hull if we have at least 3 points
            if len(embeddings_c) >= 3:
                # variances=embeddings_c.var(axis=0)
                # top_2=variances.argsort()[-2:][::-1]
                # embeddings_c=embeddings_c[:,:2]
                try:
                    hull = ConvexHull(embeddings_c)
                    # Get the vertices of the convex hull
                    hull_points = embeddings_c[hull.vertices]
                    
                    # Create a polygon patch
                    polygon = patches.Polygon(hull_points, linewidth=2, edgecolor='black', 
                                            facecolor='none', alpha=0.7, linestyle='--')
                    ax.add_patch(polygon)
                except:
                    # Skip if convex hull cannot be computed (e.g., collinear points)
                    pass
            
            # Plot the cluster number
            
            x, y = embeddings_c[:, :2].mean(axis=0)
            plt.text(x, y, str(c), fontsize=12, fontweight='bold', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        plt.xlabel('UMAP x')
        plt.ylabel('UMAP y')
        g.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
        if save_plot:
            plt.savefig('clusters.png', bbox_inches='tight', dpi=300)
        plt.show()

    original_features['clusters'] = clusters
    # our sounds our to short to cut them by duration 
    # original_features.loc[original_features.duration > 0.3, 'clusters'].shape = clusters
    # original_features['clusters'] = clusters

    if save_pkl:
        pd.DataFrame(original_features).to_pickle('/mnt/f/Linnea/2024_high/dataset/clustering_test/features_with_clusters.pkl')
    return original_features