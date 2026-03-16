"""Clustering utilities for RAPTOR tree construction.

Based on:
- Original RAPTOR paper (Sarthi et al., 2024)
- LangChain cookbook implementation
"""

import logging
from typing import List, Optional, Tuple
import numpy as np
from sklearn.mixture import GaussianMixture
from tenacity import retry, stop_after_attempt, wait_fixed

logger = logging.getLogger(__name__)

RANDOM_SEED = 224


try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logger.warning("UMAP not available. Using PCA for dimensionality reduction.")


def reduce_dimensions(
    embeddings: np.ndarray,
    n_components: int = 10,
    n_neighbors: Optional[int] = None,
    metric: str = "cosine"
) -> np.ndarray:
    """Reduce dimensionality of embeddings using UMAP or PCA.
    
    Args:
        embeddings: Input embeddings (n_samples, n_features)
        n_components: Target dimensionality
        n_neighbors: Number of neighbors for UMAP
        metric: Distance metric for UMAP
        
    Returns:
        Reduced embeddings
    """
    n_samples = len(embeddings)
    
    # Adjust dimensions if needed
    n_components = min(n_components, n_samples - 1)
    
    if n_components < 2:
        return embeddings
    
    if UMAP_AVAILABLE:
        if n_neighbors is None:
            n_neighbors = max(2, int((n_samples - 1) ** 0.5))
        
        n_neighbors = min(n_neighbors, n_samples - 1)
        
        try:
            reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                n_components=n_components,
                metric=metric,
                random_state=RANDOM_SEED
            )
            return reducer.fit_transform(embeddings)
        except Exception as e:
            logger.warning(f"UMAP failed: {e}, falling back to PCA")
    
    # Fallback to PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    return pca.fit_transform(embeddings)


def get_optimal_clusters(
    embeddings: np.ndarray,
    max_clusters: int = 50,
    random_state: int = RANDOM_SEED
) -> int:
    """Determine optimal number of clusters using BIC.
    
    Args:
        embeddings: Input embeddings
        max_clusters: Maximum clusters to consider
        random_state: Random seed
        
    Returns:
        Optimal number of clusters
    """
    max_clusters = min(max_clusters, len(embeddings) - 1)
    max_clusters = max(1, max_clusters)
    
    if len(embeddings) <= 2:
        return 1
    
    n_clusters_range = range(1, max_clusters + 1)
    bics = []
    
    for n in n_clusters_range:
        try:
            gmm = GaussianMixture(
                n_components=n,
                random_state=random_state,
                covariance_type='full',
                max_iter=100
            )
            gmm.fit(embeddings)
            bics.append(gmm.bic(embeddings))
        except Exception as e:
            logger.warning(f"GMM failed for n={n}: {e}")
            bics.append(float('inf'))
    
    optimal_n = n_clusters_range[np.argmin(bics)]
    logger.info(f"Optimal clusters: {optimal_n} (BICs: {bics[:5]}...)")
    
    return optimal_n


def gmm_cluster(
    embeddings: np.ndarray,
    n_clusters: Optional[int] = None,
    threshold: float = 0.5,
    random_state: int = RANDOM_SEED
) -> Tuple[List[np.ndarray], int]:
    """Cluster embeddings using Gaussian Mixture Model.
    
    Args:
        embeddings: Input embeddings
        n_clusters: Number of clusters (auto-detected if None)
        threshold: Probability threshold for cluster membership
        random_state: Random seed
        
    Returns:
        Tuple of (cluster_assignments, actual_n_clusters)
    """
    if n_clusters is None:
        n_clusters = get_optimal_clusters(embeddings)
    
    if n_clusters >= len(embeddings):
        # Each point in its own cluster
        labels = [np.array([i]) for i in range(len(embeddings))]
        return labels, len(embeddings)
    
    try:
        gmm = GaussianMixture(
            n_components=n_clusters,
            random_state=random_state,
            covariance_type='full',
            max_iter=200,
            n_init=3
        )
        gmm.fit(embeddings)
        
        # Get probabilities for each point
        probs = gmm.predict_proba(embeddings)
        
        # Assign to clusters where probability > threshold
        # Points can belong to multiple clusters (soft clustering)
        labels = []
        for prob in probs:
            assigned = np.where(prob > threshold)[0]
            if len(assigned) == 0:
                # Assign to most likely cluster if none pass threshold
                assigned = np.array([np.argmax(prob)])
            labels.append(assigned)
        
        return labels, n_clusters
        
    except Exception as e:
        logger.error(f"GMM clustering failed: {e}")
        # Fallback: single cluster
        return [np.array([0]) for _ in range(len(embeddings))], 1


def perform_clustering(
    embeddings: np.ndarray,
    dim: int = 10,
    threshold: float = 0.5,
    verbose: bool = False
) -> List[List[int]]:
    """Perform hierarchical clustering on embeddings.
    
    This implements the two-level clustering from RAPTOR:
    1. Global clustering on dimensionally-reduced embeddings
    2. Local clustering within each global cluster
    
    Args:
        embeddings: Input embeddings
        dim: Dimensionality for reduction
        threshold: GMM probability threshold
        verbose: Print debug info
        
    Returns:
        List of cluster assignments (each is list of indices)
    """
    if len(embeddings) <= dim + 1:
        # Too few samples, put all in one cluster
        return [list(range(len(embeddings)))]
    
    # Step 1: Global dimensionality reduction
    reduced_global = reduce_dimensions(embeddings, dim=dim)
    
    # Step 2: Global clustering
    global_labels, n_global_clusters = gmm_cluster(reduced_global, threshold=threshold)
    
    if verbose:
        logger.info(f"Global clusters: {n_global_clusters}")
    
    # Step 3: Local clustering within each global cluster
    all_clusters = []
    
    for i in range(n_global_clusters):
        # Find points in this global cluster
        cluster_indices = [
            idx for idx, labels in enumerate(global_labels)
            if i in labels
        ]
        
        if len(cluster_indices) == 0:
            continue
            
        if verbose:
            logger.info(f"Global cluster {i}: {len(cluster_indices)} points")
        
        if len(cluster_indices) <= dim + 1:
            # Too few for local clustering
            all_clusters.append(cluster_indices)
        else:
            # Local clustering on original embeddings
            cluster_embeddings = embeddings[cluster_indices]
            
            try:
                local_reduced = reduce_dimensions(
                    cluster_embeddings,
                    dim=min(dim, len(cluster_indices) - 1),
                    n_neighbors=min(10, len(cluster_indices) - 1)
                )
                
                local_labels, n_local = gmm_cluster(local_reduced, threshold=threshold)
                
                if verbose:
                    logger.info(f"  Local clusters: {n_local}")
                
                # Convert local labels back to original indices
                for j in range(n_local):
                    local_cluster = [
                        cluster_indices[idx]
                        for idx, labels in enumerate(local_labels)
                        if j in labels
                    ]
                    if local_cluster:
                        all_clusters.append(local_cluster)
                        
            except Exception as e:
                logger.warning(f"Local clustering failed: {e}")
                all_clusters.append(cluster_indices)
    
    return all_clusters
