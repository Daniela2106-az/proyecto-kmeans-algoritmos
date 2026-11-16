"""
Implementación desde cero del algoritmo K-Means (Cormen et al., 4th Ed., Sección 33.1).
Soporta inicialización aleatoria ('random') y K-Means++ ('kmeans++').
"""

import numpy as np


def _initialize_centroids(
    X: np.ndarray, k: int, init: str, rng: np.random.Generator
) -> np.ndarray:
    """
    Inicializa k centroides a partir de los puntos en X.
    
    Complejidad temporal:
        - 'random': O(k * d)
        - 'kmeans++': O(k * n * d)
    """
    n_samples, n_features = X.shape

    if init == "random":
        indices = rng.choice(n_samples, size=k, replace=False)
        return X[indices].copy()

    elif init == "kmeans++":
        centroids = np.empty((k, n_features), dtype=np.float64)
        # 1. Primer centroide elegido uniformemente al azar
        first_idx = rng.integers(0, n_samples)
        centroids[0] = X[first_idx]

        # 2. Elegir los k-1 centroides restantes con probabilidad proporcional a D(x)^2
        closest_dist_sq = np.sum((X - centroids[0]) ** 2, axis=1)

        for c_idx in range(1, k):
            total_dist_sq = np.sum(closest_dist_sq)
            if total_dist_sq == 0:
                # Todos los puntos restantes coinciden con los centroides actuales
                idx = rng.integers(0, n_samples)
            else:
                probabilities = closest_dist_sq / total_dist_sq
                idx = rng.choice(n_samples, p=probabilities)

            centroids[c_idx] = X[idx]
            new_dist_sq = np.sum((X - centroids[c_idx]) ** 2, axis=1)
            closest_dist_sq = np.minimum(closest_dist_sq, new_dist_sq)

        return centroids

    else:
        raise ValueError(f"Método de inicialización no soportado: '{init}'. Use 'random' o 'kmeans++'.")


def _assign_clusters(X: np.ndarray, centroids: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Asigna cada punto en X al centroide más cercano según distancia euclidiana al cuadrado.
    
    Complejidad temporal: O(k * n * d)
    
    Retorna:
        labels: arreglo (n,) con el índice del cluster [0, k-1] para cada punto.
        inertia: suma de distancias euclidianas al cuadrado (WCSS).
    """
    # diff tiene forma (n, k, d)
    diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)  # forma (n, k)

    labels = np.argmin(dist_sq, axis=1)
    min_dist_sq = dist_sq[np.arange(X.shape[0]), labels]
    inertia = float(np.sum(min_dist_sq))

    return labels, inertia


def _update_centroids(
    X: np.ndarray,
    labels: np.ndarray,
    k: int,
    old_centroids: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Recalcula cada centroide como la media aritmética de los puntos asignados a su cluster.
    Si un cluster queda vacío, se reinicializa con un punto aleatorio del dataset.
    
    Complejidad temporal: O(n * d)
    """
    n_samples, n_features = X.shape
    new_centroids = np.zeros((k, n_features), dtype=np.float64)

    for j in range(k):
        cluster_points = X[labels == j]
        if len(cluster_points) > 0:
            new_centroids[j] = np.mean(cluster_points, axis=0)
        else:
            # Manejo de cluster vacío: reubicar en un punto aleatorio
            random_idx = rng.integers(0, n_samples)
            new_centroids[j] = X[random_idx]

    return new_centroids


def kmeans(
    X: np.ndarray,
    k: int,
    init: str = "random",
    max_iter: int = 300,
    tol: float = 1e-6,
    seed: int | None = None,
) -> dict:
    """
    Ejecuta el algoritmo de Lloyd para K-Means Clustering.

    Parámetros:
        X: matriz de datos numéricos de forma (n, d).
        k: número de clusters (1 <= k <= n).
        init: estrategia de inicialización ('random' o 'kmeans++').
        max_iter: número máximo de iteraciones permitidas.
        tol: tolerancia en el desplazamiento de centroides (norma de Frobenius) para convergencia.
        seed: semilla aleatoria para reproducibilidad.

    Retorna:
        Diccionario estándar (contrato del equipo) con:
            - 'centroids': np.ndarray (k, d)
            - 'labels': np.ndarray (n,)
            - 'inertia': float (WCSS final)
            - 'iterations': int (iteraciones ejecutadas hasta converger)
            - 'converged': bool (True si convergió antes de max_iter)
            - 'history': list[np.ndarray] (centroides en cada paso, incl. inicialización)
            - 'inertia_history': list[float] (inercia en cada iteración)
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X debe ser una matriz bidimensional de forma (n, d).")

    n_samples, _ = X.shape
    if not (1 <= k <= n_samples):
        raise ValueError(f"k debe estar entre 1 y n={n_samples}. Se recibió k={k}.")

    rng = np.random.default_rng(seed)

    centroids = _initialize_centroids(X, k, init, rng)
    history = [centroids.copy()]
    inertia_history = []

    converged = False
    labels = np.zeros(n_samples, dtype=np.int64)
    inertia = 0.0
    iteration = 0

    for iteration in range(1, max_iter + 1):
        # Paso 1: Asignación de puntos al centroide más cercano
        labels, inertia = _assign_clusters(X, centroids)
        inertia_history.append(inertia)

        # Paso 2: Actualización de centroides
        new_centroids = _update_centroids(X, labels, k, centroids, rng)
        history.append(new_centroids.copy())

        # Paso 3: Criterio de parada (desplazamiento total <= tol)
        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids

        if shift <= tol:
            # Recalcular asignación e inercia final con los centroides convergidos
            labels, inertia = _assign_clusters(X, centroids)
            converged = True
            break

    return {
        "centroids": centroids,
        "labels": labels,
        "inertia": inertia,
        "iterations": iteration,
        "converged": converged,
        "history": history,
        "inertia_history": inertia_history,
    }