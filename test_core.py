import numpy as np
from kmeans_core import kmeans


def test_kmeans_basic():
    rng = np.random.default_rng(42)
    # 3 grupos bien separados en R^2
    c1 = rng.normal(loc=[0.0, 0.0], scale=0.3, size=(50, 2))
    c2 = rng.normal(loc=[5.0, 5.0], scale=0.3, size=(50, 2))
    c3 = rng.normal(loc=[0.0, 6.0], scale=0.3, size=(50, 2))
    X = np.vstack([c1, c2, c3])

    for init_method in ["random", "kmeans++"]:
        res = kmeans(X, k=3, init=init_method, seed=10)
        print(f"--- Método: {init_method} ---")
        print(f"Convergió: {res['converged']} en {res['iterations']} iteraciones")
        print(f"Inercia final (WCSS): {res['inertia']:.4f}")
        print(f"Centroides:\n{np.round(res['centroids'], 2)}\n")

        assert res["converged"], "El algoritmo debería converger en datos bien separados"
        assert res["centroids"].shape == (3, 2)
        assert res["labels"].shape == (150,)
        assert len(res["history"]) == res["iterations"] + 1


if __name__ == "__main__":
    test_kmeans_basic()