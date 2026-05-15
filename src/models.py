"""
Modelos semi-supervisados para clasificación de Dry Bean (Lab 10).

Tres clases principales:
  - SupervisedBaseline  : GMM + RandomForest entrenados solo con la fracción etiquetada.
  - SemiSupervisedGMM   : EM manual con restricciones duras en puntos etiquetados.
  - ConstrainedKMeans   : COP-KMeans con must-link / cannot-link derivadas de etiquetas.

Convenio de etiquetas: y_partial usa -1 para muestras sin etiquetar
(compatible con sklearn.semi_supervised).
"""

import numpy as np

from scipy.linalg import solve_triangular
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture


# ---------------------------------------------------------------------------
# 1. SupervisedBaseline
# ---------------------------------------------------------------------------

class SupervisedBaseline:
    """
    Baseline supervisado: entrena solo con la fracción etiquetada (y != -1).

    Incluye dos modelos en paralelo:
      - ``gmm_``  : GaussianMixture con means_init derivado de los centroides
                    por clase etiquetada.
      - ``rf_``   : RandomForestClassifier como segundo punto de comparación.

    El método ``predict`` usa el GMM; ``predict_rf`` usa el RandomForest.

    Parameters
    ----------
    n_components : int, default=7
        Número de componentes Gaussianas (idealmente = número de clases).
    covariance_type : {'full', 'tied', 'diag', 'spherical'}, default='full'
        Tipo de matriz de covarianza por componente.
    max_iter : int, default=200
        Iteraciones máximas del EM interno de GaussianMixture.
    tol : float, default=1e-4
        Tolerancia de convergencia para EM (cambio relativo en log-likelihood).
    reg_covar : float, default=1e-3
        Regularización añadida a la diagonal de cada covarianza para evitar
        matrices singulares. Valor mayor que el default de sklearn (1e-6) porque
        el dataset de 16 features estandarizadas puede producir underflow con
        regularización muy pequeña.
    n_estimators : int, default=100
        Número de árboles en el RandomForestClassifier.
    random_state : int, default=42
        Semilla global de aleatoriedad.
    """

    def __init__(
        self,
        n_components: int = 7,
        covariance_type: str = "full",
        max_iter: int = 200,
        tol: float = 1e-4,
        reg_covar: float = 1e-3,
        n_estimators: int = 100,
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X: np.ndarray, y_partial: np.ndarray) -> "SupervisedBaseline":
        """
        Entrena GMM y RandomForest usando únicamente las muestras etiquetadas.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y_partial : ndarray of shape (n_samples,)
            Etiquetas enteras para las muestras etiquetadas; -1 para las no etiquetadas.
        """
        mask = y_partial != -1
        X_lab, y_lab = X[mask], y_partial[mask]

        classes = np.unique(y_lab)
        means_init = np.array([X_lab[y_lab == c].mean(axis=0) for c in classes])

        self.gmm_ = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            max_iter=self.max_iter,
            tol=self.tol,
            reg_covar=self.reg_covar,
            means_init=means_init,
            random_state=self.random_state,
        )
        # sklearn >=1.8 eleva FloatingPointError en underflow de exp (inofensivo)
        with np.errstate(under="ignore"):
            self.gmm_.fit(X_lab)
        self._build_cluster_map(X_lab, y_lab)

        self.rf_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.rf_.fit(X_lab, y_lab)
        return self

    def _build_cluster_map(self, X_lab: np.ndarray, y_lab: np.ndarray) -> None:
        """Mapea cada componente GMM a su clase más frecuente en los puntos etiquetados."""
        n_classes = int(y_lab.max()) + 1
        cluster_assignments = self.gmm_.predict(X_lab)
        self.cluster_to_class_: dict[int, int] = {}
        for k in range(self.n_components):
            in_k = cluster_assignments == k
            if not in_k.any():
                self.cluster_to_class_[k] = int(y_lab[0])
            else:
                self.cluster_to_class_[k] = int(
                    np.bincount(y_lab[in_k], minlength=n_classes).argmax()
                )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicción usando el GMM (componente → clase por votación mayoritaria)."""
        clusters = self.gmm_.predict(X)
        return np.array([self.cluster_to_class_[c] for c in clusters])

    def predict_rf(self, X: np.ndarray) -> np.ndarray:
        """Predicción usando el RandomForestClassifier."""
        return self.rf_.predict(X)


# ---------------------------------------------------------------------------
# 2. SemiSupervisedGMM
# ---------------------------------------------------------------------------

class SemiSupervisedGMM:
    """
    GMM semi-supervisado con EM manual (restricciones duras en etiquetados).

    Algoritmo:
      1. Inicializar parámetros con los centroides y covarianzas de las
         clases etiquetadas.
      2. Paso E: calcular responsabilidades log-estabilizadas para todos los
         puntos; para los puntos etiquetados, sobreescribir con one-hot
         según su clase verdadera.
      3. Paso M: actualizar medias, covarianzas y pesos usando todas las
         responsabilidades (etiquetadas + no etiquetadas).
      4. Repetir hasta convergencia (|Δ log-likelihood| < tol) o max_iter.

    Atributos post-fit
    ------------------
    means_ : ndarray (n_components, n_features)
    covariances_ : ndarray (n_components, n_features, n_features)
    weights_ : ndarray (n_components,)
    log_likelihood_history_ : list[float]  — log-likelihood media por iteración.
    n_iter_ : int

    Parameters
    ----------
    n_components : int, default=7
        Número de componentes Gaussianas. Debe ser igual al número de clases
        para que el mapeo one-hot en el paso E sea directo.
    covariance_type : str, default='full'
        Tipo de matriz de covarianza a usar en el EM manual:
        ``full`` (una matriz completa por componente), ``tied`` (una matriz
        completa compartida), ``diag`` (diagonal por componente) o
        ``spherical`` (varianza escalar por componente).
    max_iter : int, default=100
        Iteraciones máximas de EM.
    tol : float, default=1e-4
        Umbral de convergencia sobre la variación absoluta de log-likelihood.
    reg_covar : float, default=1e-3
        Regularización diagonal añadida a cada covarianza en el paso M.
    random_state : int, default=42
    """

    def __init__(
        self,
        n_components: int = 7,
        covariance_type: str = "full",
        max_iter: int = 100,
        tol: float = 1e-4,
        reg_covar: float = 1e-3,
        random_state: int = 42,
    ):
        allowed_covariances = {"full", "tied", "diag", "spherical"}
        if covariance_type not in allowed_covariances:
            raise ValueError(
                f"covariance_type debe estar en {sorted(allowed_covariances)}, "
                f"recibido: {covariance_type!r}"
            )
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state

    # ------------------------------------------------------------------
    # EM helpers
    # ------------------------------------------------------------------

    def _log_gaussian_prob(
        self,
        X: np.ndarray,
        means: np.ndarray,
        covs: np.ndarray,
    ) -> np.ndarray:
        """Log densidad gaussiana vectorizada por componente."""
        n, d = X.shape
        log_prob = np.empty((n, self.n_components))
        log_2pi = d * np.log(2.0 * np.pi)

        for k in range(self.n_components):
            diff = X - means[k]
            try:
                if self.covariance_type in {"diag", "spherical"}:
                    var = np.maximum(np.diag(covs[k]), 1e-300)
                    log_det = np.log(var).sum()
                    mahalanobis = (diff * diff / var).sum(axis=1)
                else:
                    chol = np.linalg.cholesky(covs[k])
                    solved = solve_triangular(
                        chol,
                        diff.T,
                        lower=True,
                        check_finite=False,
                    )
                    log_det = 2.0 * np.log(np.diag(chol)).sum()
                    mahalanobis = (solved * solved).sum(axis=0)

                log_prob[:, k] = -0.5 * (log_2pi + log_det + mahalanobis)
            except np.linalg.LinAlgError:
                log_prob[:, k] = -np.inf

        return log_prob

    def _e_step(
        self,
        X: np.ndarray,
        means: np.ndarray,
        covs: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calcula log-responsabilidades normalizadas y log-normalización."""
        log_resp = self._log_gaussian_prob(X, means, covs)
        log_resp += np.log(weights + 1e-300)

        # logsumexp manual con errstate para silenciar underflow inofensivo
        with np.errstate(under="ignore"):
            shift = log_resp.max(axis=1, keepdims=True)
            log_norm = shift + np.log(np.exp(log_resp - shift).sum(axis=1, keepdims=True) + 1e-300)
        log_resp -= log_norm
        return log_resp, log_norm

    def _estimate_covariances(
        self, X: np.ndarray, resp: np.ndarray, means: np.ndarray, nk: np.ndarray
    ) -> np.ndarray:
        """Estima covarianzas respetando ``covariance_type``."""
        _, d = X.shape
        covs = np.empty((self.n_components, d, d))

        if self.covariance_type == "tied":
            tied_cov = np.zeros((d, d))
            for k in range(self.n_components):
                diff = X - means[k]
                tied_cov += (resp[:, k : k + 1] * diff).T @ diff
            tied_cov /= nk.sum()
            tied_cov += self.reg_covar * np.eye(d)
            covs[:] = tied_cov
            return covs

        for k in range(self.n_components):
            diff = X - means[k]
            if self.covariance_type == "full":
                cov = (resp[:, k : k + 1] * diff).T @ diff / nk[k]
                cov += self.reg_covar * np.eye(d)
            elif self.covariance_type == "diag":
                var = (resp[:, k : k + 1] * diff**2).sum(axis=0) / nk[k]
                cov = np.diag(var + self.reg_covar)
            elif self.covariance_type == "spherical":
                var = (resp[:, k] * (diff**2).sum(axis=1)).sum() / (nk[k] * d)
                cov = (var + self.reg_covar) * np.eye(d)
            else:  # Defensa adicional; __init__ ya valida.
                raise ValueError(f"covariance_type no soportado: {self.covariance_type}")
            covs[k] = cov
        return covs

    def _m_step(
        self, X: np.ndarray, resp: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Actualiza (means, covarianzas, pesos) a partir de responsabilidades."""
        n = X.shape[0]
        nk = resp.sum(axis=0) + 1e-10
        weights = nk / n
        means = (resp.T @ X) / nk[:, np.newaxis]
        covs = self._estimate_covariances(X, resp, means, nk)
        return means, covs, weights

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y_partial: np.ndarray) -> "SemiSupervisedGMM":
        """
        Ajusta el GMM semi-supervisado.

        Parameters
        ----------
        X : ndarray (n_samples, n_features)
        y_partial : ndarray (n_samples,)
            Etiquetas en 0..n_classes-1; -1 para no etiquetados.
        """
        labeled_mask = y_partial != -1
        X_lab = X[labeled_mask]
        y_lab = y_partial[labeled_mask].astype(int)
        n, d = X.shape
        n_lab = labeled_mask.sum()
        rng = np.random.default_rng(self.random_state)

        # — Inicialización con estadísticos de las clases etiquetadas —
        means = np.empty((self.n_components, d))
        for k in range(self.n_components):
            in_class = y_lab == k
            if in_class.any():
                means[k] = X_lab[in_class].mean(axis=0)
            else:
                means[k] = X_lab[rng.integers(0, len(X_lab))]

        init_resp = np.zeros((n_lab, self.n_components))
        for i, c in enumerate(y_lab):
            init_resp[i, c] = 1.0
        init_nk = init_resp.sum(axis=0) + 1e-10
        covs = self._estimate_covariances(X_lab, init_resp, means, init_nk)
        for k in range(self.n_components):
            if np.sum(y_lab == k) <= 1:
                covs[k] = np.eye(d) * self.reg_covar
        weights = np.bincount(y_lab, minlength=self.n_components).astype(float)
        weights = np.where(weights == 0, 1e-10, weights)
        weights /= weights.sum()

        # — Máscara one-hot para el paso E sobre puntos etiquetados —
        labeled_resp_fixed = np.zeros((n_lab, self.n_components))
        for i, c in enumerate(y_lab):
            labeled_resp_fixed[i, c] = 1.0

        self.log_likelihood_history_: list[float] = []
        prev_ll = -np.inf

        # Suprimir underflow en exp/multiply: es inofensivo (valores muy pequeños → 0)
        with np.errstate(under="ignore"):
            for iteration in range(self.max_iter):
                # Paso E
                log_resp, log_norm = self._e_step(X, means, covs, weights)
                resp = np.exp(log_resp)

                # Forzar responsabilidades de puntos etiquetados a one-hot
                resp[labeled_mask] = labeled_resp_fixed

                # Log-likelihood media (solo para seguimiento; no afecta al M-step)
                ll = float(log_norm.mean())
                self.log_likelihood_history_.append(ll)

                if abs(ll - prev_ll) < self.tol:
                    break
                prev_ll = ll

                # Paso M
                means, covs, weights = self._m_step(X, resp)

        self.means_ = means
        self.covariances_ = covs
        self.weights_ = weights
        self.n_iter_ = iteration + 1
        self._build_cluster_map(X_lab, y_lab)
        return self

    def _build_cluster_map(self, X_lab: np.ndarray, y_lab: np.ndarray) -> None:
        n_classes = int(y_lab.max()) + 1
        clusters = self._predict_raw(X_lab)
        self.cluster_to_class_: dict[int, int] = {}
        for k in range(self.n_components):
            in_k = clusters == k
            if not in_k.any():
                self.cluster_to_class_[k] = int(y_lab[0])
            else:
                self.cluster_to_class_[k] = int(
                    np.bincount(y_lab[in_k], minlength=n_classes).argmax()
                )

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Asigna cada punto a la componente más probable (índice de componente)."""
        log_prob = self._log_gaussian_prob(X, self.means_, self.covariances_)
        log_prob += np.log(self.weights_ + 1e-300)
        return log_prob.argmax(axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicción de clases (componente más probable → clase por votación)."""
        clusters = self._predict_raw(X)
        return np.array([self.cluster_to_class_[c] for c in clusters])


# ---------------------------------------------------------------------------
# 3. ConstrainedKMeans
# ---------------------------------------------------------------------------

class ConstrainedKMeans:
    """
    K-Means con restricciones must-link (ML) y cannot-link (CL) derivadas
    de la fracción etiquetada. Implementa una variante practica de COP-KMeans:
    en cada iteración asigna cada punto al centroide más cercano que no viole
    sus restricciones indexadas.

    Generación de restricciones:
      - Must-link  : dos puntos etiquetados con la misma clase.
      - Cannot-link: dos puntos etiquetados con distinta clase.
      - Se muestrean aleatoriamente hasta ``max_ml`` y ``max_cl`` pares
        para no saturar el solver.

    La predicción sobre nuevos puntos asigna cada muestra al centroide más
    cercano.

    Parameters
    ----------
    n_clusters : int, default=7
        Número de clusters (idealmente = número de clases).
    max_iter : int, default=300
        Iteraciones máximas de COP-KMeans.
    max_ml : int, default=500
        Número máximo de pares must-link a usar.
    max_cl : int, default=500
        Número máximo de pares cannot-link a usar.
    max_constraints : int, optional
        Alias usado para sensibilidad: si se especifica, fija simultáneamente
        ``max_ml`` y ``max_cl`` al mismo valor.
    random_state : int, default=42
    """

    def __init__(
        self,
        n_clusters: int = 7,
        max_iter: int = 300,
        max_ml: int = 500,
        max_cl: int = 500,
        max_constraints=None,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        if max_constraints is not None:
            max_ml = max_constraints
            max_cl = max_constraints
        self.max_constraints = max_constraints
        self.max_ml = max_ml
        self.max_cl = max_cl
        self.random_state = random_state

    def _generate_constraints(
        self, y_partial: np.ndarray, labeled_mask: np.ndarray
    ) -> tuple[list, list]:
        """
        Muestrea pares ML y CL a partir de los índices etiquetados.

        Estrategia vectorizada: se generan muestras aleatorias de pares de
        índices etiquetados y se clasifican como ML/CL según si comparten clase.
        """
        rng = np.random.default_rng(self.random_state)
        labeled_indices = np.where(labeled_mask)[0]
        y_lab = y_partial[labeled_mask]
        n_lab = len(labeled_indices)

        # Oversample para compensar colisiones y duplicados
        n_sample = (self.max_ml + self.max_cl) * 6
        ii = rng.integers(0, n_lab, size=n_sample)
        jj = rng.integers(0, n_lab, size=n_sample)
        valid = ii != jj
        ii, jj = ii[valid], jj[valid]

        same = y_lab[ii] == y_lab[jj]
        orig_i = labeled_indices[ii]
        orig_j = labeled_indices[jj]

        # Normalizar orden para deduplicar
        swap = orig_i > orig_j
        orig_i[swap], orig_j[swap] = orig_j[swap], orig_i[swap]

        pairs = np.stack([orig_i, orig_j], axis=1)

        def _unique_pairs(mask: np.ndarray, limit: int) -> list:
            sub = pairs[mask]
            _, idx = np.unique(sub, axis=0, return_index=True)
            sub = sub[idx[:limit]]
            return [tuple(p) for p in sub]

        ml = _unique_pairs(same, self.max_ml)
        cl = _unique_pairs(~same, self.max_cl)
        return ml, cl

    def _build_constraint_adjacency(self, n_samples: int) -> None:
        """Indexa restricciones por punto para validar asignaciones rapido."""
        self.ml_adj_ = [[] for _ in range(n_samples)]
        self.cl_adj_ = [[] for _ in range(n_samples)]
        for i, j in self.ml_:
            self.ml_adj_[i].append(j)
            self.ml_adj_[j].append(i)
        for i, j in self.cl_:
            self.cl_adj_[i].append(j)
            self.cl_adj_[j].append(i)

    def _initial_centers(self, X: np.ndarray, y_partial: np.ndarray) -> np.ndarray:
        """Inicializa centroides con medias etiquetadas y fallback aleatorio."""
        rng = np.random.default_rng(self.random_state)
        n, d = X.shape
        centers = np.empty((self.n_clusters, d))
        for k in range(self.n_clusters):
            in_class = y_partial == k
            if in_class.any():
                centers[k] = X[in_class].mean(axis=0)
            else:
                centers[k] = X[rng.integers(0, n)]
        return centers

    def _nearest_labels(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        dists = np.linalg.norm(X[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
        return dists.argmin(axis=1)

    def _can_assign(self, index: int, cluster: int, labels: np.ndarray) -> bool:
        """Revisa si asignar `index` a `cluster` viola restricciones conocidas."""
        for other in self.ml_adj_[index]:
            if labels[other] != cluster:
                return False
        for other in self.cl_adj_[index]:
            if labels[other] == cluster:
                return False
        return True

    def _update_centers(
        self, X: np.ndarray, labels: np.ndarray, old_centers: np.ndarray
    ) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        centers = old_centers.copy()
        n = X.shape[0]
        for k in range(self.n_clusters):
            in_cluster = labels == k
            if in_cluster.any():
                centers[k] = X[in_cluster].mean(axis=0)
            else:
                centers[k] = X[rng.integers(0, n)]
        return centers

    def fit(self, X: np.ndarray, y_partial: np.ndarray) -> "ConstrainedKMeans":
        """
        Ajusta COP-KMeans con restricciones derivadas de la fracción etiquetada.

        Parameters
        ----------
        X : ndarray (n_samples, n_features)
        y_partial : ndarray (n_samples,)
            Etiquetas en 0..n_classes-1; -1 para no etiquetados.
        """
        labeled_mask = y_partial != -1
        self.ml_, self.cl_ = self._generate_constraints(y_partial, labeled_mask)
        self._build_constraint_adjacency(len(X))

        rng = np.random.default_rng(self.random_state)
        centers = self._initial_centers(X, y_partial)
        labels = self._nearest_labels(X, centers)
        order = np.arange(len(X))

        for iteration in range(self.max_iter):
            old_labels = labels.copy()
            rng.shuffle(order)

            dists = np.linalg.norm(
                X[:, np.newaxis, :] - centers[np.newaxis, :, :],
                axis=2,
            )
            for i in order:
                for candidate in np.argsort(dists[i]):
                    if self._can_assign(i, int(candidate), labels):
                        labels[i] = int(candidate)
                        break

            centers = self._update_centers(X, labels, centers)
            if np.array_equal(labels, old_labels):
                break

        self.cluster_centers_ = centers
        self.labels_ = labels
        self.n_iter_ = iteration + 1

        # — Mapear clusters a clases por votación mayoritaria en etiquetados —
        y_lab = y_partial[labeled_mask].astype(int)
        labeled_idx = np.where(labeled_mask)[0]
        cluster_of_labeled = self.labels_[labeled_idx]
        n_classes = int(y_lab.max()) + 1
        self.cluster_to_class_: dict[int, int] = {}
        for k in range(self.n_clusters):
            in_k = cluster_of_labeled == k
            if not in_k.any():
                self.cluster_to_class_[k] = int(y_lab[0])
            else:
                self.cluster_to_class_[k] = int(
                    np.bincount(y_lab[in_k], minlength=n_classes).argmax()
                )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Asigna cada muestra al centroide más cercano y mapea a clase."""
        centers = self.cluster_centers_
        dists = np.linalg.norm(X[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
        clusters = dists.argmin(axis=1)
        return np.array([self.cluster_to_class_[c] for c in clusters])
