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

from scipy.stats import multivariate_normal
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture

from active_semi_clustering.semi_supervised.pairwise_constraints.copkmeans import COPKMeans


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
        Solo 'full' está implementado en el EM manual. Para otros tipos,
        usar SupervisedBaseline con sklearn.
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
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state

    # ------------------------------------------------------------------
    # EM helpers
    # ------------------------------------------------------------------

    def _e_step(
        self,
        X: np.ndarray,
        means: np.ndarray,
        covs: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calcula log-responsabilidades normalizadas y log-normalización."""
        n = X.shape[0]
        log_resp = np.empty((n, self.n_components))
        for k in range(self.n_components):
            try:
                log_resp[:, k] = np.log(weights[k] + 1e-300) + multivariate_normal.logpdf(
                    X, mean=means[k], cov=covs[k]
                )
            except np.linalg.LinAlgError:
                log_resp[:, k] = -np.inf
        # logsumexp manual con errstate para silenciar underflow inofensivo
        with np.errstate(under="ignore"):
            shift = log_resp.max(axis=1, keepdims=True)
            log_norm = shift + np.log(np.exp(log_resp - shift).sum(axis=1, keepdims=True) + 1e-300)
        log_resp -= log_norm
        return log_resp, log_norm

    def _m_step(
        self, X: np.ndarray, resp: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Actualiza (means, covarianzas, pesos) a partir de responsabilidades."""
        n, d = X.shape
        nk = resp.sum(axis=0) + 1e-10
        weights = nk / n
        means = (resp.T @ X) / nk[:, np.newaxis]
        covs = np.empty((self.n_components, d, d))
        for k in range(self.n_components):
            diff = X - means[k]
            covs[k] = (resp[:, k : k + 1] * diff).T @ diff / nk[k]
            covs[k] += self.reg_covar * np.eye(d)
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
        classes = np.unique(y_lab)
        n, d = X.shape

        # — Inicialización con estadísticos de las clases etiquetadas —
        means = np.array([X_lab[y_lab == c].mean(axis=0) for c in classes])
        covs = np.empty((self.n_components, d, d))
        for i, c in enumerate(classes):
            X_c = X_lab[y_lab == c]
            if len(X_c) > 1:
                covs[i] = np.cov(X_c.T) + self.reg_covar * np.eye(d)
            else:
                covs[i] = np.eye(d) * self.reg_covar
        weights = np.array([np.sum(y_lab == c) for c in classes], dtype=float)
        weights /= weights.sum()

        # — Máscara one-hot para el paso E sobre puntos etiquetados —
        n_lab = labeled_mask.sum()
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
        n = X.shape[0]
        log_prob = np.empty((n, self.n_components))
        for k in range(self.n_components):
            log_prob[:, k] = np.log(self.weights_[k] + 1e-300) + multivariate_normal.logpdf(
                X, mean=self.means_[k], cov=self.covariances_[k]
            )
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
    de la fracción etiquetada. Usa COP-KMeans de active-semi-supervised-clustering.

    Generación de restricciones:
      - Must-link  : dos puntos etiquetados con la misma clase.
      - Cannot-link: dos puntos etiquetados con distinta clase.
      - Se muestrean aleatoriamente hasta ``max_ml`` y ``max_cl`` pares
        para no saturar el solver.

    La predicción sobre nuevos puntos asigna cada muestra al centroide más
    cercano (COP-KMeans no expone ``predict``).

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
    random_state : int, default=42
    """

    def __init__(
        self,
        n_clusters: int = 7,
        max_iter: int = 300,
        max_ml: int = 500,
        max_cl: int = 500,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
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

        self.model_ = COPKMeans(n_clusters=self.n_clusters, max_iter=self.max_iter)
        self.model_.fit(X, ml=self.ml_, cl=self.cl_)
        self.labels_ = self.model_.labels_

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
        centers = self.model_.cluster_centers_
        dists = np.linalg.norm(X[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
        clusters = dists.argmin(axis=1)
        return np.array([self.cluster_to_class_[c] for c in clusters])
