import numpy as np

from src.scores import knn, logit, mahalanobis, react


def test_logit_scores():
    lg = np.array([[10.0, 0.0, 0.0], [1.0, 1.0, 1.0]])  # confident vs uniform
    assert np.allclose(logit.msp(lg)[1], 1 / 3)
    assert logit.msp(lg)[0] > logit.msp(lg)[1]
    assert np.allclose(logit.max_logit(lg), [10.0, 1.0])
    assert np.allclose(logit.energy(lg)[1], 1 + np.log(3))


def test_react_clips():
    feats = np.array([[1.0, 100.0]])
    w, b = np.eye(2), np.zeros(2)
    thr = react.fit(np.array([[0.0, 1.0], [2.0, 3.0]]), percentile=50)  # median of {0,1,2,3} = 1.5
    assert thr == 1.5
    # clipped features [1, 1.5] -> logits [1, 1.5] -> logsumexp
    assert np.allclose(react.score(feats, w, b, thr), np.log(np.exp(1) + np.exp(1.5)))


def test_knn_distance():
    train = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    idx = knn.fit(train)
    s = knn.score(np.array([[2.0, 0.0]]), idx, k=1)  # normalised (1,0) sits on a training point
    assert np.allclose(s, 0.0, atol=1e-6)
    s2 = knn.score(np.array([[-1.0, 0.0]]), idx, k=1)  # farthest direction
    assert s2[0] < s[0]


def test_mahalanobis_prefers_near_class_mean():
    rng = np.random.default_rng(0)
    x = np.r_[rng.normal(0, 1, (200, 2)), rng.normal(5, 1, (200, 2))]
    y = np.r_[np.zeros(200, int), np.ones(200, int)]
    means, prec = mahalanobis.fit(x, y, num_classes=2)
    s = mahalanobis.score(np.array([[0.0, 0.0], [5.0, 5.0], [20.0, -20.0]]), means, prec)
    assert s[0] > s[2] and s[1] > s[2]
