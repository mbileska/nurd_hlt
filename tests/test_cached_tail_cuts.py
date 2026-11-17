import numpy as np

from scripts.scan_cached_tail_cuts import scan, stable_choice


def test_cached_scan_matches_brute_force_with_ties():
    rng = np.random.default_rng(7)
    x, y = rng.integers(0, 10, size=(2, 300))
    w = rng.uniform(.1, 3, size=300)
    cuts = np.array([1., 3., 3., 6., 8.])
    result = scan(x, y, w, cuts, cuts, [1]*4)
    for i, a in enumerate(cuts):
        for j, b in enumerate(cuts):
            masks = ((x>a)&(y>b), (x>a)&(y<=b), (x<=a)&(y>b), (x<=a)&(y<=b))
            for k, mask in enumerate(masks):
                assert np.isclose(result['yields'][k,i,j], w[mask].sum())
                assert np.isclose(result['sumw2'][k,i,j], (w[mask]**2).sum())
                assert np.isclose(result['raw'][k,i,j], mask.sum())


def test_choice_requires_stable_valid_neighborhood():
    data = dict(valid=np.ones((5,5), bool), efficiency=np.full((5,5), .1),
                ratio=np.full((5,5), 1.2), sigma=np.full((5,5), .03))
    data['ratio'][2,2] = 1.0
    chosen = stable_choice(data, .02, .25)
    assert chosen is not None
    data['valid'][:] = False
    data['valid'][2,2] = True
    assert stable_choice(data, .02, .25) is None
