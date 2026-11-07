"""
Regression tests for the issues found in the statistical / implementation audit.
Each test names the audit item it covers.
"""
import warnings

import numpy as np
import pytest
from scipy import integrate
from scipy.stats import norm, lognorm, weibull_min, ecdf, CensoredData

from ndimpute import impute
from ndimpute._ros_left import impute_ros_left, weibull_ppoints
from ndimpute._interval import impute_interval_ros
from ndimpute._turnbull import turnbull_em, predict_turnbull
from ndimpute._preprocess import parse_censored
from ndimpute import _truncated as T


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yield


# ---------------------------------------------------------------- C1
def test_C1_right_markers_with_mixed_type_are_right_censored():
    df = impute(["1", "2", "3", ">4", "5", "6", ">7"], censoring_type='mixed', method='substitution')
    assert df['censoring_status'].tolist() == [0, 0, 0, 1, 0, 0, 1]
    # right substitution default = value
    assert df.loc[3, 'imputed_value'] == 4.0 and df.loc[6, 'imputed_value'] == 7.0


def test_C1_left_markers_with_mixed_type_are_left_censored():
    df = impute(["<1", "2", "3", "4"], censoring_type='mixed', method='substitution')
    assert df['censoring_status'].tolist() == [-1, 0, 0, 0]


def test_C1_conflicting_type_raises():
    with pytest.raises(ValueError):
        impute(["<1", "2", ">3"], censoring_type='left')
    with pytest.raises(ValueError):
        impute(["<1", "2", "3"], censoring_type='right')


# ---------------------------------------------------------------- C2 / H5
def test_C2_integer_01_status_defaults_to_left():
    v = np.array([0.5, 2, 3, 4, 5, 6, 7, 8.])
    s = [1, 0, 0, 0, 0, 0, 0, 0]
    df = impute(v, s)
    assert df.attrs['censoring_type'] == 'left'
    assert df.loc[0, 'imputed_value'] <= 0.5
    bool_df = impute(v, np.array(s, dtype=bool))
    np.testing.assert_allclose(df['imputed_value'], bool_df['imputed_value'])


def test_C2_integer_status_right_when_requested():
    v = np.array([10., 2, 3, 4, 5, 6, 7, 8])
    df = impute(v, [1, 0, 0, 0, 0, 0, 0, 0], censoring_type='right', method='parametric')
    assert df.loc[0, 'imputed_value'] >= 10.0


def test_C2_status_with_minus_one_is_mixed():
    df = impute([1., 2, 3, 4, 5, 6, 7, 8, 9, 10], [-1, 0, 0, 0, 0, 0, 0, 0, 0, 1], method='parametric')
    assert df.attrs['censoring_type'] == 'mixed'


def test_C2_invalid_status_values_raise():
    with pytest.raises(ValueError):
        impute([1., 2, 3], [0, 2, 0])
    with pytest.raises(ValueError):
        impute([1., 2, 3], [-1, 0, 0], censoring_type='left')


def test_lwptrends_string_status():
    df = impute([0.5, 2, 3, 4, 5, 6, 30.], ['lt', 'not', 'not', 'not', 'not', 'not', 'gt'],
                method='parametric')
    assert df['censoring_status'].tolist() == [-1, 0, 0, 0, 0, 0, 1]


# ---------------------------------------------------------------- H1
def test_H1_textbook_tce_matches_nada():
    # Output of NADA::ros (R, NADA 1.6-1.1), mapped back to original order.
    x = np.array([1, 1, 1, 2.4, 3.5, 5.1, 8, 1, 1, 12, 1, 4.2])
    c = x == 1
    out = impute_ros_left(x, c, clamp=False)
    np.testing.assert_allclose(out[c], [0.3461, 0.5619, 0.7866, 1.0357, 1.3213, 1.6578], atol=5e-5)
    capped = impute_ros_left(x, c)  # default clamp=True
    assert np.all(capped[c] <= 1.0)


def test_H1_no_downward_shift_without_censoring_structure():
    # With a single censored value below all detects, detects get
    # (1-P) + P r/(A+1) -- the largest detect must not be shifted to (n-1)/(n+1).
    from ndimpute._ros_left import hirsch_stedinger_ppoints
    x = np.array([0.5, 1, 2, 3, 4, 5.])
    c = np.array([True, False, False, False, False, False])
    pp, _ = hirsch_stedinger_ppoints(x, c)
    P = 5 / 6
    np.testing.assert_allclose(pp[1:], (1 - P) + P * np.arange(1, 6) / 6)


# ---------------------------------------------------------------- H3
def test_H3_parametric_stochastic_respects_bounds_in_extreme_tail():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 200)
    st = np.zeros(200, int)
    x[0], st[0] = 7.0, 1      # far upper tail
    x[1], st[1] = -7.0, -1    # far lower tail
    for k in range(20):
        df = impute(x, st, censoring_type='mixed', method='parametric', dist='normal',
                    impute_type='stochastic', random_state=k)
        assert df.loc[0, 'imputed_value'] >= 7.0
        assert df.loc[1, 'imputed_value'] <= -7.0


@pytest.mark.parametrize('dist', ['lognormal', 'weibull'])
def test_H3_positive_dists_stochastic_bounds(dist):
    rng = np.random.default_rng(2)
    x = rng.lognormal(0, 0.3, 200)
    st = np.zeros(200, int)
    x[0], st[0] = 50.0, 1
    x[1], st[1] = 1e-3, -1
    for k in range(20):
        df = impute(x, st, censoring_type='mixed', method='parametric', dist=dist,
                    impute_type='stochastic', random_state=k)
        assert df.loc[0, 'imputed_value'] >= 50.0
        assert 0 < df.loc[1, 'imputed_value'] <= 1e-3
    df = impute(x, st, censoring_type='mixed', method='parametric', dist=dist, impute_type='mean')
    assert df.loc[0, 'imputed_value'] > 50.0
    assert 0 < df.loc[1, 'imputed_value'] < 1e-3


# ---------------------------------------------------------------- H4
@pytest.mark.parametrize('impute_type', ['mean', 'stochastic'])
def test_H4_interval_ros_tails_are_finite_and_inside(impute_type):
    x = np.random.default_rng(0).normal(0, 1, 500)
    left = np.r_[x, -np.inf, 30.0]
    right = np.r_[x, -15.0, np.inf]
    out = impute_interval_ros(left, right, dist='normal', impute_type=impute_type, random_state=0)
    assert np.isfinite(out[-2]) and out[-2] <= -15.0
    assert np.isfinite(out[-1]) and out[-1] > 30.0


def test_H4_interval_ros_lognormal_left_tail_not_zero():
    x = np.random.default_rng(0).lognormal(0, 0.3, 500)
    left = np.r_[x, 0.0]
    right = np.r_[x, 1e-4]
    for it in ('mean', 'stochastic'):
        out = impute_interval_ros(left, right, dist='lognormal', impute_type=it, random_state=1)
        assert 0 < out[-1] <= 1e-4


# ---------------------------------------------------------------- truncated helpers
def _quad_mean(pdf, lo, hi):
    num = integrate.quad(lambda t: t * pdf(t), lo, hi, limit=200)[0]
    den = integrate.quad(pdf, lo, hi, limit=200)[0]
    return num / den


@pytest.mark.parametrize('lo,hi', [(-np.inf, 8.0), (11.0, np.inf), (9.0, 12.0), (-np.inf, 2.0), (17.0, np.inf)])
def test_norm_trunc_mean_matches_quadrature(lo, hi):
    mu, s = 10.0, 2.0
    got = T.norm_trunc_mean(mu, s, lo, hi)
    want = _quad_mean(norm(mu, s).pdf, max(lo, mu - 40 * s), min(hi, mu + 40 * s))
    np.testing.assert_allclose(got, want, rtol=1e-6)


@pytest.mark.parametrize('lo,hi', [(0.0, 3.0), (20.0, np.inf), (5.0, 9.0), (0.0, 0.5), (80.0, np.inf)])
def test_lognorm_trunc_mean_matches_quadrature(lo, hi):
    mu, s = 2.0, 0.6
    d = lognorm(s=s, scale=np.exp(mu))
    got = T.lognorm_trunc_mean(mu, s, lo, hi)
    want = _quad_mean(d.pdf, lo, min(hi, d.ppf(1 - 1e-14)))
    np.testing.assert_allclose(got, want, rtol=1e-5)


@pytest.mark.parametrize('lo,hi', [(0.0, 3.0), (15.0, np.inf), (5.0, 9.0), (0.0, 0.2), (30.0, np.inf)])
def test_weibull_trunc_mean_matches_quadrature(lo, hi):
    k, lam = 2.0, 10.0
    d = weibull_min(k, scale=lam)
    got = T.weibull_trunc_mean(k, lam, lo, hi)
    want = _quad_mean(d.pdf, lo, min(hi, lo + 50 * lam))
    np.testing.assert_allclose(got, want, rtol=1e-5)


def test_trunc_samplers_match_distribution():
    rng = np.random.default_rng(3)
    u = rng.uniform(size=200000)
    s = T.weibull_trunc_sample(2.0, 10.0, 15.0, np.inf, u)
    np.testing.assert_allclose(s.mean(), T.weibull_trunc_mean(2.0, 10.0, 15.0, np.inf), rtol=2e-3)
    s = T.lognorm_trunc_sample(2.0, 0.6, 0.0, 3.0, u)
    np.testing.assert_allclose(s.mean(), T.lognorm_trunc_mean(2.0, 0.6, 0.0, 3.0), rtol=2e-3)
    s = T.norm_trunc_sample(10.0, 2.0, 9.0, 12.0, u)
    np.testing.assert_allclose(s.mean(), T.norm_trunc_mean(10.0, 2.0, 9.0, 12.0), rtol=2e-3)


# ---------------------------------------------------------------- M2
def test_M2_ros_mean_is_original_scale_conditional_mean():
    rng = np.random.default_rng(4)
    x = rng.lognormal(0, 1, 60)
    c = x < 0.6
    v = np.where(c, 0.6, x)
    out_q = impute_ros_left(v, c, impute_type='quantile', clamp=False)
    out_m = impute_ros_left(v, c, impute_type='mean', clamp=False)
    # The conditional mean equals the limit of the average of many quantile-spaced
    # imputations, and exceeds exp(mean of log) (Jensen) -- the old log-scale bug.
    assert np.all(np.isclose(out_m[c], out_m[c][0]))
    assert out_m[c][0] > np.exp(np.mean(np.log(out_q[c])))


# ---------------------------------------------------------------- M3
def test_M3_simple_ranks_censored_below_tied_detect():
    v = np.array([1, 1, 1, 2, 3, 4, 5, 6.])
    c = np.array([1, 0, 1, 0, 0, 0, 0, 0], bool)
    pp, _ = weibull_ppoints(v, c)
    assert pp[0] < pp[1] and pp[2] < pp[1]
    out = impute_ros_left(v, c, plotting_position='simple')
    assert np.all(out[c] < 1.0)


def test_M3_simple_warns_on_multiple_limits():
    v = np.array([1, 2, 1.5, 3, 4, 5, 6, 7.])
    c = np.array([1, 1, 0, 0, 0, 0, 0, 0], bool)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        impute_ros_left(v, c, plotting_position='simple')
    assert any('only valid for a single detection limit' in str(x.message) for x in w)


# ---------------------------------------------------------------- M4
def test_M4_quantile_mode_is_deterministic_and_ignores_seed():
    x = np.random.default_rng(5).lognormal(0, 1, 40)
    c = x < 0.7
    v = np.where(c, 0.7, x)
    a = impute(v, c, censoring_type='left')
    b = impute(v, c, censoring_type='left', random_state=123)
    np.testing.assert_array_equal(a['imputed_value'], b['imputed_value'])


def test_M4_stochastic_mode_seeded():
    x = np.random.default_rng(5).lognormal(0, 1, 40)
    c = x < 0.7
    v = np.where(c, 0.7, x)
    a = impute(v, c, censoring_type='left', impute_type='stochastic', random_state=1)
    b = impute(v, c, censoring_type='left', impute_type='stochastic', random_state=1)
    d = impute(v, c, censoring_type='left', impute_type='stochastic', random_state=2)
    np.testing.assert_array_equal(a['imputed_value'], b['imputed_value'])
    assert not np.allclose(a['imputed_value'][c], d['imputed_value'][c])
    assert np.all(a['imputed_value'][c] <= 0.7)


# ---------------------------------------------------------------- M5
def test_M5_identical_detects_raise():
    with pytest.raises(ValueError):
        impute_ros_left([1, 2, 2, 2.], [True, False, False, False])


def test_M5_sparse_data_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        impute_ros_left([0.5, 0.5, 0.5, 0.5, 1, 2, 3.], [True] * 4 + [False] * 3)
    assert any('detected values' in str(x.message) for x in w)


# ---------------------------------------------------------------- Low
def test_parametric_left_is_supported():
    x = np.random.default_rng(6).lognormal(1, 0.5, 100)
    c = x < 2.0
    df = impute(np.where(c, 2.0, x), c, censoring_type='left', method='parametric')
    assert np.all(df.loc[c, 'imputed_value'] < 2.0)


def test_parametric_auto_selects_by_likelihood():
    x = weibull_min(1.5, scale=10).rvs(size=2000, random_state=7)
    st = np.where(x > 20, 1, 0)
    df = impute(np.where(st == 1, 20, x), st, method='parametric', censoring_type='right', dist='auto')
    assert df.attrs['best_dist'] == 'weibull'


def test_missing_values_pass_through():
    df = impute([1, 2, np.nan, "<0.5", 4, 5, None, "NA", 7])
    assert df['imputed_value'].isna().tolist() == [False, False, True, False, False, False, True, True, False]
    assert not df.loc[2, 'is_imputed']
    assert df.attrs['n_missing'] == 3


def test_parser_formats_and_bools():
    v, s, _, _ = parse_censored(["5.", ".5", "< 2", ">1e3", "-3"])
    np.testing.assert_allclose(v, [5, 0.5, 2, 1000, -3])
    assert s.tolist() == [0, 0, -1, 1, 0]
    with pytest.raises(ValueError):
        parse_censored([True, 2.0])


def test_unknown_kwarg_raises():
    with pytest.raises(TypeError):
        impute([1., 2, 3, 4], [True, False, False, False], plotting_positon='simple')


def test_turnbull_reverse_km_for_left_censoring():
    rng = np.random.default_rng(8)
    x = rng.lognormal(0, 1, 60)
    L = rng.choice([0.5, 1.0], 60)
    c = x < L
    v = np.where(c, L, x)
    I, p = turnbull_em(np.where(c, -np.inf, v), v)
    # Reverse KM: F(t) = P(X <= t) estimated via KM on -X
    res = ecdf(CensoredData(uncensored=-v[~c], right=-v[c]))
    t = np.sort(v[~c])
    F_km = res.sf.evaluate(-t - 1e-12)  # P(-X > -t) = P(X < t) ... evaluated just above t
    F_tb = 1 - predict_turnbull(I, p, t)
    np.testing.assert_allclose(F_tb, F_km, atol=1e-8)


def test_mixed_sequential_bounds_and_lwp_rule():
    rng = np.random.default_rng(9)
    x = rng.lognormal(1, 1, 100)
    st = np.where(x < 1.5, -1, np.where(x > 12, 1, 0))
    v = np.where(st == -1, 1.5, np.where(st == 1, 12, x))
    df = impute(v, st, method='ros', random_state=1)
    assert np.all(df.loc[st == -1, 'imputed_value'] <= 1.5)
    assert np.all(df.loc[st == 1, 'imputed_value'] >= 12)
    small = impute(v[:20], st[:20], method='ros', lwp_compat=True, random_state=1)
    r = st[:20] == 1
    np.testing.assert_allclose(small.loc[r, 'imputed_value'], 1.1 * v[:20][r])


def test_interval_parametric():
    rng = np.random.default_rng(10)
    x = rng.lognormal(1, 0.5, 300)
    lo = np.floor(x)
    hi = lo + 1
    lo = np.where(lo == 0, 0.0, lo)
    df = impute(np.column_stack([lo, hi]), censoring_type='interval', method='parametric')
    assert np.all(df['imputed_value'] >= lo) and np.all(df['imputed_value'] <= hi)
    p = df.attrs['fit']['params']
    assert abs(p['mu'] - 1) < 0.1 and abs(p['sigma'] - 0.5) < 0.1
