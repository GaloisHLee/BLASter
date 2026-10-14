# Stage 5 (D-2): Exact pruned enumeration + BKZ reference — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (native/inline) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This plan is the *implementation* half of the D-2 gate; the *spec* it argues from is `docs/spec/stage-5-bkz-enumeration.md`, which must be reviewed together with this plan before any code is written.

**Goal:** Transcribe FastInt64's `bkz_reduce`/`svp`/`enumeration` into exact `Fraction` arithmetic and certify the result with `verify_reduction(mode="bkz")`.

**Architecture:** A new `src/blaster/_bkz.py` holds the pure enumeration core (pruning table, GH radius, Schnorr–Euchner recursion); `src/blaster/reference.py` gains `_svp` + `bkz_reduce` that call it through the existing `_add_col`/`_swap_cols`/`_deeplll_local` machinery; `src/blaster/verify.py` gains the `"bkz"` mode gating `size_reduced_ok and deep_ok` with depth fixed at 4.

**Tech Stack:** Python stdlib (`math`, `fractions.Fraction`, `numbers.Integral`) + NumPy (input conversion only) + isolated Sage GSO accelerator. No new dependencies.

**Spec:** `docs/spec/stage-5-bkz-enumeration.md` (this plan argues from it; the executor reads both).

## Global Constraints

Copied verbatim from `docs/plans/2026-10-14-blaster-refinement-loop.md` §3 and the spec — every task inherits these:

- Reference reducer is a transcription, never calls Sage `LLL()`/`BKZ()`/fplll as the engine.
- No floating point in the verifier or reference except the GH radius (`math.lgamma`/`math.exp`/`math.log`/`math.pi`) and the `delta` dyadic path; every enumeration decision is exact `Fraction`.
- Arbitrary-precision everywhere on the exact path (matrix entries parse to Python `int`).
- Tie rule `nearest_core` (tie away from zero) is the only `round` used inside the enumeration; no site substitutes Python `round`.
- Column orientation `B0 · U == B`; row API `T · A0 == A` with `T = U^T`.
- Sage imports isolated to `verify_sage.py`; `blaster/__init__.py` never imports it.
- `unittest`, not pytest; `make test` and `make test-sage` are the authoritative suites.
- Never `git add t.py`.
- TDD: failing test first (RED → GREEN), both suites green, independent commit per item.
- End commit messages with `Co-Authored-By: Claude Code <noreply@anthropic.com>`.
- `beta` is an integer in `[2, 80]` (`bool`/non-integral rejected); `beta > n` selects the HKZ branch, so `beta <= n` is not required.

## Review Focus

The input classes / failure modes the spec implies but no task's test otherwise exercises (most likely to bite first):

1. **Odd block sizes.** `get_pruning_coefficients(3)` must slice the BKZ-4 row (`PRUNING_PARAMS[2]`) to its first 3 entries — an odd `w` must not read past the row or assume row length `== w`. *Pinned in Task 1.*
2. **The `l[i+1] == 0` first-visit test.** If this is coded as `l[i+1] == Fraction(0)` on a stale array, the zig-zag mis-steps and the enumeration loops forever or misses the short vector. *Pinned by the 2-dim trace test in Task 2.*
3. **`beta=2` short-circuit.** `bkz_reduce(beta=2)` must return after deep-LLL-4 and never call `svp`; `beta > n` must run the HKZ `svp(i, n-i)` shape, not the BKZ `svp(i, beta)` shape. *Pinned in Task 3.*
4. **Depth fixed at 4 in the verifier, independent of the user `depth` kwarg.** A caller passing `depth=1` with `mode="bkz"` must still gate on `deep_lll_holds(..., 4)`, and a size-reduced-but-not-deep basis must be rejected. *Pinned in Task 4.*
5. **Zero vector must never be returned.** The base case `l[0] == 0` early-return is what keeps the enumeration from emitting `sol = [0, …, 0]`; a 1-dim block with `r=[0]` or an all-zero radius path must return norm `Fraction(0)` (no solution), not a zero vector. *Pinned in Task 2.*

---

### Task 1: `_bkz.py` scaffolding — pruning table + GH radius

**Files:**
- Create: `src/blaster/_bkz.py`
- Test: `tests/test_bkz.py`

**Interfaces:**
- Produces: `PRUNING_PARAMS` (tuple of 41 int-tuples, index `k` holds `2k` values), `get_pruning_coefficients(block_size) -> list[Fraction]`, `_gh_squared(dim, log_volume) -> float`, `_safe_gh_squared(dim, log_volume) -> Fraction`.
- Consumes: `Fraction` (stdlib), `numbers.Integral`.

- [ ] **Step 1: Write the failing test**

```python
import math
import unittest
from fractions import Fraction

from blaster._bkz import (
    PRUNING_PARAMS,
    _gh_squared,
    _safe_gh_squared,
    get_pruning_coefficients,
)


class TestPruningParams(unittest.TestCase):
    def test_table_has_41_rows_and_2k_lengths(self):
        self.assertEqual(len(PRUNING_PARAMS), 41)
        for k in range(1, 41):
            self.assertEqual(len(PRUNING_PARAMS[k]), 2 * k)

    def test_bkz2_coefficients(self):
        self.assertEqual(
            get_pruning_coefficients(2),
            [Fraction(1, 1), Fraction(1, 10000)],
        )

    def test_odd_block_slices_next_even_row(self):
        # block 3 -> index (3+1)//2 = 2 (BKZ-4 row), first 3 of its 4 entries
        self.assertEqual(
            get_pruning_coefficients(3),
            [Fraction(1, 1), Fraction(7347, 10000), Fraction(4695, 10000)],
        )

    def test_block_size_range_and_type(self):
        for bad in (1, 81, 3.5, True):
            with self.assertRaises(ValueError):
                get_pruning_coefficients(bad)

    def test_gh_squared_dim2_volume_zero(self):
        # lgamma(2.0) == 0, so _gh_squared(2, 0.0) == exp(0)/pi == 1/pi
        self.assertEqual(_gh_squared(2, 0.0), 1.0 / math.pi)

    def test_safe_gh_squared_returns_exact_fraction(self):
        v = _safe_gh_squared(20, math.log(2.0))
        self.assertIsInstance(v, Fraction)
        self.assertGreater(v, 0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n sage106 env PYTHONPATH=src python -m unittest tests.test_bkz -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'blaster._bkz'`.

- [ ] **Step 3: Write minimal implementation** (`src/blaster/_bkz.py`)

```python
"""Exact pruned-enumeration core for BKZ (Stage 5).

Pure exact-rational transcription of FastInt64 ``enumeration`` / ``svp`` /
``bkz_reduce``. The only floating point is the GH radius's transcendental
``lgamma``/``exp``/``log``/``pi``, snapped to an exact :class:`~fractions.Fraction`
before any enumeration decision. This module never imports Sage.
"""
import math
from fractions import Fraction
from numbers import Integral

from ._exact import nearest_core

# pruned-GNR coefficients, integer x10000 form (index k holds block size 2k),
# parsed programmatically from core/pruning_params.cpp.
PRUNING_PARAMS = (
    (),
    (10000, 1),
    (10000, 7347, 4695, 395),
    (10000, 8297, 6595, 4725, 3040, 2369),
    (10000, 9040, 8081, 6523, 4992, 3503, 2606, 2169),
    (10000, 10000, 9966, 9966, 5197, 5197, 3264, 3264, 3190, 3190),
    (10000, 10000, 9900, 9900, 6683, 6683, 4022, 4022, 3848, 3848, 3848, 3848),
    (10000, 10000, 10000, 9452, 8287, 7184, 5753, 5091, 4316, 3494, 3046, 2968, 2946, 2930),
    (10000, 10000, 10000, 10000, 9305, 9305, 5958, 5958, 4306, 4306, 3598, 3598, 3564, 3564, 3564, 3564),
    (10000, 10000, 10000, 10000, 9279, 8667, 7565, 6663, 5543, 4969, 4252, 3901, 3255, 3167, 3133, 3115, 3096, 3078),
    (10000, 10000, 10000, 10000, 9307, 8839, 8004, 7443, 6453, 5792, 4931, 4417, 3841, 3342, 3306, 3299, 3299, 3299, 3299, 3299),
    (10000, 10000, 10000, 10000, 9708, 9357, 8518, 7911, 7070, 6378, 5536, 5053, 4485, 4075, 3581, 3247, 2910, 2910, 2910, 2910, 2910, 2910),
    (10000, 10000, 10000, 10000, 9725, 9691, 8760, 8735, 7544, 7519, 6280, 6236, 5157, 5077, 4169, 4161, 3304, 3176, 2547, 2490, 1783, 1741, 1680, 1679),
    (10000, 10000, 10000, 10000, 9769, 9681, 8912, 8891, 7798, 7759, 6649, 6580, 5590, 5495, 4639, 4540, 3804, 3706, 3068, 2969, 2422, 2418, 1881, 1777, 1571, 1571),
    (10000, 10000, 10000, 10000, 9812, 9773, 9050, 9024, 8053, 8053, 6992, 6972, 5969, 5944, 5061, 5018, 4260, 4201, 3561, 3479, 2972, 2849, 2405, 2284, 1852, 1787, 1673, 1673),
    (10000, 10000, 10000, 10000, 9847, 9806, 9182, 9156, 8292, 8269, 7291, 7272, 6322, 6307, 5438, 5421, 4656, 4625, 3978, 3919, 3364, 3295, 2821, 2729, 2338, 2219, 1909, 1767, 1458, 1449),
    (10000, 10000, 10000, 10000, 9885, 9885, 9292, 9273, 8484, 8456, 7540, 7504, 6631, 6579, 5787, 5721, 5014, 4946, 4322, 4255, 3711, 3642, 3165, 3092, 2681, 2593, 2251, 2143, 1887, 1762, 1488, 1274),
    (10000, 10000, 10000, 10000, 10000, 10000, 9523, 9523, 8683, 8683, 7915, 7915, 7073, 7073, 6227, 6227, 5357, 5357, 4749, 4749, 4107, 4107, 3527, 3527, 2998, 2998, 2515, 2515, 2078, 2078, 1729, 1729, 1705, 1705),
    (10000, 10000, 10000, 10000, 9947, 9903, 9493, 9453, 8848, 8809, 8000, 7977, 7174, 7143, 6389, 6350, 5656, 5616, 4991, 4951, 4396, 4354, 3864, 3816, 3387, 3326, 2962, 2885, 2584, 2490, 2270, 2173, 1947, 1880, 1598, 1470),
    (10000, 10000, 10000, 10000, 9935, 9918, 9622, 9563, 8949, 8921, 8212, 8186, 7432, 7409, 6665, 6648, 5948, 5934, 5292, 5277, 4693, 4675, 4146, 4121, 3662, 3631, 3226, 3188, 2837, 2787, 2491, 2424, 2202, 2119, 1918, 1902, 1593, 1591),
    (10000, 10000, 10000, 10000, 9945, 9945, 9671, 9615, 9091, 9065, 8403, 8378, 7662, 7640, 6926, 6910, 6233, 6218, 5592, 5576, 5001, 4983, 4462, 4440, 3977, 3951, 3541, 3509, 3148, 3106, 2798, 2741, 2488, 2410, 2149, 2139, 1997, 1970, 1874, 1874),
    (10000, 10000, 10000, 10000, 10000, 9999, 9709, 9673, 9223, 9187, 8560, 8536, 7858, 7829, 7162, 7126, 6490, 6453, 5860, 5823, 5279, 5242, 4747, 4710, 4261, 4223, 3822, 3780, 3423, 3375, 3061, 3002, 2731, 2662, 2408, 2387, 2174, 2088, 2057, 2056, 2055, 2054),
    (10000, 10000, 10000, 10000, 10000, 10000, 9775, 9750, 9315, 9279, 8708, 8650, 8049, 7974, 7378, 7299, 6727, 6648, 6110, 6035, 5535, 5464, 5007, 4939, 4524, 4459, 4082, 4017, 3682, 3618, 3318, 3248, 2984, 2907, 2696, 2696, 2378, 2366, 2156, 2151, 2145, 2145, 2145, 2145),
    (10000, 10000, 10000, 10000, 10000, 10000, 9818, 9793, 9356, 9356, 8842, 8769, 8216, 8130, 7573, 7483, 6941, 6854, 6337, 6255, 5772, 5696, 5250, 5179, 4771, 4706, 4329, 4268, 3929, 3869, 3561, 3499, 3224, 3158, 2917, 2845, 2652, 2581, 2407, 2398, 2254, 2253, 2253, 2253, 2253, 2253),
    (10000, 10000, 10000, 10000, 10000, 10000, 9896, 9840, 9498, 9429, 8881, 8881, 8386, 8260, 7763, 7634, 7152, 7026, 6567, 6446, 6015, 5900, 5500, 5391, 5025, 4925, 4594, 4508, 4159, 4072, 3798, 3719, 3459, 3384, 3146, 3086, 2858, 2760, 2582, 2505, 2479, 2479, 2479, 2479, 2479, 2479, 2479, 2479),
    (10000, 10000, 10000, 10000, 10000, 10000, 10000, 9996, 9683, 9650, 9234, 9034, 8593, 8418, 7963, 7813, 7358, 7221, 6781, 6648, 6227, 6092, 5692, 5564, 5212, 5133, 4835, 4750, 4391, 4300, 4044, 3950, 3679, 3594, 3351, 3271, 3017, 2909, 2743, 2739, 2602, 2591, 2587, 2583, 2579, 2575, 2571, 2566, 2562, 2558),
    (10000, 10000, 10000, 10000, 10000, 10000, 9899, 9898, 9588, 9559, 9138, 9106, 8600, 8567, 8020, 7987, 7448, 7416, 6872, 6843, 6313, 6313, 5822, 5799, 5343, 5325, 4909, 4893, 4512, 4497, 4134, 4123, 3800, 3792, 3497, 3488, 3232, 3209, 2992, 2949, 2772, 2712, 2510, 2482, 2417, 2417, 2417, 2417, 2417, 2417, 2417, 2417),
    (10000, 10000, 10000, 10000, 10000, 10000, 9917, 9892, 9645, 9619, 9232, 9202, 8722, 8692, 8178, 8145, 7619, 7587, 7069, 7038, 6537, 6509, 6029, 6004, 5554, 5554, 5112, 5096, 4718, 4704, 4353, 4340, 4002, 3991, 3694, 3682, 3415, 3394, 3165, 3130, 2939, 2888, 2692, 2667, 2504, 2501, 2401, 2401, 2401, 2401, 2401, 2401, 2401, 2401),
    (10000, 10000, 10000, 10000, 10000, 10000, 9948, 9946, 9672, 9672, 9312, 9286, 8836, 8809, 8315, 8289, 7780, 7753, 7248, 7220, 6731, 6703, 6239, 6212, 5766, 5743, 5333, 5314, 4917, 4902, 4554, 4547, 4206, 4193, 3894, 3882, 3607, 3590, 3347, 3321, 3111, 3074, 2874, 2858, 2707, 2645, 2503, 2501, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500),
    (10000, 10000, 10000, 10000, 10000, 10000, 9947, 9947, 9742, 9716, 9389, 9361, 8942, 8914, 8446, 8420, 7931, 7906, 7416, 7391, 6911, 6886, 6423, 6401, 5966, 5941, 5532, 5520, 5124, 5106, 4757, 4750, 4407, 4397, 4084, 4073, 3797, 3783, 3531, 3510, 3288, 3258, 3067, 3026, 2839, 2822, 2660, 2654, 2514, 2513, 2512, 2512, 2512, 2512, 2512, 2512, 2512, 2512),
    (10000, 10000, 10000, 10000, 10000, 10000, 9962, 9962, 9779, 9752, 9458, 9426, 9039, 9008, 8569, 8541, 8072, 8046, 7573, 7554, 7082, 7056, 6603, 6585, 6151, 6133, 5717, 5697, 5314, 5313, 4956, 4942, 4589, 4577, 4285, 4274, 3995, 3970, 3709, 3693, 3459, 3435, 3225, 3194, 3010, 2970, 2803, 2758, 2681, 2679, 2679, 2679, 2679, 2679, 2679, 2679, 2679, 2679, 2679, 2679),
    (10000, 10000, 10000, 10000, 10000, 10000, 9993, 9989, 9810, 9774, 9519, 9471, 9129, 9078, 8678, 8632, 8198, 8159, 7710, 7678, 7229, 7201, 6761, 6738, 6314, 6294, 5891, 5872, 5491, 5475, 5118, 5103, 4773, 4759, 4450, 4436, 4152, 4138, 3875, 3860, 3619, 3602, 3383, 3362, 3168, 3141, 2974, 2936, 2854, 2854, 2652, 2648, 2648, 2648, 2648, 2648, 2648, 2648, 2648, 2648, 2648, 2648),
    (10000, 10000, 10000, 10000, 10000, 10000, 10000, 9996, 9837, 9794, 9576, 9510, 9212, 9139, 8782, 8714, 8319, 8261, 7844, 7797, 7372, 7335, 6913, 6884, 6472, 6449, 6054, 6035, 5659, 5644, 5283, 5266, 4948, 4936, 4617, 4601, 4323, 4307, 4038, 4022, 3776, 3758, 3534, 3513, 3315, 3291, 3125, 3109, 2983, 2895, 2823, 2823, 2702, 2702, 2702, 2702, 2702, 2702, 2702, 2702, 2702, 2702, 2702, 2702),
    (10000, 10000, 10000, 10000, 10000, 10000, 10000, 9998, 9848, 9800, 9634, 9545, 9315, 9206, 8903, 8795, 8451, 8355, 7987, 7903, 7524, 7452, 7072, 7010, 6635, 6583, 6218, 6174, 5824, 5785, 5454, 5419, 5108, 5077, 4785, 4756, 4478, 4446, 4201, 4169, 3942, 3910, 3703, 3667, 3482, 3442, 3333, 3333, 3098, 3049, 2919, 2911, 2785, 2782, 2782, 2782, 2782, 2782, 2782, 2782, 2782, 2782, 2782, 2782, 2782, 2782),
    (10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 9907, 9902, 9800, 9800, 9359, 9308, 9035, 8904, 8571, 8467, 8113, 8026, 7660, 7586, 7216, 7154, 6786, 6734, 6375, 6330, 5984, 5945, 5615, 5582, 5270, 5240, 4946, 4919, 4637, 4608, 4359, 4329, 4096, 4067, 3854, 3822, 3630, 3593, 3423, 3380, 3230, 3182, 3033, 3019, 2864, 2854, 2837, 2837, 2837, 2837, 2837, 2837, 2837, 2837, 2837, 2837, 2837, 2837, 2837, 2837),
    (10000, 10000, 10000, 10000, 10000, 10000, 10000, 9995, 9902, 9889, 9832, 9800, 9444, 9399, 9140, 8990, 8688, 8559, 8237, 8124, 7792, 7693, 7355, 7270, 6931, 6858, 6523, 6461, 6136, 6082, 5769, 5722, 5425, 5384, 5102, 5067, 4788, 4749, 4511, 4471, 4244, 4209, 3999, 3960, 3770, 3729, 3559, 3514, 3362, 3314, 3171, 3148, 3029, 3023, 2889, 2886, 2885, 2885, 2885, 2885, 2885, 2885, 2885, 2885, 2885, 2885, 2885, 2885, 2885, 2885),
    (10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 9913, 9881, 9805, 9800, 9800, 9800, 9268, 9161, 8853, 8688, 8364, 8240, 7914, 7814, 7484, 7399, 7068, 6995, 6666, 6603, 6282, 6228, 5917, 5870, 5573, 5531, 5249, 5212, 4947, 4916, 4651, 4613, 4392, 4355, 4142, 4107, 3912, 3875, 3698, 3657, 3500, 3454, 3316, 3265, 3119, 3092, 2958, 2946, 2929, 2929, 2929, 2929, 2929, 2929, 2929, 2929, 2929, 2929, 2929, 2929, 2929, 2929, 2929, 2929),
    (10000, 10000, 10000, 10000, 10000, 10000, 10000, 9998, 9920, 9909, 9890, 9890, 9800, 9800, 9538, 9538, 8956, 8848, 8509, 8352, 8041, 7920, 7612, 7510, 7201, 7112, 6805, 6727, 6424, 6357, 6061, 6002, 5718, 5665, 5394, 5347, 5089, 5046, 4805, 4768, 4528, 4484, 4285, 4241, 4049, 4007, 3833, 3788, 3632, 3583, 3444, 3391, 3246, 3215, 3086, 3049, 2978, 2974, 2974, 2974, 2974, 2974, 2974, 2974, 2974, 2974, 2974, 2974, 2974, 2974, 2974, 2974, 2974, 2974),
    (10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 9936, 9908, 9830, 9826, 9800, 9800, 9800, 9800, 9188, 9161, 8697, 8498, 8161, 8040, 7725, 7636, 7322, 7246, 6934, 6868, 6556, 6500, 6193, 6146, 5849, 5809, 5524, 5490, 5218, 5188, 4934, 4905, 4652, 4626, 4415, 4379, 4176, 4147, 3961, 3928, 3760, 3721, 3570, 3527, 3393, 3346, 3219, 3203, 3067, 3015, 2984, 2983, 2983, 2983, 2983, 2983, 2983, 2983, 2983, 2983, 2983, 2983, 2983, 2983, 2983, 2983, 2983, 2983),
    (10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 9934, 9913, 9877, 9877, 9854, 9854, 9800, 9800, 9484, 9484, 8803, 8798, 8334, 8206, 7854, 7795, 7445, 7410, 7062, 7036, 6683, 6665, 6315, 6305, 5964, 5959, 5633, 5633, 5324, 5302, 5037, 5036, 4758, 4722, 4512, 4512, 4287, 4284, 4075, 4073, 3886, 3875, 3702, 3679, 3521, 3492, 3350, 3317, 3179, 3159, 3048, 3037, 3031, 3031, 3031, 3031, 3031, 3031, 3031, 3031, 3031, 3031, 3031, 3031, 3031, 3031, 3031, 3031, 3031, 3031),
    (10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 9937, 9911, 9869, 9867, 9853, 9853, 9800, 9800, 9623, 9623, 8995, 8926, 8496, 8281, 7976, 7852, 7558, 7466, 7176, 7097, 6809, 6740, 6455, 6396, 6117, 6066, 5795, 5750, 5492, 5455, 5207, 5167, 4933, 4909, 4669, 4632, 4438, 4400, 4210, 4177, 4001, 3966, 3806, 3768, 3624, 3582, 3454, 3409, 3297, 3249, 3153, 3093, 3064, 3063, 3063, 3063, 3063, 3063, 3063, 3063, 3063, 3063, 3063, 3063, 3063, 3063, 3063, 3063, 3063, 3063, 3063, 3063),
)


def get_pruning_coefficients(block_size):
    """Pruning coefficients for a block, as exact ``Fraction``s.

    Mirrors FastInt64 ``get_pruning_coefficients``: row index
    ``(block_size + 1) // 2`` (integer division) holds the BKZ-``2k`` row; an
    odd block consumes the first ``block_size`` entries of the next even row.
    """
    if isinstance(block_size, bool) or not isinstance(block_size, Integral):
        raise ValueError("block_size must be an integer")
    block_size = int(block_size)
    if block_size < 2 or block_size > 80:
        raise ValueError("block_size must be in [2, 80]")
    return [Fraction(v, 10000) for v in PRUNING_PARAMS[(block_size + 1) // 2][:block_size]]


def _gh_squared(dim, log_volume):
    """FastInt64 ``gh_squared``: Gaussian-heuristic squared norm (binary64)."""
    log_gamma = math.lgamma(dim / 2.0 + 1.0)
    return math.exp(2.0 * (log_gamma + log_volume) / dim) / math.pi


def _safe_gh_squared(dim, log_volume):
    """FastInt64 ``safe_gh_squared`` snapped to an exact ``Fraction``."""
    gh2 = _gh_squared(dim, log_volume)
    factor = max(1.05, min(2.0, 1.0 + 4.0 / dim))
    return Fraction(*(gh2 * factor * factor).as_integer_ratio())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n sage106 env PYTHONPATH=src python -m unittest tests.test_bkz -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/blaster/_bkz.py tests/test_bkz.py
git commit -m "test(bkz): pin pruning table and GH radius helpers (RED -> GREEN)

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 2: `enumeration` exact core

**Files:**
- Modify: `src/blaster/_bkz.py` (add `enumeration`)
- Test: `tests/test_bkz.py` (add `TestEnumeration`)

**Interfaces:**
- Consumes: `nearest_core` (from `_exact.py`), `Fraction`.
- Produces: `enumeration(mu, r, pr, radius) -> (sol: list[int], norm: Fraction)` where `mu` is an `n x n` block GSO matrix (`mu[a][b]` = GSO coeff for `a < b`, else `0`), `r` is `[r_0..r_{n-1}]`, `pr` is `len-n` coefficients, `radius` a `Fraction`; returns the shortest vector found and its squared norm, or `(sol, Fraction(0))` when no nonzero vector beats `radius` (mirrors `_A < radius ? _A : 0.0`).

- [ ] **Step 1: Write the failing test**

```python
from blaster._bkz import enumeration


class TestEnumeration(unittest.TestCase):
    def test_single_dim_finds_shortest_nonzero(self):
        # 1-dim block, r=[4], radius=5 -> shortest nonzero is +-2 (norm 4)
        mu = [[Fraction(0)]]
        r = [Fraction(4)]
        pr = [Fraction(1)]
        sol, norm = enumeration(mu, r, pr, Fraction(5))
        self.assertEqual(norm, Fraction(4))
        self.assertEqual(sol, [2])

    def test_single_dim_no_solution_is_zero_norm(self):
        # radius already <= shortest nonzero norm -> returns (sol, 0)
        mu = [[Fraction(0)]]
        r = [Fraction(4)]
        pr = [Fraction(1)]
        _, norm = enumeration(mu, r, pr, Fraction(4))
        self.assertEqual(norm, Fraction(0))

    def test_two_dim_unpruned_finds_combination(self):
        # r=[1, 1/4], mu[0][1]=3/4, no pruning. norm^2(x0,x1) =
        # (x0 + 3/4 x1)^2 + x1^2/4; minimum over Z^2\{0} is (-1,1) -> 5/16.
        mu = [[Fraction(0), Fraction(3, 4)], [Fraction(0), Fraction(0)]]
        r = [Fraction(1), Fraction(1, 4)]
        pr = [Fraction(1), Fraction(1)]
        sol, norm = enumeration(mu, r, pr, Fraction(1))
        self.assertEqual(norm, Fraction(5, 16))
        self.assertEqual(sol, [-1, 1])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n sage106 env PYTHONPATH=src python -m unittest tests.test_bkz.TestEnumeration -v`
Expected: FAIL with `ImportError: cannot import name 'enumeration' from 'blaster._bkz'`.

- [ ] **Step 3: Write minimal implementation** (append to `src/blaster/_bkz.py`)

```python
def enumeration(mu, r, pr, radius):
    """Exact Schnorr--Euchner pruned enumeration (transcription of ``lattice_enum_t``).

    Returns ``(sol, norm)``: ``sol`` is the best coefficient vector (ints) and
    ``norm`` its exact squared norm, or ``Fraction(0)`` if no nonzero vector is
    strictly shorter than ``radius``.
    """
    n = len(r)
    A = radius
    AA = [A * pr[k] for k in range(n)]
    x = [0] * n
    Dx = [0] * n
    D2x = [-1] * n
    c = [Fraction(0)] * n
    rr = [n - 1] * n
    l = [Fraction(0)] * (n + 1)
    sigT = [[Fraction(0)] * n for _ in range(n)]
    sol = [0] * n

    def update_AA():
        for k in range(n):
            AA[k] = A * pr[k]

    def recur(i):
        nonlocal A
        if i == -1:
            if l[0] > A or l[0] == 0:
                return
            for j in range(n):
                sol[j] = x[j]
            A = l[0]
            update_AA()
            return
        ci = sigT[i][i]
        xi = nearest_core(ci)
        yi = ci - xi
        li = l[i + 1] + yi * yi * r[i]
        if li > AA[i]:
            return
        Dx[i] = D2x[i] = 1 if yi >= 0 else -1
        c[i] = ci
        x[i] = xi
        l[i] = li
        if i >= 1:
            if rr[i] > rr[i - 1]:
                rr[i - 1] = rr[i]
            for j in range(rr[i - 1], i - 1, -1):
                sigT[i - 1][j - 1] = sigT[i - 1][j] - x[j] * mu[i - 1][j]
        while True:
            recur(i - 1)
            if l[i + 1] == 0:
                x[i] += 1
            else:
                x[i] += Dx[i]
                D2x[i] = -D2x[i]
                Dx[i] = D2x[i] - Dx[i]
            if i >= 1:
                rr[i - 1] = i
            yi2 = c[i] - x[i]
            li2 = l[i + 1] + yi2 * yi2 * r[i]
            if li2 > AA[i]:
                return
            l[i] = li2
            if i >= 1:
                sigT[i - 1][i - 1] = sigT[i - 1][i] - x[i] * mu[i - 1][i]

    recur(n - 1)
    return sol, (A if A < radius else Fraction(0))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n sage106 env PYTHONPATH=src python -m unittest tests.test_bkz -v`
Expected: PASS (all `TestPruningParams` + `TestEnumeration`).

- [ ] **Step 5: Commit**

```bash
git add src/blaster/_bkz.py tests/test_bkz.py
git commit -m "feat(bkz): exact pruned Schnorr-Euchner enumeration core

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 3: `_svp` + `bkz_reduce` in the reference engine

**Files:**
- Modify: `src/blaster/reference.py` (add `_svp`, `_beta_param`, `bkz_reduce`; extend imports to include `_safe_gh_squared`, `get_pruning_coefficients`, `enumeration` from `_bkz` and `Fraction`, `math`)
- Test: `tests/test_bkz.py` (add `TestBKZReduce`)

**Interfaces:**
- Consumes: `to_int_matrix`, `column`, `transpose_cols`, `full_column_rank`, `delta_to_fraction` (from `_exact`); `_add_col`, `_swap_cols`, `_deeplll_local` (from `reference`); `resolve_gso`; `_bkz` helpers.
- Produces: `_svp(cols, Ucols, i, w, delta, tr, gso)` (in place), `bkz_reduce(B0, delta=0.99, *, beta, gso=None) -> (cols, U_rows, transcript)`.

- [ ] **Step 1: Write the failing test**

```python
from blaster.reference import bkz_reduce, _deeplll_local
from blaster._exact import to_int_matrix, column, transpose_cols, delta_to_fraction
from blaster.verify_sage import resolve_gso


def _deeplll4_reference(B0, delta=0.99):
    B0_rows = to_int_matrix(B0, name="B0")
    n = len(B0_rows[0])
    d = delta_to_fraction(delta)
    cols = [column(B0_rows, j) for j in range(n)]
    Ucols = [[1 if r == j else 0 for r in range(n)] for j in range(n)]
    tr = []
    _deeplll_local(cols, Ucols, 0, n, d, 4, tr, resolve_gso(None))
    return cols, transpose_cols(Ucols), tr


class TestBKZReduce(unittest.TestCase):
    def test_bkz2_is_deeplll4(self):
        B0 = [[2, 3, 5], [0, 4, 6], [0, 0, 7]]
        cols_a, U_a, tr_a = bkz_reduce(B0, beta=2)
        cols_b, U_b, tr_b = _deeplll4_reference(B0)
        self.assertEqual(cols_a, cols_b)
        self.assertEqual(U_a, U_b)
        self.assertEqual(tr_a, tr_b)

    def test_bkz_tour_verifies(self):
        B0 = [[2, 3, 5, 7, 11], [0, 4, 6, 8, 10], [0, 0, 7, 9, 12],
              [0, 0, 0, 11, 13], [0, 0, 0, 0, 17]]
        cols, U_rows, tr = bkz_reduce(B0, beta=3)
        rep = verify_reduction(B0, transpose_cols(cols), U_rows, mode="bkz")
        self.assertTrue(rep.verified)

    def test_hkz_tour_verifies(self):
        B0 = [[2, 3, 5], [0, 4, 6], [0, 0, 7]]
        cols, U_rows, tr = bkz_reduce(B0, beta=4)
        rep = verify_reduction(B0, transpose_cols(cols), U_rows, mode="bkz")
        self.assertTrue(rep.verified)

    def test_bad_beta_rejected(self):
        B0 = [[2, 3], [0, 4]]
        for bad in (1, 81, 3.5, True):
            with self.assertRaises(ValueError):
                bkz_reduce(B0, beta=bad)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n sage106 env PYTHONPATH=src python -m unittest tests.test_bkz.TestBKZReduce -v`
Expected: FAIL with `ImportError: cannot import name 'bkz_reduce' from 'blaster.reference'`.

- [ ] **Step 3: Write minimal implementation** (append to `src/blaster/reference.py`, and extend the existing import block)

Extend imports (add `import math`, `from fractions import Fraction`, and the `_bkz` names):

```python
import math
from fractions import Fraction
from numbers import Integral

from ._bkz import _safe_gh_squared, enumeration, get_pruning_coefficients
```

Append the functions:

```python
def _svp(cols, Ucols, i, w, delta, tr, gso):
    """Exact transcription of FastInt64 ``svp``: pruned block enumeration.

    Enumerates the ``w``-sublattice at ``b_i..b_{i+w-1}`` under the safe GH
    radius, inserts a found short vector at ``i`` (one column add + ``j``
    adjacent swaps), then re-deep-reduces ``[0, min(i+w, n))`` with depth 4.
    """
    mu, r = gso(cols)
    mu_b = [[mu[i + a][i + b] for b in range(w)] for a in range(w)]
    r_b = [r[i + a] for a in range(w)]
    log_volume = sum(math.log(float(r[i + a])) for a in range(w))
    expected = min(Fraction(1023, 1024) * r[i], _safe_gh_squared(w, log_volume))
    pr = get_pruning_coefficients(w)
    sol, norm = enumeration(mu_b, r_b, pr, expected)
    if norm > 0:
        j = w - 1
        while j > 0 and sol[j] == 0:
            j -= 1
        if j > 0 and sol[j] == -1:
            for k in range(j + 1):
                sol[k] = -sol[k]
        if j > 0 and sol[j] == 1:
            for k in range(j):
                if sol[k]:
                    _add_col(cols, i + k, i + j, sol[k])
                    _add_col(Ucols, i + k, i + j, sol[k])
                    tr.append(("add", i + k, i + j, sol[k], "enum"))
            for p in range(j - 1, -1, -1):
                _swap_cols(cols, i + p, i + p + 1)
                _swap_cols(Ucols, i + p, i + p + 1)
                tr.append(("swap", i + p, i + p + 1))
    _deeplll_local(cols, Ucols, 0, min(i + w, len(cols)), delta, 4, tr, gso)


def _beta_param(beta):
    if isinstance(beta, bool) or not isinstance(beta, Integral):
        raise ValueError("beta must be an integer")
    beta = int(beta)
    if beta < 2 or beta > 80:
        raise ValueError("beta must be in [2, 80]")
    return beta


def bkz_reduce(B0, delta=0.99, *, beta, gso=None):
    """Exact BKZ reduction returning ``(cols, U_rows, transcript)``.

    Pre-reduces the whole basis with deep-LLL-4, then runs a progressive BKZ
    tour (``svp(i, beta)`` for each block start) or, when ``beta > n``, an HKZ
    tour (``svp(i, n-i)``). ``beta <= 2`` returns immediately after the
    pre-reduction (no ``svp`` is called). This never calls Sage ``BKZ``.
    """
    B0_rows = to_int_matrix(B0, name="B0")
    if not full_column_rank(B0_rows):
        raise ValueError("basis vectors must be linearly independent")
    n = len(B0_rows[0])
    d = delta_to_fraction(delta)
    beta = _beta_param(beta)
    gso_fn = resolve_gso(gso)

    cols = [column(B0_rows, j) for j in range(n)]
    Ucols = [[1 if r == j else 0 for r in range(n)] for j in range(n)]
    tr = []

    _deeplll_local(cols, Ucols, 0, n, d, 4, tr, gso_fn)
    if beta <= 2:
        return cols, transpose_cols(Ucols), tr
    if beta > n:
        for i in range(n - 1):
            _svp(cols, Ucols, i, n - i, d, tr, gso_fn)
    else:
        for i in range(n - beta + 1):
            _svp(cols, Ucols, i, beta, d, tr, gso_fn)
    return cols, transpose_cols(Ucols), tr
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n sage106 env PYTHONPATH=src python -m unittest tests.test_bkz -v`
Expected: FAIL with `ImportError: cannot import name 'verify_reduction'` in `TestBKZReduce` (the `"bkz"` mode does not exist yet). **This is the expected RED for Task 4, not a Task 3 defect.** To prove Task 3's own behavior is green, temporarily gate the two `*_verifies` tests on `verify_reduction(..., mode="deeplll")` is not possible (depth differs); instead run only the non-verifier tests to confirm Task 3:

Run: `conda run -n sage106 env PYTHONPATH=src python -m unittest tests.test_bkz.TestBKZReduce.test_bkz2_is_deeplll4 tests.test_bkz.TestBKZReduce.test_bad_beta_rejected -v`
Expected: PASS (2 tests) — `bkz_reduce(beta=2)` matches `_deeplll_local(..., depth=4)` exactly and bad `beta` is rejected. The two `*_verifies` tests remain RED until Task 4 lands.

- [ ] **Step 5: Commit**

```bash
git add src/blaster/reference.py tests/test_bkz.py
git commit -m "feat(bkz): exact svp insertion and bkz_reduce tour

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 4: `verify_reduction(mode="bkz")`

**Files:**
- Modify: `src/blaster/verify.py` (mode registry + deep-depth selection + gating)
- Test: `tests/test_bkz.py` (add `TestVerifyBKZ`)

**Interfaces:**
- Consumes: `deep_lll_holds` (from `_exact`), the existing `VerificationReport`.
- Produces: `verify_reduction(..., mode="bkz")` gates `verified = size_reduced_ok and deep_ok` with `deep_lll_holds(mu, r, d, 4)`.

- [ ] **Step 1: Write the failing test**

```python
from blaster.verify import verify_reduction


class TestVerifyBKZ(unittest.TestCase):
    def test_bkz_mode_accepted_and_verifies(self):
        B0 = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
        B = B0
        U = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        rep = verify_reduction(B0, B, U, mode="bkz")
        self.assertTrue(rep.verified)
        self.assertEqual(rep.mode, "bkz")
        self.assertTrue(rep.size_reduced_ok)
        self.assertTrue(rep.deep_ok)

    def test_bkz_mode_rejects_not_deep(self):
        # size-reduced but NOT deep-LLL-4: b_1 is far shorter than b_0's norm.
        B0 = [[10, 0], [0, 1]]
        B = B0
        U = [[1, 0], [0, 1]]
        rep = verify_reduction(B0, B, U, mode="bkz")
        self.assertFalse(rep.verified)
        self.assertTrue(rep.structural_ok)
        self.assertTrue(rep.size_reduced_ok)
        self.assertFalse(rep.deep_ok)

    def test_bkz_mode_ignores_depth_kwarg(self):
        # depth=1 must NOT override the fixed depth-4 gating for bkz.
        B0 = [[10, 0], [0, 1]]
        B = B0
        U = [[1, 0], [0, 1]]
        rep = verify_reduction(B0, B, U, mode="bkz", depth=1)
        self.assertFalse(rep.verified)
        self.assertFalse(rep.deep_ok)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n sage106 env PYTHONPATH=src python -m unittest tests.test_bkz.TestVerifyBKZ -v`
Expected: FAIL with `ValueError: unknown reduction mode 'bkz'; expected 'lll', 'seysen' or 'deeplll'`.

- [ ] **Step 3: Write minimal implementation** (edit `src/blaster/verify.py`)

Change the mode registry (line 87):

```python
    if mode not in ("lll", "seysen", "deeplll", "bkz"):
        raise ValueError(f"unknown reduction mode {mode!r}; expected 'lll', 'seysen', 'deeplll' or 'bkz'")
```

Change the deep-depth selection (line 119):

```python
    deep_ok = deep_lll_holds(mu, r, d, 4 if mode == "bkz" else (_depth_param(depth) if mode == "deeplll" else 1))
```

Change the gating (lines 121–126):

```python
    if mode == "seysen":
        verified = size_reduced_ok
    elif mode in ("deeplll", "bkz"):
        verified = size_reduced_ok and deep_ok
    else:
        verified = weak_lll_ok and size_reduced_ok
```

Also add `"bkz"` to the `mode` docstring (lines 75–85):

```python
    * ``"bkz"`` — size reduction **and** the deep condition, the latter gated
      with a fixed depth of 4 (the BKZ/HKZ tour's final whole-basis deep-LLL-4),
      independent of the user ``depth`` kwarg.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n sage106 env PYTHONPATH=src python -m unittest tests.test_bkz -v`
Expected: PASS (all `TestPruningParams`, `TestEnumeration`, `TestBKZReduce`, `TestVerifyBKZ`).

- [ ] **Step 5: Commit**

```bash
git add src/blaster/verify.py tests/test_bkz.py
git commit -m "feat(verify): add mode=bkz postcondition (size + deep-LLL-4)

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 5: Full-suite verification

**Files:** none (verification only).

- [ ] **Step 1: Run the authoritative suites**

Run: `make test` then `make test-sage` (each runs `build_ext --inplace` first).
Expected: both green, no regressions in the existing LLL/Seysen/DeepLLL suites.

- [ ] **Step 2: Confirm no `t.py` is staged**

Run: `git status --porcelain`
Expected: `t.py` still untracked (never staged).

- [ ] **Step 3: (No commit — verification task)**

---

## Self-review

- **Spec coverage:** §2 mapping → Tasks 1–2 (`Fraction` table, `nearest_core`, `risq`→`r`); §3 GH/pruning → Task 1; §4 recursion → Task 2; §5 svp/tour → Task 3; §6 verifier contract → Task 4; §8 corruption property → Task 4 `test_bkz_mode_rejects_not_deep`.
- **Placeholder scan:** none — every code step carries full source.
- **Type consistency:** `enumeration(mu, r, pr, radius) -> (list[int], Fraction)` is produced by Task 2 and consumed by `_svp` in Task 3; `bkz_reduce(B0, delta=0.99, *, beta, gso=None) -> (cols, U_rows, tr)` is produced by Task 3 and consumed by `verify_reduction(..., mode="bkz")` in Task 4.
- **Review Focus:** all five lines are pinned (Task 1 odd-block; Task 2 first-visit + zero-vector; Task 3 beta=2/HKZ; Task 4 fixed depth-4).
