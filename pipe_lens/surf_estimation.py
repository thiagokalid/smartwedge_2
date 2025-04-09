import numpy as np

from parameter_estimation import intsurf_estimation
from parameter_estimation.intsurf_estimation import img_line_improved

from bisect import bisect


def image_correction(ref, target, t_span):
    end_time = bisect(t_span, 56)
    ref = ref[:end_time, ...]
    target_ext = target[:end_time, ...]
    result = np.zeros_like(target)

    ref_max_idx = np.zeros(ref.shape[1])
    tar_max_idx = np.zeros(target_ext.shape[1])

    for i in range(ref.shape[1]):
        ref_max_idx[i] = np.argmax(ref[:, i])
        tar_max_idx[i] = np.argmax(target_ext[:, i]).astype(int)

    ref_mean = int(np.round(np.mean(ref_max_idx)))
    diff = tar_max_idx - ref_mean

    for i in range(target.shape[1]):
        result[:, i] = np.roll(target[:, i], -int(diff[i]))

    return result

def surfaces(img, threshold=0.99, lamb=1e-2, rho=100):
    max = np.max(img, axis=0)
    img = img / max
    a = img_line_improved(img, threshold)
    z = a[0].astype(int)
    w = np.diag((a[1]))
    idx, resf, kf, pk, sk = intsurf_estimation.profile_fadmm(w, z, lamb=lamb, x0=z, rho=rho,
                                                             eta=.999, itmax=25, tol=1e-3)
    idx = idx.astype(int)
    return idx