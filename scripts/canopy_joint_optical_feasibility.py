"""Conditional sparse optical-depth feasibility; never opacity supervision.

K must come from fixed-geometry foreground support. Bounds must carry their
observation provenance. Results concern this restricted uncapped submodel only;
native cap/cutoff/depth ordering and all other views must be checked separately.
"""
import numpy as np
from scipy import sparse
from scipy.optimize import linprog


def solve_optical_intervals(kernel, lower, upper, tau_upper, *, weights=None, reference_tau=None, time_limit=120.):
    K = sparse.csr_matrix(kernel, dtype=np.float64)
    K.sum_duplicates(); K.eliminate_zeros()
    lo, hi, cap = [np.asarray(x, dtype=np.float64) for x in (lower, upper, tau_upper)]
    m, n = K.shape
    if (lo.shape != (m,) or hi.shape != (m,) or cap.shape != (n,)
            or not np.isfinite(K.data).all() or (K.data < 0).any()
            or not np.isfinite(lo).all() or (lo < 0).any()
            or np.isnan(hi).any() or (hi < lo).any()
            or not np.isfinite(cap).all() or (cap < 0).any() or not m or not n):
        raise ValueError('Finite nonnegative support/caps and ordered ray intervals required')
    w = np.ones(m) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != (m,) or not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError('Positive explicit ray weights required')
    finite_hi = np.flatnonzero(np.isfinite(hi))
    # Independent slacks expose both unmet occlusion and excess extinction.
    eye = sparse.eye(m, format='csr'); zero = sparse.csr_matrix((m, m))
    A = sparse.vstack([sparse.hstack([-K, -eye, zero]),
                       sparse.hstack([K[finite_hi], zero[finite_hi], -eye[finite_hi]])], format='csr')
    b = np.concatenate([-lo, hi[finite_hi]])
    result = linprog(np.concatenate([np.zeros(n), w, w]), A_ub=A, b_ub=b,
                     bounds=[(0., x) for x in cap]+[(0., None)]*(2*m), method='highs',options={'time_limit':time_limit})
    if not result.success:
        raise RuntimeError('Slack LP failed; no infeasibility claim permitted: '+result.message)
    if reference_tau is not None:
        ref=np.asarray(reference_tau,dtype=np.float64)
        if ref.shape!=(n,) or not np.isfinite(ref).all() or (ref<0).any() or (ref>cap+1e-6).any():
            raise ValueError('Reference tau must obey tested bounds')
        # Lexicographic tie-break: preserve minimal slack, then minimize total
        # absolute optical-depth change instead of accepting an arbitrary vertex.
        ident=sparse.eye(n,format='csr');z=sparse.csr_matrix((n,2*m))
        objective=np.concatenate([np.zeros(n),w,w])
        A2=sparse.vstack([sparse.hstack([A,sparse.csr_matrix((A.shape[0],n))]),
            sparse.csr_matrix(np.concatenate([objective,np.zeros(n)])[None]),
            sparse.hstack([ident,z,-ident]),sparse.hstack([-ident,z,-ident])],format='csr')
        b2=np.concatenate([b,[result.fun+1e-8],ref,-ref])
        result=linprog(np.concatenate([np.zeros(n+2*m),np.ones(n)]),A_ub=A2,b_ub=b2,
            bounds=[(0.,x) for x in cap]+[(0.,None)]*(2*m+n),method='highs-ipm',options={'time_limit':time_limit})
        if not result.success: raise RuntimeError('Minimum-change tie-break failed: '+result.message)
    tau = result.x[:n]; optical = K @ tau
    deficit = np.maximum(lo-optical, 0); excess = np.maximum(optical-hi, 0)
    return dict(tau=tau, optical_depth=optical, lower_slack=deficit, upper_slack=excess,
                weighted_slack=float(w @ (deficit+excess)),
                feasible_within_1e_7=bool(max(deficit.max(), excess.max()) <= 1e-7),
                scope='fixed_support_uncapped_submodel__conditional_on_supplied_intervals__not_geometry_truth')
