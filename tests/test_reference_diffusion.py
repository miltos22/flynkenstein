import math
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flybrain import params as P
from flybrain import reference as R

def grid(nt=None):
    n = 4
    return {'grid_shape': np.array([8, 8, 16]), 'grid_spacing_um': np.array([20.0, 16.0, 15.0]), 'nt': np.array(nt if nt is not None else [4, 0, 5, 6], dtype=np.int8), 'voxel': np.array([0, 5, 300, 1023], dtype=np.int32)}

def test_zero_flux_conserves_mass_without_uptake():
    d = R.Diffusion(grid(), device='cpu', dtype=torch.float64)
    d.Vmax.zero_()
    d.C[0, 2, 3, 4] = 10.0
    d.C[5, 7, 7, 15] = 3.0
    m0 = d.C.sum(dim=(1, 2, 3)).clone()
    for _ in range(500):
        d.step(torch.zeros(4, dtype=torch.float64))
    assert torch.allclose(d.C.sum(dim=(1, 2, 3)), m0, rtol=1e-12, atol=1e-12)
    assert float(d.C[0].max()) < 10.0 * 0.5

def test_uptake_matches_semi_implicit_formula_uniform_field():
    d = R.Diffusion(grid(), device='cpu', dtype=torch.float64)
    d.C[:] = 1.0
    d.step(torch.zeros(4, dtype=torch.float64))
    dts = 0.001 * P.dt_diff_ms
    want_amine = 1.0 / (1.0 + dts * P.V_max_amine_uM_per_s / (P.K_M_uM + 1.0))
    want_pep = 1.0 / (1.0 + dts * P.V_max_pep_uM_per_s / (P.K_M_uM + 1.0))
    assert abs(float(d.C[0].mean()) - want_amine) < 1e-12 and float(d.C[0].std()) < 1e-12
    assert abs(float(d.C[6].mean()) - want_pep) < 1e-12

def test_secretion_lands_in_the_right_channel_and_voxel():
    d = R.Diffusion(grid(), device='cpu', dtype=torch.float64)
    d.D.zero_()
    d.Vmax.zero_()
    s_bar = torch.tensor([0.01, 0.5, 0.02, 0.03], dtype=torch.float64)
    d.step(s_bar)
    dts = 0.001 * P.dt_diff_ms
    flat = d.C.view(12, -1)
    assert abs(float(flat[0, 0]) - dts * 1000 * P.alpha_amine_uM * 0.01) < 1e-15
    assert abs(float(flat[1, 300]) - dts * 1000 * P.alpha_amine_uM * 0.02) < 1e-15
    assert abs(float(flat[3, 1023]) - dts * 1000 * P.alpha_amine_uM * 0.03) < 1e-15
    assert float(flat.sum()) - float(flat[0, 0] + flat[1, 300] + flat[3, 1023]) < 1e-15
if __name__ == '__main__':
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            try:
                fn()
                print('PASS', name)
            except AssertionError as e:
                fails += 1
                print('FAIL', name, '-', e)
    sys.exit(1 if fails else 0)
