"""Tests for naval_salvo.admissibility."""

from __future__ import annotations

import numpy as np
import pytest

from naval_salvo.admissibility import (
    DEFAULT_CHI,
    Admissibility,
    canonical_matrix,
    degenerate_johns_pilnick_hughes,
)
from naval_salvo.domains import DOMAIN_ORDER, N_DOMAINS, Domain


class TestDegenerateMatrix:
    """The JPH baseline must have only the (S, S) cell active."""

    def test_shape(self):
        M = degenerate_johns_pilnick_hughes()
        assert M.shape == (N_DOMAINS, N_DOMAINS)

    def test_only_ss_cell_is_one(self):
        M = degenerate_johns_pilnick_hughes()
        s = Domain.SURFACE.index
        assert M[s, s] == 1.0
        # Every other cell is zero.
        mask = np.ones_like(M, dtype=bool)
        mask[s, s] = False
        assert np.all(M[mask] == 0.0)

    def test_dtype(self):
        assert degenerate_johns_pilnick_hughes().dtype == np.float64


class TestCanonicalMatrix:
    """Canonical default matrix structure (document 1.4 §2.2 Table 1)."""

    def test_shape_and_range(self):
        M = canonical_matrix()
        assert M.shape == (N_DOMAINS, N_DOMAINS)
        assert np.all(M >= 0.0)
        assert np.all(M <= 1.0)

    def test_diagonal_kinetic_domains_are_one(self):
        M = canonical_matrix()
        for d in (Domain.SURFACE, Domain.UNDERWATER, Domain.AIR, Domain.COASTAL):
            assert M[d.index, d.index] == 1.0

    def test_cyber_self_loop_is_one(self):
        # Intra-X attrition is canonical "1".
        x = Domain.CYBER.index
        assert canonical_matrix()[x, x] == 1.0

    def test_submarines_immune_to_cyber(self):
        # Decision 1.4 §2.3.d -- delta_X^U = 0 (enforced doubly).
        M = canonical_matrix()
        assert M[Domain.CYBER.index, Domain.UNDERWATER.index] == 0.0

    def test_submarines_do_not_project_cyber(self):
        # M[U, X] = 0 in the default.
        M = canonical_matrix()
        assert M[Domain.UNDERWATER.index, Domain.CYBER.index] == 0.0

    def test_torpedo_vs_aircraft_structurally_null(self):
        M = canonical_matrix()
        assert M[Domain.UNDERWATER.index, Domain.AIR.index] == 0.0

    def test_chi_propagates_to_marginal_cells(self):
        chi = 0.3
        M = canonical_matrix(chi=chi)
        # Surface vs Underwater is marginal (helicopter dipping etc.)
        assert M[Domain.SURFACE.index, Domain.UNDERWATER.index] == chi
        # Air vs Underwater is marginal (MPA)
        assert M[Domain.AIR.index, Domain.UNDERWATER.index] == chi

    def test_default_chi(self):
        M = canonical_matrix()
        assert M[Domain.SURFACE.index, Domain.UNDERWATER.index] == DEFAULT_CHI

    @pytest.mark.parametrize("bad_chi", [-0.1, 1.5, np.nan, np.inf])
    def test_chi_out_of_range_raises(self, bad_chi):
        with pytest.raises(ValueError):
            canonical_matrix(chi=bad_chi)


class TestAdmissibilityClass:
    """Behaviour of the immutable wrapper."""

    def test_canonical_constructor(self):
        adm = Admissibility.canonical()
        assert isinstance(adm, Admissibility)
        assert adm.matrix.shape == (N_DOMAINS, N_DOMAINS)

    def test_degenerate_constructor(self):
        adm = Admissibility.degenerate()
        assert adm[Domain.SURFACE, Domain.SURFACE] == 1.0
        assert adm[Domain.AIR, Domain.AIR] == 0.0

    def test_indexing_with_strings(self):
        adm = Admissibility.canonical(chi=0.4)
        assert adm["S", "U"] == 0.4
        assert adm["s", "u"] == 0.4  # case-insensitive via parse_domain

    def test_indexing_returns_python_float(self):
        adm = Admissibility.canonical()
        v = adm[Domain.SURFACE, Domain.SURFACE]
        assert isinstance(v, float)

    def test_indexing_bad_key(self):
        adm = Admissibility.canonical()
        with pytest.raises(KeyError):
            _ = adm[Domain.SURFACE]  # type: ignore[index]
        with pytest.raises(ValueError):
            _ = adm["S", "Z"]  # invalid domain code

    def test_is_admissible(self):
        adm = Admissibility.canonical()
        assert adm.is_admissible(Domain.SURFACE, Domain.SURFACE) is True
        assert adm.is_admissible(Domain.UNDERWATER, Domain.AIR) is False

    def test_admissible_pairs_iteration(self):
        adm = Admissibility.degenerate()
        pairs = list(adm.admissible_pairs())
        assert pairs == [(Domain.SURFACE, Domain.SURFACE)]

    def test_canonical_admissible_pairs_count(self):
        # Count non-zero cells in the canonical matrix.  Used to lock in
        # the admissibility topology of document 1.4 Table 1.
        adm = Admissibility.canonical()
        n_admissible = sum(1 for _ in adm.admissible_pairs())
        # Hand-counted from canonical_matrix definition:
        #   S row: 5 non-zero (S, U-chi, A-chi, C, X-chi)
        #   U row: 3 non-zero (S, U, C-chi)        -- A=0, X=0
        #   A row: 5 non-zero (S, U-chi, A, C, X-chi)
        #   C row: 5 non-zero (S, U-chi, A-chi, C, X-chi)
        #   X row: 4 non-zero (S-chi, A-chi, C-chi, X)  -- U=0
        assert n_admissible == 5 + 3 + 5 + 5 + 4

    def test_immutability_of_underlying_array(self):
        adm = Admissibility.canonical()
        with pytest.raises(ValueError):
            adm.matrix[0, 0] = 0.5  # frozen array refuses writes

    def test_dataframe_dict_round_trip(self):
        adm = Admissibility.canonical(chi=0.7)
        d = adm.as_dataframe_dict()
        assert set(d.keys()) == {dom.value for dom in DOMAIN_ORDER}
        # Spot-check one entry.
        assert d["S"]["S"] == 1.0
        assert d["S"]["U"] == 0.7

    def test_from_array_validates_shape(self):
        with pytest.raises(ValueError):
            Admissibility.from_array(np.zeros((4, 5)))

    def test_from_array_validates_range(self):
        bad = np.zeros((5, 5))
        bad[0, 0] = 1.5
        with pytest.raises(ValueError):
            Admissibility.from_array(bad)
        bad[0, 0] = -0.1
        with pytest.raises(ValueError):
            Admissibility.from_array(bad)

    def test_from_array_rejects_non_finite(self):
        bad = np.zeros((5, 5))
        bad[2, 2] = np.nan
        with pytest.raises(ValueError):
            Admissibility.from_array(bad)
