"""Tests for naval_salvo.domains."""

from __future__ import annotations

import pytest

from naval_salvo.domains import (
    CYBER_SUBTYPES,
    DOMAIN_ORDER,
    KINETIC_DOMAINS,
    N_DOMAINS,
    CyberSubtype,
    Domain,
    parse_domain,
)


class TestDomainEnum:
    """Sanity checks on the canonical domain enum."""

    def test_all_five_domains_present(self):
        assert len(DOMAIN_ORDER) == 5
        assert N_DOMAINS == 5

    def test_canonical_ordering_is_S_U_A_C_X(self):
        codes = [d.value for d in DOMAIN_ORDER]
        assert codes == ["S", "U", "A", "C", "X"]

    def test_indices_are_consecutive(self):
        for i, d in enumerate(DOMAIN_ORDER):
            assert d.index == i

    def test_kinetic_property(self):
        for d in (Domain.SURFACE, Domain.UNDERWATER, Domain.AIR, Domain.COASTAL):
            assert d.is_kinetic is True
        assert Domain.CYBER.is_kinetic is False

    def test_kinetic_domains_constant(self):
        assert KINETIC_DOMAINS == (
            Domain.SURFACE,
            Domain.UNDERWATER,
            Domain.AIR,
            Domain.COASTAL,
        )

    def test_str_subclass_round_trip(self):
        # Domain inherits from str, so equality with the code works.
        assert Domain.SURFACE == "S"
        # And the value is the canonical code.
        assert Domain.AIR.value == "A"

    def test_uniqueness(self):
        assert len(set(DOMAIN_ORDER)) == 5


class TestParseDomain:
    """parse_domain accepts strings (short or long) and Domain instances."""

    def test_short_code(self):
        assert parse_domain("S") is Domain.SURFACE
        assert parse_domain("U") is Domain.UNDERWATER
        assert parse_domain("A") is Domain.AIR
        assert parse_domain("C") is Domain.COASTAL
        assert parse_domain("X") is Domain.CYBER

    def test_long_name(self):
        assert parse_domain("SURFACE") is Domain.SURFACE
        assert parse_domain("CYBER") is Domain.CYBER

    def test_case_insensitive(self):
        assert parse_domain("surface") is Domain.SURFACE
        assert parse_domain("sUrFaCe") is Domain.SURFACE
        assert parse_domain("x") is Domain.CYBER

    def test_passthrough_domain(self):
        assert parse_domain(Domain.AIR) is Domain.AIR

    def test_strips_whitespace(self):
        assert parse_domain("  A  ") is Domain.AIR

    def test_raises_on_unknown(self):
        with pytest.raises(ValueError):
            parse_domain("Z")
        with pytest.raises(ValueError):
            parse_domain("")
        with pytest.raises(ValueError):
            parse_domain("submarine")  # not a member name; only UNDERWATER

    def test_raises_on_non_string(self):
        with pytest.raises(ValueError):
            parse_domain(3)  # type: ignore[arg-type]


class TestCyberSubtypes:
    """The four cyber sub-types must match the canonical decomposition."""

    def test_four_subtypes(self):
        assert len(CYBER_SUBTYPES) == 4

    def test_canonical_codes(self):
        codes = {s.value for s in CYBER_SUBTYPES}
        assert codes == {"C2", "SEN", "WPN", "LOG"}

    def test_members_accessible(self):
        # Sanity: name -> member lookup works as for any Enum.
        assert CyberSubtype["C2"] is CyberSubtype.C2
        assert CyberSubtype["SENSOR"] is CyberSubtype.SENSOR
