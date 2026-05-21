"""
Home page of the "Multi-Domain Salvo Equation" application.

This is the landing page of the Streamlit app. It describes the project,
its purposes, and what the user will find on each of the other pages
accessible from the sidebar menu.
"""

from __future__ import annotations

import streamlit as st


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("⚓ Multi-Domain Salvo Equation - v. 1.0")
st.caption(
    "Interactive exploration tool for multi-domain naval analyses "
    "— experimental version - published on 17/05/2026"
)

st.markdown("---")


# ---------------------------------------------------------------------------
# Project overview
# ---------------------------------------------------------------------------

st.header("Project overview")

st.markdown(
    """
    This application is an **interactive tool** for exploring the
    *Multi-Domain Salvo Equation* applied to naval scenarios. It allows
    the user to vary force composition parameters, offensive and defensive
    coefficients, targeting selection policies, and cyber effects, and
    observe the resulting trajectories over successive salvos.

    The current version is **experimental**. The implemented models are
    still in a calibration phase, and the numerical values pre-filled on
    the following pages follow reasonable orders of magnitude, but do not
    constitute validated parameters for real situations. The tool should be
    understood as **an aid for reflection**, not as a predictor of combat
    outcomes.

    The main purpose of the application is to serve as an aid in:

    - **naval strategy and tactics** analyses, especially exploring the
      relative effect of different parameters on asset survival and
      engagement duration;
    - **war games** and doctrinal exercises, providing a quantitative
      basis for discussing force composition trade-offs;
    - **force planning** studies, examining, for example, the marginal
      impact of adding a platform, expanding the escort, or incorporating
      cyber capabilities.

    The application draws on a well-established tradition of modelling in
    military operations research. The main references that underpin the
    implemented models are:

    - **Hughes (1995)** — original formulation of the homogeneous salvo
      equation.
    - **Johns, Pilnick and Hughes (2001)** — heterogeneous generalisation
      of the Hughes model, with multiple unit types per force.
    - **Armstrong (2005, 2013, 2014)** — model extensions for lethality,
      area fire, and sequential salvo exchanges.
    - **MacKay (2009)** and **Hausken & Moxnes (2026)** — models with
      proportional fire reallocation and variable death rates.
    - **Lucas and McGunnigle (2003)** — discussion of the utility and
      limits of simple naval combat models.

    The application also incorporates a new chapter: the **Multi-Domain
    Salvo Equation**, which extends the classical formulations to five
    domains — surface, subsurface, air, coastal, and cyber — with a
    cross-domain admissibility matrix and a multiplicative cyber
    modulator. This extension is described in greater detail on the
    **About** page.
    """
)

st.info(
    "💡 **On interpreting results.** Each simulation produces a "
    "deterministically exact trajectory for the given parameters, "
    "but the sensitivity of the result to those parameters is high. "
    "The analytical value of the tool lies in comparing configurations "
    "with each other, not in setting absolute predictions about a "
    "specific confrontation."
)


# ---------------------------------------------------------------------------
# Available pages
# ---------------------------------------------------------------------------

st.markdown("---")
st.header("Available pages")

st.markdown(
    """
    The sidebar menu provides access to the following pages. Each is
    independent and can be used on its own.
    """
)


st.subheader("Hughes 1995 — classic homogeneous model")

st.markdown(
    """
    On this page the user works with the **classic homogeneous salvo
    model** proposed by Wayne P. Hughes Jr. in 1995. Each force is
    represented by a single unit class, characterised by its initial
    quantity, offensive power, defensive power, and staying power.

    The equation calculates, for each salvo, the losses of each side as
    a function of the differential between incoming fire and intercepted
    fire. It is the leanest model in the family and serves as a
    didactic introduction to salvo equation reasoning and as a limiting
    validation case for the more complex models.
    """
)


st.subheader("Multi-Domain Model Scenario")

st.markdown(
    """
    This page implements a **configurable multi-domain scenario** inspired
    by the defence of an oil and gas production platform area, resembling
    the Campos Basin. The scenario represents an operational problem for a
    coastal-waters navy: protecting high economic and strategic value
    assets against a hostile force operating across multiple domains.

    The user can configure surface units, submarines, coastal batteries,
    FPSOs, strike aviation, cyber capabilities, engagement matrices, and
    targeting selection policies.

    Internally, the model applies a **cross-domain admissibility matrix**
    with values at three levels: primary, calibrable marginal, and null.
    Submarines are treated as immune to the cyber domain, by an explicit
    doctrinal choice of the model.
    """
)


st.subheader("Cyber — cyber modulation analysis")

st.markdown(
    """
    This page is dedicated to the **analysis of the effect of cyber
    modulation Φ** on the model's parameters. Instead of treating the
    cyber domain solely as a source of direct attrition, the model
    represents it as a *multiplicative modulator* of the offensive and
    defensive kinetic coefficients.

    The central intuition is that a cyber advantage can degrade the
    opponent's ability to aim, launch, and defend, without necessarily
    destroying physical units.
    """
)


st.subheader("Validation — numerical verification and sensitivity")

st.markdown(
    """
    The validation page brings together the **numerical validation report**
    of the implemented engine. It shows the machine-precision reproduction
    of two canonical cases from the literature: the Hughes (1995)
    homogeneous engagement and the Battle of Coronel, from the worked
    example of Johns, Pilnick and Hughes (2001).

    Beyond the validations, the page offers **sensitivity analyses** on
    the main parameters of the multi-domain model.
    """
)


st.subheader("About — conceptual information, authorship and references")

st.markdown(
    """
    The **About** page brings together the conceptual description of the
    Multi-Domain Salvo Equation, the modelling choices, authorship
    information and the project's institutional context, as well as the
    complete bibliographic references.
    """
)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.caption(
    "Experimental version. The models and parameters implemented here "
    "are in a calibration phase and do not replace formal operational "
    "analyses. Use the pages from the sidebar menu."
)
