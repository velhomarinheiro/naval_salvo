"""
"About" page of the application.

Brings together in one place: conceptual description of the model,
authorship, institutional context, declaration of AI tool use as
support, and bibliographic references.
"""

from __future__ import annotations

import streamlit as st


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("About")
st.caption(
    "Conceptual description, authorship, AI use declaration and "
    "bibliographic references"
)

st.markdown("---")


# ---------------------------------------------------------------------------
# Conceptual description
# ---------------------------------------------------------------------------

st.header("Conceptual description of the model")

st.markdown(
    """
    The *Multi-Domain Salvo Equation* implemented in this application is
    an extension of the classical naval salvo equations (Hughes 1995;
    Johns, Pilnick and Hughes 2001) to an operational environment in which
    interaction occurs simultaneously across five domains:

    - **surface** (S) — surface combatants, USVs, production platforms,
      and embarked helicopters, the latter absorbed as a capability of
      the parent platform;
    - **subsurface** (U) — submarines, UUVs, and submarine mines;
    - **air** (A) — manned and unmanned aircraft;
    - **coastal** (C) — coastal anti-ship missile batteries,
      coastal artillery, and coastal mines;
    - **cyber-electromagnetic** (X) — cyber and electronic warfare effects,
      decomposed into four functional sub-types
      (C2, sensors, weapons, and logistics).

    The interaction between domains is mediated by a **5×5 admissibility
    matrix**, which encodes at three levels which attacker-defender pairs
    are doctrinally possible: primary (1), calibrable marginal (χ ∈ [0, 1]),
    and structurally null (0). For example, a torpedo fired by a submarine
    cannot reach an airborne aircraft (null interaction), while a helicopter
    operating dipping sonar against a submarine is a marginal interaction
    whose effectiveness depends on the calibration of χ.

    The cyber domain receives a distinct treatment from the others. Instead
    of acting predominantly through direct attrition — which would be
    unrealistic — it acts as a **multiplicative modulator Φ** of the
    offensive and defensive coefficients of kinetic pairs. When a force
    has a cyber advantage over its opponent, its sensors, weapon systems,
    command networks, and logistics operate at higher relative efficiency,
    and the opponent's force coefficients are correspondingly degraded.
    The Φ function is a sigmoid parameterised by the cyber strength ratio
    between the two sides on each channel.

    Submarines are treated as **immune to the cyber domain** by an
    explicit doctrinal choice of the model (isolation by depth and
    emission silence). This choice is debatable, and relaxing it is a
    natural point for future evolution of the application.

    The combat regime is **simultaneous pulsed**: at each salvo, both
    sides calculate their losses based on the pre-salvo state and apply
    the result jointly. This is the convention adopted by Hughes (1995),
    Johns-Pilnick-Hughes (2001), and Armstrong (2005). The sequential
    exchange variant (Armstrong 2014) is not implemented in this version.
    """
)


# ---------------------------------------------------------------------------
# Authorship
# ---------------------------------------------------------------------------

st.markdown("---")
st.header("Authorship")

col_info, col_contact = st.columns([2, 1])

with col_info:
    st.markdown(
        """
        **Alberto Ferreira Filho**

        Senior Officer of the Brazilian Navy. Master's in Defence and
        Strategic Studies from the *U.S. Naval War College*. Doctoral
        candidate in Public Administration at the Brazilian School of
        Public and Business Administration (EBAPE) of Fundação Getulio
        Vargas.

        This application is an applied outgrowth of his research interests
        at the intersections of operations research, strategic studies, and
        quantitative naval combat modelling. The tool is offered as a
        contribution to the academic and doctrinal debate on the use of
        salvo models in multi-domain naval analyses.
        """
    )

with col_contact:
    st.markdown(
        """
        **Contact**

        ✉️ ferreirafilhoalberto@gmail.com

        🔗 [LinkedIn](https://www.linkedin.com/in/alberto-ferreira-filho-0b21a71a3)
        """
    )


# ---------------------------------------------------------------------------
# AI use declaration
# ---------------------------------------------------------------------------

st.markdown("---")
st.header("Use of artificial intelligence tools")

st.markdown(
    """
    In the development of this application, generative artificial
    intelligence tools were employed — in particular **Claude**
    (Anthropic) and **ChatGPT** (OpenAI) — as **support** for the
    author's work in the following activities:

    - review and organisation of the **mathematical formulations** of the
      model, including the systematisation of notation, consistency
      checking across the various equations of the family, and
      verification of algebraic passages;
    - **drafting and reviewing the Python code** that makes up the
      `naval_salvo` package and the Streamlit pages of this application,
      with emphasis on code structure, documentation, numerical
      validation, and correction of inconsistencies.

    The use of these tools followed a logic of **support, not delegation**:
    the modelling choices, the selection of references, the interpretation
    of results, and the responsibility for the final content rest entirely
    with the author. The tools were used as productivity assistants, in
    the manner of a technical reviewer or a peer available for discussion,
    without replacing human judgement on the substantive decisions of the
    project.

    This declaration is made in accordance with contemporary best
    practices of transparency regarding AI use in academic and technical
    production.
    """
)


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------

st.markdown("---")
st.header("Bibliographic references")

st.markdown(
    """
    **Classical salvo equation family**

    - Hughes, W. P. (1995). A Salvo Model of Warships in Missile
      Combat Used to Evaluate Their Staying Power. *Naval Research
      Logistics*, 42(2), 267–289.
    - Johns, M. E.; Pilnick, S. E.; Hughes, W. P. (2001).
      *Heterogeneous Salvo Model for the Navy After Next* (NPS-IJWA-
      01-010). Naval Postgraduate School.
    - Armstrong, M. J. (2005). A Stochastic Salvo Model for Naval
      Surface Combat. *Operations Research*, 53(5), 830–841.
    - Armstrong, M. J. (2013). The Salvo Combat Model with Area Fire.
      *Naval Research Logistics*, 60(8), 652–660.
    - Armstrong, M. J. (2014). The Salvo Combat Model with a
      Sequential Exchange of Fire. *Journal of the Operational
      Research Society*, 65(10), 1593–1601.

    **Fire reallocation and extended Lanchester models**

    - MacKay, N. J. (2009). Lanchester combat models. *Journal of the
      Operational Research Society*, 60, 1421–1427.
    - Hausken, K.; Moxnes, J. F. (2026). A multi-equipment
      Lanchester model with proportional reallocation. *Annals of
      Operations Research*, 357, 1003–1019.

    **Methodological discussion and applications**

    - Lucas, T. W.; McGunnigle, J. E. (2003). When is model
      complexity too much? Illustrating the benefits of simple models
      with Hughes' salvo equations. *Naval Research Logistics*, 50(3),
      197–217.

    **Auxiliary applications and calibrations used in the project**

    - Beall, T. R. (1990). *Naval Gunnery and Naval Salvo Combat
      Data*. Original data used in the historical reproductions
      (Coronel, Coral Sea, Savo Island) implemented in the validation
      tab.
    - Christiansen, K. P. (2008). *Fitting Salvo Equations to Naval
      Combat Data*. Naval Postgraduate School.
    - Casola, K. (2017). *Optimisation of Naval Gun Firing Patterns
      for Engagement of Manoeuvring Surface Targets*. Naval
      Postgraduate School.
    - Vasankari, L. (2024). *Littoral Naval Warfare with Multi-Agent
      Reinforcement Learning*. Master's dissertation.

    **Author's own work**

    - Araujo, T. M. P. de C.; Ferreira Filho, A.; Santos, M. dos;
      Gomes, C. F. S.; Fróes, B. E. (2025). Presentation of a web
      application to assist in calculating salvo equations.
      *Proceedings of the LVII Brazilian Symposium on Operations
      Research*, Gramado, RS.
    """
)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.caption(
    "Experimental version — suggestions, criticism, and collaboration "
    "proposals are welcome via the contact channels listed above."
)
