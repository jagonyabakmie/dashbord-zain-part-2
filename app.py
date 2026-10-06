"""Mathematics Achievement Analytics: six-page multilevel dashboard.

Run with:  streamlit run app.py
Statistics are read from MASTER_STREAMLIT_MULTILEVEL.xlsx (and LOSO_ROBUSTNESS_MULTILEVEL.xlsx for
leave-one-school-out robustness); nothing is re-fitted here.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Mathematics Achievement Analytics",
    page_icon="📗",
    layout="wide",
    initial_sidebar_state="auto",
)

from utils import data as D  # noqa: E402
from utils import styles as S  # noqa: E402
from views import (diagnostics, exploration, model_lab, overview,  # noqa: E402
                   research_questions, school_explorer)

PAGES = {
    "overview": (":material/home: Overview Profile", overview.render),
    "exploration": (":material/travel_explore: Exploratory Analysis", exploration.render),
    "rq": (":material/lightbulb: Research Questions", research_questions.render),
    "lab": (":material/science: Multilevel Model Selection", model_lab.render),
    "school": (":material/school: School Performance Explorer", school_explorer.render),
    "diag": (":material/verified_user: Diagnostics & Robustness", diagnostics.render),
}
# Card-grid sizing per page: KPI rows above the cards and whether the page uses internal tabs.
LAYOUT = {
    "overview": dict(kpi_rows=2, tabs=True),
    "exploration": dict(kpi_rows=0, tabs=True),
    "rq": dict(kpi_rows=1, tabs=False),
    "lab": dict(kpi_rows=1, tabs=True),
    "school": dict(kpi_rows=1, tabs=True),
    "diag": dict(kpi_rows=1, tabs=True),
}
FILTER_PAGES = {"overview", "exploration"}   # school / gender / student are real filters here
HIGHLIGHT_PAGES = {"lab", "diag"}             # school / student selections only highlight

ALL_SCHOOLS = "All Schools"
ALL = "All"
GENDERS = ["All", "Male", "Female"]


def _init_state() -> None:
    st.session_state.setdefault("page", "overview")
    st.session_state.setdefault("f_schools", [])
    st.session_state.setdefault("f_gender", ALL)
    st.session_state.setdefault("f_students", [])
    st.session_state.setdefault("hl_schools", [])
    st.session_state.setdefault("hl_students", [])


def _reset_filters() -> None:
    st.session_state["f_schools"] = []
    st.session_state["f_gender"] = ALL
    st.session_state["f_students"] = []


def _reset_highlights() -> None:
    st.session_state["hl_schools"] = []
    st.session_state["hl_students"] = []


def _persist(widget, label: str, options: list, store_key: str, multi: bool = False, **kw):
    """Widget whose value survives pages where it is not rendered (stored under store_key)."""
    wkey = "w_" + store_key
    val = st.session_state[store_key]
    if multi:
        val = [v for v in val if v in options]
    elif val not in options:
        val = options[0]
    st.session_state[store_key] = val
    st.session_state[wkey] = val

    def _sync():
        st.session_state[store_key] = st.session_state[wkey]

    return widget(label, options, key=wkey, on_change=_sync, **kw)


def _gender_control() -> None:
    wkey = "w_f_gender"
    st.session_state[wkey] = st.session_state["f_gender"]

    def _sync():
        st.session_state["f_gender"] = st.session_state[wkey] or ALL

    st.segmented_control("Gender", GENDERS, key=wkey, on_change=_sync, selection_mode="single",
                         help="Single choice: all students, male students only or female students only.")


def _branding() -> None:
    wm = S.img_uri(S.wordmark_b64())
    logos = S.img_uri(S.logo_strip_b64())
    wm_html = f'<img class="wordmark" src="{wm}" alt="RASIO 10.0"/>' if wm else '<div class="brand-title">RASIO 10.0</div>'
    logo_html = f'<img class="logos" src="{logos}" alt="Partner logos"/>' if logos else ""
    st.markdown(f'<div class="rasio-brand">{wm_html}{logo_html}<div class="rasio-team">Team Gabutt · 014SO2026</div></div>',
                unsafe_allow_html=True)


def sidebar(students) -> str:
    with st.sidebar:
        _branding()
        st.markdown(
            f"""<div class="brand"><div class="brand-mark">{S.icon('menu_book')}</div>
<div><div class="brand-title">Math Achievement</div><div class="brand-sub">Multilevel Analytics</div></div></div>""",
            unsafe_allow_html=True,
        )
        st.markdown("<hr/>", unsafe_allow_html=True)
        keys = list(PAGES)
        page = st.radio(
            "Navigation", keys, key="page", label_visibility="collapsed",
            format_func=lambda k: PAGES[k][0],
        )
        st.markdown("<hr/>", unsafe_allow_html=True)

        school_opts = [D.school_label(s) for s in sorted(students["school_id"].unique())]
        if page in FILTER_PAGES:
            st.markdown(f'<div class="side-label">{S.icon("tune")} Filters</div>', unsafe_allow_html=True)
            school_help = ("Filters the students in every chart. Leave empty for all schools."
                           if page == "overview" else
                           "Filters student-level charts. On the School Relationships tab the schools are only highlighted.")
            _persist(st.multiselect, "School", school_opts, "f_schools", multi=True,
                     placeholder="All schools", help=school_help)
            _gender_control()
            pool = students
            if st.session_state["f_schools"]:
                pool = pool[pool["school_label"].isin(st.session_state["f_schools"])]
            if st.session_state["f_gender"] != ALL:
                pool = pool[pool["gender"] == st.session_state["f_gender"]]
            stud_help = ("Optional. Limits the view to the chosen students. Options follow the school and gender filters."
                         if page == "overview" else
                         "Optional. Chosen students are highlighted in the scatter plot. Options follow the school and gender filters.")
            _persist(st.multiselect, "Student ID", sorted(pool["student_id"].tolist()), "f_students", multi=True,
                     placeholder="All matching students", help=stud_help)
            st.button("Reset filters", on_click=_reset_filters, width="stretch", icon=":material/restart_alt:")
        elif page in HIGHLIGHT_PAGES:
            st.markdown(f'<div class="side-label">{S.icon("my_location")} Highlight</div>', unsafe_allow_html=True)
            _persist(st.multiselect, "Highlight schools", school_opts, "hl_schools", multi=True, placeholder="None",
                     help="Highlights schools in the plots only. Model results always use the full sample.")
            pool = students
            if st.session_state["hl_schools"]:
                pool = pool[pool["school_label"].isin(st.session_state["hl_schools"])]
            _persist(st.multiselect, "Highlight students", sorted(pool["student_id"].tolist()), "hl_students", multi=True,
                     placeholder="None", help="Highlights students in residual and fit plots only.")
            st.button("Clear highlights", on_click=_reset_highlights, width="stretch", icon=":material/restart_alt:")
            st.markdown(
                f'<div class="side-note">Results use the full sample (900 students, 30 schools). '
                f'Selections only highlight; nothing is re-fitted.</div>',
                unsafe_allow_html=True,
            )
        elif page == "rq":
            st.markdown(
                '<div class="side-note">Research questions always use the locked final model on the full '
                'sample (900 students, 30 schools). There are no filters on this page.</div>',
                unsafe_allow_html=True,
            )
        elif page == "school":
            st.markdown(
                '<div class="side-note">Choose up to three schools to compare at the top of the page. '
                'School values come from the locked final model.</div>',
                unsafe_allow_html=True,
            )
    return page


def main() -> None:
    _init_state()
    page = st.session_state.get("page", "overview")
    S.inject_css(**LAYOUT.get(page, {}))
    students = D.students()
    page = sidebar(students)
    ctx = {
        "schools": [int(x.split()[-1]) for x in st.session_state["f_schools"]],
        "gender": st.session_state["f_gender"],
        "students": list(st.session_state["f_students"]),
        "hl_schools": [int(x.split()[-1]) for x in st.session_state["hl_schools"]],
        "hl_students": list(st.session_state["hl_students"]),
        "ALL_SCHOOLS": ALL_SCHOOLS,
        "ALL": ALL,
    }
    PAGES[page][1](ctx)


main()
