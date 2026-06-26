import streamlit as st
import numpy as np
import pandas as pd

from hovorka_model import HovorkaModel

st.set_page_config(
    page_title="Hovorka Virtual Patient Model",
    page_icon="🩸",
    layout="wide",
)

st.title("🩸 Hovorka Virtual Patient Model")
st.caption(
    "Extended EGP-based Hovorka model (proposed) vs. classic Hovorka baseline — "
    "24-hour glucose-insulin simulation."
)

# ---------------------------------------------------------------------------
# Default schedule, taken directly from the MATLAB `meal(t)` / `insulin(t)`
# functions used as the reference implementation.
# ---------------------------------------------------------------------------
DEFAULT_MEAL_TIMES = [420, 720, 960, 1080, 1380]
DEFAULT_MEAL_DURATIONS = [10, 20, 10, 20, 10]
DEFAULT_MEAL_CHO = [12.42, 3.45, 3.45, 7.69, 3.45]

DEFAULT_BOLUS_TIMES = [415, 715, 955, 1075, 1375]
DEFAULT_BOLUS_VALUES = [700, 250, 200, 490, 210]
DEFAULT_BOLUS_DURATION = 10.0

DEFAULT_BASAL = 12.9127
DEFAULT_BW = 70.0


def minutes_to_clock(m):
    """Render a minute offset (0-1440) as an HH:MM clock label."""
    m = int(m) % 1440
    return f"{m // 60:02d}:{m % 60:02d}"


# ---------------------------------------------------------------------------
# Sidebar: patient parameters
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Patient parameters")
    BW = st.number_input("Body weight (kg)", min_value=20.0, max_value=200.0, value=DEFAULT_BW, step=1.0)
    u_basal = st.number_input(
        "Basal insulin rate (mU/min)",
        min_value=0.0, max_value=100.0, value=DEFAULT_BASAL, step=0.1, format="%.4f",
    )

    st.divider()
    st.header("Schedule editor")
    reset = st.button("Reset to MATLAB reference schedule", use_container_width=True)

# ---------------------------------------------------------------------------
# Session state for the editable meal / bolus tables
# ---------------------------------------------------------------------------
def make_meal_df():
    return pd.DataFrame({
        "start_min": DEFAULT_MEAL_TIMES,
        "duration_min": DEFAULT_MEAL_DURATIONS,
        "cho_g": DEFAULT_MEAL_CHO,
    })


def make_bolus_df():
    return pd.DataFrame({
        "start_min": DEFAULT_BOLUS_TIMES,
        "duration_min": [DEFAULT_BOLUS_DURATION] * len(DEFAULT_BOLUS_TIMES),
        "rate_mU_per_min": DEFAULT_BOLUS_VALUES,
    })


if "meal_df" not in st.session_state or reset:
    st.session_state.meal_df = make_meal_df()
if "bolus_df" not in st.session_state or reset:
    st.session_state.bolus_df = make_bolus_df()

# ---------------------------------------------------------------------------
# Editable tables
# ---------------------------------------------------------------------------
col_meal, col_bolus = st.columns(2)

with col_meal:
    st.subheader("🍽️ Meals")
    st.caption("Times are minutes from midnight (e.g. 420 = 07:00).")
    meal_df = st.data_editor(
        st.session_state.meal_df,
        num_rows="dynamic",
        use_container_width=True,
        key="meal_editor",
        column_config={
            "start_min": st.column_config.NumberColumn("Start (min)", min_value=0, max_value=1439, step=1),
            "duration_min": st.column_config.NumberColumn("Duration (min)", min_value=1, max_value=200, step=1),
            "cho_g": st.column_config.NumberColumn("CHO (g)", min_value=0.0, max_value=200.0, step=0.1, format="%.2f"),
        },
    )
    if len(meal_df) > 0:
        preview = meal_df.copy()
        preview.insert(0, "time", preview["start_min"].apply(minutes_to_clock))
        st.dataframe(preview[["time", "duration_min", "cho_g"]], hide_index=True, use_container_width=True)

with col_bolus:
    st.subheader("💉 Insulin boluses")
    st.caption(f"Basal rate ({u_basal:.4f} mU/min) applies outside these windows.")
    bolus_df = st.data_editor(
        st.session_state.bolus_df,
        num_rows="dynamic",
        use_container_width=True,
        key="bolus_editor",
        column_config={
            "start_min": st.column_config.NumberColumn("Start (min)", min_value=0, max_value=1439, step=1),
            "duration_min": st.column_config.NumberColumn("Duration (min)", min_value=1, max_value=200, step=1),
            "rate_mU_per_min": st.column_config.NumberColumn("Rate (mU/min)", min_value=0.0, max_value=2000.0, step=1.0),
        },
    )
    if len(bolus_df) > 0:
        preview = bolus_df.copy()
        preview.insert(0, "time", preview["start_min"].apply(minutes_to_clock))
        st.dataframe(preview[["time", "duration_min", "rate_mU_per_min"]], hide_index=True, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Run simulation
# ---------------------------------------------------------------------------
run = st.button("▶️ Run simulation", type="primary", use_container_width=True)

if run:
    if len(meal_df) == 0 or len(bolus_df) == 0:
        st.error("Add at least one meal and one bolus row before running.")
    else:
        meal_times = meal_df["start_min"].astype(float).tolist()
        meal_durations = meal_df["duration_min"].astype(float).tolist()
        meal_cho = meal_df["cho_g"].astype(float).tolist()

        bolus_times = bolus_df["start_min"].astype(float).tolist()
        bolus_values = bolus_df["rate_mU_per_min"].astype(float).tolist()
        # All reference bolus windows are 10 min wide; use the first row's
        # duration as the uniform bolus_duration argument expected by simulate().
        bolus_duration = float(bolus_df["duration_min"].iloc[0])

        with st.spinner("Integrating ODE system over 24 hours..."):
            model = HovorkaModel(BW=BW, u_basal=u_basal)
            t, G_proposed, G_hovorka, I_proposed = model.simulate(
                meal_times, meal_durations, meal_cho,
                bolus_times, bolus_values, bolus_duration=bolus_duration,
            )
            fig = model.plot(t, G_proposed, G_hovorka, I_proposed)

        st.session_state["last_result"] = {
            "t": t, "G_proposed": G_proposed, "G_hovorka": G_hovorka, "I_proposed": I_proposed,
        }

        st.pyplot(fig, use_container_width=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Final glucose", f"{G_proposed[-1]:.1f} mg/dL")
        m2.metric("Peak glucose", f"{G_proposed.max():.1f} mg/dL", f"at t={minutes_to_clock(t[np.argmax(G_proposed)])}")
        m3.metric("Min glucose", f"{G_proposed.min():.1f} mg/dL", f"at t={minutes_to_clock(t[np.argmin(G_proposed)])}")
else:
    st.info("Adjust the schedule above (or use the MATLAB reference defaults) and click **Run simulation**.")
