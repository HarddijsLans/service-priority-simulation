import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# STREAMLIT PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Pakalpojumu prioritizācijas simulācija",
    layout="wide"
)

st.title("Pakalpojumu pieteikumu prioritizācijas simulācija")

st.markdown("""
Šī simulācija modelē:
- mainīgu ienākošo darbu plūsmu,
- mainīgu komandas kapacitāti,
- prioritizācijas algoritmu,
- rindu veidošanos,
- SLA riskus.
""")

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("Simulācijas parametri")

seed = st.sidebar.number_input(
    "Random Seed",
    min_value=1,
    value=42
)

days = st.sidebar.slider(
    "Dienu skaits",
    min_value=1,
    max_value=120,
    value=30
)

arrivals_input = st.sidebar.text_input(
    "Ienākošie darbi pa dienām",
    value="110,95,95"
)

capacity_input = st.sidebar.text_input(
    "Kapacitāte pa dienām",
    value="90,80,130"
)

pipa_prob = st.sidebar.slider(
    "PIPA varbūtība",
    min_value=0.0,
    max_value=1.0,
    value=0.20,
    step=0.01
)

st.sidebar.markdown("---")

st.sidebar.subheader("ObjectCount sadalījums")

small_share = st.sidebar.slider(
    "ObjectCount 1-5",
    0.0,
    1.0,
    0.90,
    0.01
)

medium_share = st.sidebar.slider(
    "ObjectCount 6-9",
    0.0,
    1.0,
    0.07,
    0.01
)

large_share = 1.0 - small_share - medium_share

st.sidebar.write(f"ObjectCount > 9: {large_share:.2%}")

if large_share < 0:
    st.sidebar.error("Sadalījums pārsniedz 100%")
    st.stop()

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def parse_pattern(text):
    return [int(x.strip()) for x in text.split(",")]


def generate_requests(day, start_id, n):

    ids = np.arange(start_id, start_id + n)

    categories = np.random.choice(
        ["small", "medium", "large"],
        size=n,
        p=[small_share, medium_share, large_share]
    )

    object_counts = []

    for cat in categories:

        if cat == "small":
            value = np.random.randint(1, 6)

        elif cat == "medium":
            value = np.random.randint(6, 10)

        else:
            value = np.random.randint(10, 151)

        object_counts.append(value)

    df = pd.DataFrame({
        "ID": ids,
        "CreatedDay": day,
        "DaysToDeadline": np.random.choice(
            [5, 10, 15],
            size=n
        ),
        "ObjectCount": object_counts,
        "PIPA": np.random.choice(
            [0, 1],
            size=n,
            p=[1 - pipa_prob, pipa_prob]
        ),
        "BusinessDaysWaiting": 0,
        "DaysWaiting": 0
    })

    return df


def calculate_priority(df):

    df = df.copy()

    # ---------------------------------
    # GROUP PRIORITY
    # ---------------------------------
    df["GroupPriority"] = np.select(
        [
            df["DaysToDeadline"] <= 2,
            (df["PIPA"] == 1) &
            (df["BusinessDaysWaiting"] >= 1)
        ],
        [3, 2],
        default=1
    )

    # ---------------------------------
    # P90
    # ---------------------------------
    p90 = np.percentile(df["ObjectCount"], 90)

    p90 = max(p90, 1)

    # ---------------------------------
    # O FACTOR
    # ---------------------------------
    df["O"] = np.sqrt(
        np.minimum(df["ObjectCount"], p90) / p90
    )

    # ---------------------------------
    # T FACTOR
    # ---------------------------------
    bounded_deadline = np.maximum(
        df["DaysToDeadline"],
        0
    )

    df["T"] = np.exp(
        -0.25 * bounded_deadline
    )

    # ---------------------------------
    # P FACTOR
    # ---------------------------------
    df["P"] = np.select(
        [
            (df["PIPA"] == 1) &
            (df["BusinessDaysWaiting"] >= 1),

            (df["PIPA"] == 1) &
            (df["BusinessDaysWaiting"] < 1)
        ],
        [1.0, 0.85],
        default=0
    )

    # ---------------------------------
    # V FACTOR
    # ---------------------------------
    df["V"] = 1 - np.exp(
        -0.15 * df["BusinessDaysWaiting"]
    )

    # ---------------------------------
    # RANDOM FACTOR
    # ---------------------------------
    df["R"] = np.random.random(len(df))

    # ---------------------------------
    # PRIORITY SCORE
    # ---------------------------------
    df["PriorityScore"] = (
        (0.50 * df["T"]) +
        (0.25 * df["P"]) +
        (0.15 * df["O"]) +
        (0.08 * df["V"]) +
        (0.02 * df["R"])
    )

    return df


def sort_queue(df):

    g3 = df[df["GroupPriority"] == 3].sort_values(
        by=["DaysToDeadline", "BusinessDaysWaiting"],
        ascending=[True, False]
    )

    g2 = df[df["GroupPriority"] == 2].sort_values(
        by="PriorityScore",
        ascending=False
    )

    g1 = df[df["GroupPriority"] == 1].sort_values(
        by="PriorityScore",
        ascending=False
    )

    return pd.concat(
        [g3, g2, g1],
        ignore_index=True
    )


# =========================================================
# SIMULATION
# =========================================================
np.random.seed(seed)

arrivals_pattern = parse_pattern(arrivals_input)
capacity_pattern = parse_pattern(capacity_input)

queue = pd.DataFrame()

next_id = 1

daily_stats = []
completed_stats = []

for day in range(1, days + 1):

    daily_arrivals = arrivals_pattern[
        (day - 1) % len(arrivals_pattern)
    ]

    daily_capacity = capacity_pattern[
        (day - 1) % len(capacity_pattern)
    ]

    # ---------------------------------
    # NEW REQUESTS
    # ---------------------------------
    new_requests = generate_requests(
        day=day,
        start_id=next_id,
        n=daily_arrivals
    )

    next_id += daily_arrivals

    queue = pd.concat(
        [queue, new_requests],
        ignore_index=True
    )

    # ---------------------------------
    # PRIORITIES
    # ---------------------------------
    queue = calculate_priority(queue)

    # ---------------------------------
    # SORT
    # ---------------------------------
    queue = sort_queue(queue)

    # ---------------------------------
    # COMPLETE JOBS
    # ---------------------------------
    actual_completed = min(
        daily_capacity,
        len(queue)
    )

    completed_today = queue.head(
        actual_completed
    ).copy()

    queue = queue.iloc[
        actual_completed:
    ].copy().reset_index(drop=True)

    completed_counts = (
        completed_today["GroupPriority"]
        .value_counts()
        .to_dict()
    )

    # ---------------------------------
    # UPDATE WAITING
    # ---------------------------------
    if len(queue) > 0:

        queue["BusinessDaysWaiting"] += 1
        queue["DaysWaiting"] += 1
        queue["DaysToDeadline"] -= 1

    sla_breaches = (
        queue["DaysToDeadline"] < 0
    ).sum()

    # ---------------------------------
    # SAVE STATS
    # ---------------------------------
    daily_stats.append({
        "Day": day,
        "DailyArrivals": daily_arrivals,
        "DailyCapacity": daily_capacity,
        "QueueLength": len(queue),
        "SLABreaches": sla_breaches
    })

    completed_stats.append({
        "Day": day,
        "GroupPriority_1": completed_counts.get(1, 0),
        "GroupPriority_2": completed_counts.get(2, 0),
        "GroupPriority_3": completed_counts.get(3, 0),
        "ActualCompleted": actual_completed
    })

# =========================================================
# DATAFRAMES
# =========================================================
stats_df = pd.DataFrame(daily_stats)

completed_df = pd.DataFrame(completed_stats)

results_df = stats_df.merge(
    completed_df,
    on="Day"
)

# =========================================================
# KPI
# =========================================================
st.subheader("Kopsavilkums")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Kopā ienāca",
    int(stats_df["DailyArrivals"].sum())
)

col2.metric(
    "Kopā izpildīts",
    int(completed_df["ActualCompleted"].sum())
)

col3.metric(
    "Rinda perioda beigās",
    int(stats_df["QueueLength"].iloc[-1])
)

col4.metric(
    "SLA pārkāpumi",
    int(stats_df["SLABreaches"].iloc[-1])
)

# =========================================================
# TABLE
# =========================================================
st.subheader("Dienas statistika")

st.dataframe(
    results_df,
    use_container_width=True
)

# =========================================================
# GRAPH 1
# =========================================================
st.subheader("Rindas garums")

fig1, ax1 = plt.subplots(figsize=(10, 5))

ax1.plot(
    stats_df["Day"],
    stats_df["QueueLength"],
    marker="o"
)

ax1.set_title("Rindas garuma izmaiņas")
ax1.set_xlabel("Diena")
ax1.set_ylabel("Rindas garums")

ax1.grid(True)

st.pyplot(fig1)

# =========================================================
# GRAPH 2
# =========================================================
st.subheader("SLA pārkāpumi")

fig2, ax2 = plt.subplots(figsize=(10, 5))

ax2.plot(
    stats_df["Day"],
    stats_df["SLABreaches"],
    marker="o"
)

ax2.set_title("SLA pārkāpumu dinamika")
ax2.set_xlabel("Diena")
ax2.set_ylabel("SLA pārkāpumi")

ax2.grid(True)

st.pyplot(fig2)

# =========================================================
# GRAPH 3
# =========================================================
st.subheader("Izpildīto darbu sadalījums")

fig3, ax3 = plt.subplots(figsize=(12, 6))

ax3.bar(
    completed_df["Day"],
    completed_df["GroupPriority_1"],
    label="Priority 1"
)

ax3.bar(
    completed_df["Day"],
    completed_df["GroupPriority_2"],
    bottom=completed_df["GroupPriority_1"],
    label="Priority 2"
)

ax3.bar(
    completed_df["Day"],
    completed_df["GroupPriority_3"],
    bottom=(
        completed_df["GroupPriority_1"] +
        completed_df["GroupPriority_2"]
    ),
    label="Priority 3"
)

ax3.set_title(
    "Izpildīto darbu sadalījums"
)

ax3.set_xlabel("Diena")
ax3.set_ylabel("Darbu skaits")

ax3.legend()

ax3.grid(axis="y")

st.pyplot(fig3)

# =========================================================
# GRAPH 4
# =========================================================
st.subheader("Ienākošie darbi pret kapacitāti")

fig4, ax4 = plt.subplots(figsize=(10, 5))

ax4.plot(
    stats_df["Day"],
    stats_df["DailyArrivals"],
    marker="o",
    label="Ienākošie darbi"
)

ax4.plot(
    stats_df["Day"],
    stats_df["DailyCapacity"],
    marker="o",
    label="Kapacitāte"
)

ax4.set_title(
    "Ienākošo darbu skaits pret dienas kapacitāti"
)

ax4.set_xlabel("Diena")
ax4.set_ylabel("Darbu skaits")

ax4.legend()

ax4.grid(True)

st.pyplot(fig4)

# =========================================================
# FINAL QUEUE
# =========================================================
st.subheader("Atlikusī rinda perioda beigās")

st.dataframe(
    queue,
    use_container_width=True
)
