from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# -----------------------------------------------------------------------------
# Page configuration and constants
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Digital Retail Loan Product Analytics",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "loan_applications.csv"

REQUIRED_COLUMNS = [
    "Application_ID",
    "Employment_Type",
    "Monthly_Income",
    "Credit_Score",
    "Monthly_Obligation",
    "Loan_Amount",
    "Channel",
    "Approval_Status",
    "Approval_TAT_Hours",
    "Disbursement_Status",
    "Dropoff_Stage",
    "DPD_30",
]

CREDIT_BAND_ORDER = ["<650", "650-699", "700-749", "750+"]
CHANNEL_ORDER = ["Branch", "Mobile App", "Partner", "Website"]

# Illustrative project rules only; these are not bank underwriting policies.
ELIGIBLE_MIN_CREDIT_SCORE = 750
ELIGIBLE_MAX_DTI = 40.0
NOT_ELIGIBLE_MAX_CREDIT_SCORE = 650
NOT_ELIGIBLE_MIN_DTI = 40.0


# -----------------------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------------------
@st.cache_data

def load_data(csv_path: str) -> pd.DataFrame:
    """Load the CSV and prepare fields used by the dashboard."""
    data = pd.read_csv(csv_path)

    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(
            "The dataset is missing required columns: " + ", ".join(missing)
        )

    numeric_columns = [
        "Monthly_Income",
        "Credit_Score",
        "Monthly_Obligation",
        "Loan_Amount",
        "Approval_TAT_Hours",
        "DPD_30",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data["Approval_Status"] = data["Approval_Status"].astype("string").str.strip()
    data["Disbursement_Status"] = (
        data["Disbursement_Status"].astype("string").str.strip()
    )
    data["Channel"] = data["Channel"].astype("string").str.strip()
    data["Employment_Type"] = data["Employment_Type"].astype("string").str.strip()

    data["Approved_Flag"] = data["Approval_Status"].eq("Approved")
    data["Disbursed_Flag"] = data["Disbursement_Status"].eq("Disbursed")

    data["Credit_Score_Band"] = pd.cut(
        data["Credit_Score"],
        bins=[float("-inf"), 649, 699, 749, float("inf")],
        labels=CREDIT_BAND_ORDER,
        right=True,
    )

    # This is the customer's existing-obligations-to-income ratio.
    # It does not include an EMI for the newly requested loan.
    data["DTI"] = (
        data["Monthly_Obligation"] / data["Monthly_Income"].where(data["Monthly_Income"] > 0)
    ) * 100

    data["DPD_30"] = data["DPD_30"].fillna(0).astype(int)
    return data


def inr(value) -> str:
    """Format a numeric value in Indian rupees with Indian digit grouping."""
    if pd.isna(value):
        return "N/A"
    number = int(round(float(value)))
    sign = "-" if number < 0 else ""
    digits = str(abs(number))
    if len(digits) > 3:
        last_three = digits[-3:]
        remaining = digits[:-3]
        groups = []
        while remaining:
            groups.insert(0, remaining[-2:])
            remaining = remaining[:-2]
        digits = ",".join(groups + [last_three])
    return f"{sign}₹{digits}"


def safe_rate(numerator: int, denominator: int) -> float:
    return (numerator / denominator * 100) if denominator else 0.0


def apply_filters(
    data: pd.DataFrame,
    channels: list[str],
    credit_bands: list[str],
    employment_types: list[str],
) -> pd.DataFrame:
    filtered = data.copy()
    if channels:
        filtered = filtered[filtered["Channel"].isin(channels)]
    if credit_bands:
        filtered = filtered[filtered["Credit_Score_Band"].astype("string").isin(credit_bands)]
    if employment_types:
        filtered = filtered[filtered["Employment_Type"].isin(employment_types)]
    return filtered


def channel_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Create channel metrics. Approval TAT is based on approved applications only."""
    rows = []
    channel_values = [
        channel for channel in CHANNEL_ORDER if channel in data["Channel"].dropna().unique()
    ]
    channel_values += [
        channel
        for channel in sorted(data["Channel"].dropna().unique())
        if channel not in channel_values
    ]

    for channel in channel_values:
        group = data[data["Channel"].eq(channel)]
        approved = group[group["Approved_Flag"]]
        disbursed = group[group["Disbursed_Flag"]]
        apps_count = group["Application_ID"].nunique()
        approved_count = approved["Application_ID"].nunique()
        disbursed_count = disbursed["Application_ID"].nunique()
        approved_tat = approved["Approval_TAT_Hours"].mean()

        rows.append(
            {
                "Channel": channel,
                "Applications": apps_count,
                "Approved": approved_count,
                "Disbursed": disbursed_count,
                "Avg Approval TAT (Approved Apps, hrs)": round(approved_tat, 1)
                if pd.notna(approved_tat)
                else None,
                "Approval Rate": safe_rate(approved_count, apps_count),
                "Disbursement Conversion": safe_rate(disbursed_count, apps_count),
            }
        )

    return pd.DataFrame(rows)


def make_tat_by_channel(data: pd.DataFrame) -> pd.DataFrame:
    """Single shared TAT calculation used in both TAT charts."""
    approved = data[data["Approved_Flag"]].copy()
    result = (
        approved.groupby("Channel", as_index=False, dropna=True)
        .agg(Avg_Approval_TAT_Hours=("Approval_TAT_Hours", "mean"))
    )
    result["Avg_Approval_TAT_Hours"] = result["Avg_Approval_TAT_Hours"].round(1)

    # Keep channel order consistent across the app.
    order_map = {name: idx for idx, name in enumerate(CHANNEL_ORDER)}
    result["_order"] = result["Channel"].map(order_map).fillna(999)
    result = result.sort_values(["_order", "Channel"]).drop(columns="_order")
    return result


def make_credit_risk_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Show conversion across all applications and DPD rate among disbursed loans."""
    rows = []
    for band in CREDIT_BAND_ORDER:
        group = data[data["Credit_Score_Band"].astype("string").eq(band)]
        disbursed = group[group["Disbursed_Flag"]]
        applications = group["Application_ID"].nunique()
        disbursed_count = disbursed["Application_ID"].nunique()
        dpd_rate = disbursed["DPD_30"].mean() * 100 if not disbursed.empty else float("nan")
        rows.append(
            {
                "Credit Score Band": band,
                "Applications": applications,
                "Disbursed": disbursed_count,
                "Conversion Rate (%)": safe_rate(disbursed_count, applications),
                "30+ DPD Rate Among Disbursed (%)": dpd_rate,
            }
        )
    return pd.DataFrame(rows)


def make_dropoff_summary(data: pd.DataFrame) -> pd.DataFrame:
    dropoffs = data[data["Dropoff_Stage"].notna()].copy()
    dropoffs["Dropoff_Stage"] = dropoffs["Dropoff_Stage"].astype(str).str.strip()
    dropoffs = dropoffs[dropoffs["Dropoff_Stage"].ne("")]
    total_applications = data["Application_ID"].nunique()
    if dropoffs.empty:
        return pd.DataFrame(columns=["Drop-off Stage", "Applications", "Share of All Applications (%)"])
    summary = (
        dropoffs.groupby("Dropoff_Stage", as_index=False)
        .agg(Applications=("Application_ID", "nunique"))
        .rename(columns={"Dropoff_Stage": "Drop-off Stage"})
    )
    summary["Share of All Applications (%)"] = summary["Applications"].apply(
        lambda count: safe_rate(int(count), int(total_applications))
    )
    return summary.sort_values("Applications", ascending=False)


def get_preliminary_eligibility(credit_score: int, dti: float) -> tuple[str, str]:
    """Apply conservative, illustrative student-project rules."""
    if credit_score >= ELIGIBLE_MIN_CREDIT_SCORE and dti <= ELIGIBLE_MAX_DTI:
        return (
            "Eligible",
            "The profile meets the prototype's preliminary criteria: credit score is at least 750 and existing-obligation DTI is no more than 40%.",
        )

    # Require both risk signals before using the narrow Not Eligible outcome.
    # This avoids excluding a potential customer based on only one indicator.
    if credit_score < NOT_ELIGIBLE_MAX_CREDIT_SCORE and dti > NOT_ELIGIBLE_MIN_DTI:
        return (
            "Not Eligible",
            "Both preliminary risk flags are present: credit score is below 650 and existing-obligation DTI is above 40%. This is a prototype screening outcome, not a final bank credit decision.",
        )

    return (
        "Review Required",
        "The profile does not meet the automatic Eligible criteria, but it does not meet both conditions for the prototype's Not Eligible outcome. Further assessment is recommended.",
    )


# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
try:
    df = load_data(str(DATA_FILE))
except Exception as exc:
    st.error(f"Unable to load loan_applications.csv: {exc}")
    st.info("Ensure app.py and loan_applications.csv are in the same GitHub repository folder.")
    st.stop()


# -----------------------------------------------------------------------------
# Sidebar: navigation and filters
# -----------------------------------------------------------------------------
st.sidebar.title("Loan Product App")
st.sidebar.write("Digital Retail Loan Product Analytics")
st.sidebar.caption(
    "Explore portfolio performance, identify loan-funnel patterns and test the FastTrack concept."
)

page = st.sidebar.radio(
    "Navigation",
    ["Portfolio Overview", "Funnel & Risk Analysis", "FastTrack Simulator"],
)

st.sidebar.divider()
st.sidebar.subheader("Filters")

channel_options = sorted(df["Channel"].dropna().unique().tolist())
credit_band_options = CREDIT_BAND_ORDER
employment_options = sorted(df["Employment_Type"].dropna().unique().tolist())

selected_channels = st.sidebar.multiselect(
    "Channel",
    options=channel_options,
    default=channel_options,
)
selected_credit_bands = st.sidebar.multiselect(
    "Credit Score Band",
    options=credit_band_options,
    default=credit_band_options,
)
selected_employment_types = st.sidebar.multiselect(
    "Employment Type",
    options=employment_options,
    default=employment_options,
)

filtered_df = apply_filters(
    df,
    selected_channels,
    selected_credit_bands,
    selected_employment_types,
)

if page != "FastTrack Simulator" and filtered_df.empty:
    st.warning("No applications match the selected filters. Select at least one value in each filter.")
    st.stop()


# -----------------------------------------------------------------------------
# Portfolio Overview
# -----------------------------------------------------------------------------
if page == "Portfolio Overview":
    st.title("Digital Retail Loan Product Analytics")
    st.caption("Interactive product and portfolio decision-support prototype")
    st.divider()
    st.header("Portfolio Overview")

    app_count = filtered_df["Application_ID"].nunique()
    approved_df = filtered_df[filtered_df["Approved_Flag"]]
    disbursed_df = filtered_df[filtered_df["Disbursed_Flag"]]
    approved_count = approved_df["Application_ID"].nunique()
    disbursed_count = disbursed_df["Application_ID"].nunique()
    approval_rate = safe_rate(approved_count, app_count)
    conversion_rate = safe_rate(disbursed_count, app_count)
    avg_approval_tat = approved_df["Approval_TAT_Hours"].mean()
    dpd_rate = disbursed_df["DPD_30"].mean() * 100 if not disbursed_df.empty else 0.0

    metric_cols = st.columns(5)
    metric_cols[0].metric("Applications", f"{app_count:,}")
    metric_cols[1].metric("Approval Rate", f"{approval_rate:.1f}%")
    metric_cols[2].metric("Overall Conversion", f"{conversion_rate:.1f}%")
    metric_cols[3].metric(
        "Avg Approval TAT",
        f"{avg_approval_tat:.1f} hrs" if pd.notna(avg_approval_tat) else "N/A",
        help="Calculated using approved applications only.",
    )
    metric_cols[4].metric(
        "30+ DPD Rate",
        f"{dpd_rate:.1f}%",
        help="Calculated among disbursed loans only.",
    )

    st.divider()
    st.subheader("Loan Application Funnel")
    st.caption("Applications to approval to disbursement")
    funnel = pd.DataFrame(
        {
            "Stage": ["Applications", "Approved", "Disbursed"],
            "Count": [app_count, approved_count, disbursed_count],
        }
    )
    fig_funnel = px.funnel(funnel, x="Count", y="Stage", text="Count")
    fig_funnel.update_layout(showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_funnel, use_container_width=True)

    st.divider()
    st.subheader("Channel Performance")
    channel_df = channel_summary(filtered_df)
    left, right = st.columns(2)

    with left:
        if not channel_df.empty:
            conversion_chart = channel_df.melt(
                id_vars="Channel",
                value_vars=["Disbursement Conversion"],
                var_name="Metric",
                value_name="Rate (%)",
            )
            fig_conversion = px.bar(
                conversion_chart,
                x="Channel",
                y="Rate (%)",
                text=conversion_chart["Rate (%)"].map(lambda value: f"{value:.1f}%"),
                title="Disbursement Conversion Rate by Channel",
            )
            fig_conversion.update_layout(showlegend=False, yaxis_title="Conversion Rate (%)")
            st.plotly_chart(fig_conversion, use_container_width=True)

    with right:
        tat_df = make_tat_by_channel(filtered_df)
        if not tat_df.empty:
            fig_tat = px.bar(
                tat_df,
                x="Channel",
                y="Avg_Approval_TAT_Hours",
                text="Avg_Approval_TAT_Hours",
                title="Average Approval TAT by Channel (Approved Applications Only)",
            )
            fig_tat.update_traces(texttemplate="%{text:.1f}", textposition="inside")
            fig_tat.update_layout(yaxis_title="Hours", xaxis_title="Channel")
            st.plotly_chart(fig_tat, use_container_width=True)

    st.dataframe(
        channel_df.style.format(
            {
                "Avg Approval TAT (Approved Apps, hrs)": "{:.1f}",
                "Approval Rate": "{:.1f}%",
                "Disbursement Conversion": "{:.1f}%",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "TAT charts and the Avg Approval TAT KPI use the same definition: mean Approval_TAT_Hours for approved applications only."
    )


# -----------------------------------------------------------------------------
# Funnel and risk analysis
# -----------------------------------------------------------------------------
elif page == "Funnel & Risk Analysis":
    st.title("Funnel & Risk Analysis")
    st.caption("Explore recorded drop-off stages, turnaround time and portfolio risk patterns.")
    st.divider()

    st.header("Where Are Applications Dropping Off?")
    dropoff_summary = make_dropoff_summary(filtered_df)
    if dropoff_summary.empty:
        st.info("No recorded drop-off stages appear in the selected data.")
    else:
        fig_dropoff = px.bar(
            dropoff_summary,
            x="Drop-off Stage",
            y="Applications",
            text="Applications",
            title="Applications by Recorded Drop-off Stage",
        )
        fig_dropoff.update_traces(textposition="inside")
        fig_dropoff.update_layout(xaxis_title="Recorded Stage", yaxis_title="Applications")
        st.plotly_chart(fig_dropoff, use_container_width=True)
        st.dataframe(
            dropoff_summary.style.format({"Share of All Applications (%)": "{:.1f}%"}),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Share of all filtered applications recorded at each drop-off stage. The dataset does not identify whether an application stopped because the customer abandoned it, documentation was incomplete, or the bank declined it."
        )

    st.divider()
    st.header("Risk and Conversion by Credit Score")
    risk_df = make_credit_risk_summary(filtered_df)
    left, right = st.columns(2)

    with left:
        fig_score_conversion = px.bar(
            risk_df,
            x="Credit Score Band",
            y="Conversion Rate (%)",
            text=risk_df["Conversion Rate (%)"].map(lambda value: f"{value:.1f}%"),
            title="Disbursement Conversion by Credit Score Band",
            category_orders={"Credit Score Band": CREDIT_BAND_ORDER},
        )
        fig_score_conversion.update_traces(textposition="inside")
        fig_score_conversion.update_layout(yaxis_title="Disbursement Conversion (%)")
        st.plotly_chart(fig_score_conversion, use_container_width=True)

    with right:
        fig_score_dpd = px.bar(
            risk_df,
            x="Credit Score Band",
            y="30+ DPD Rate Among Disbursed (%)",
            text=risk_df["30+ DPD Rate Among Disbursed (%)"].map(
                lambda value: f"{value:.1f}%" if pd.notna(value) else "N/A"
            ),
            title="30+ DPD Rate Among Disbursed Loans by Credit Score",
            category_orders={"Credit Score Band": CREDIT_BAND_ORDER},
        )
        fig_score_dpd.update_traces(textposition="inside")
        fig_score_dpd.update_layout(yaxis_title="30+ DPD Rate (%)")
        st.plotly_chart(fig_score_dpd, use_container_width=True)

    st.dataframe(
        risk_df.style.format(
            {
                "Conversion Rate (%)": "{:.1f}%",
                "30+ DPD Rate Among Disbursed (%)": lambda value: f"{value:.1f}%" if pd.notna(value) else "N/A",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Conversion is disbursed applications divided by all applications in each credit-score band. 30+ DPD rate is calculated only among disbursed loans, which is the relevant denominator for repayment performance."
    )

    st.divider()
    st.header("Turnaround Time Analysis")
    st.caption("Approval turnaround time is measured for approved applications only, consistently across the dashboard.")
    tat_df = make_tat_by_channel(filtered_df)
    if tat_df.empty:
        st.info("No approved applications with turnaround-time data match the selected filters.")
    else:
        fig_tat_analysis = px.bar(
            tat_df,
            x="Channel",
            y="Avg_Approval_TAT_Hours",
            text="Avg_Approval_TAT_Hours",
            title="Average Approval TAT by Channel (Approved Applications Only)",
        )
        fig_tat_analysis.update_traces(texttemplate="%{text:.1f} hrs", textposition="inside")
        fig_tat_analysis.update_layout(yaxis_title="Average Approval TAT (hours)")
        st.plotly_chart(fig_tat_analysis, use_container_width=True)


# -----------------------------------------------------------------------------
# FastTrack simulator
# -----------------------------------------------------------------------------
else:
    st.title("FastTrack Pre-Qualification")
    st.caption("Illustrative product prototype based on the FastTrack product requirements document.")

    st.info(
        "Purpose: provide an indicative eligibility result before the customer proceeds through the full loan journey. "
        "This is a student project prototype and does not represent actual bank underwriting policy."
    )

    st.divider()
    st.header("Customer Information")

    col1, col2 = st.columns(2)
    with col1:
        monthly_income = st.number_input(
            "Monthly Income (₹)",
            min_value=1,
            value=50000,
            step=5000,
            format="%d",
        )
        credit_score = st.slider("Credit Score", min_value=300, max_value=900, value=720)
        employment_type = st.selectbox(
            "Employment Type",
            options=employment_options,
            index=employment_options.index("Professional") if "Professional" in employment_options else 0,
        )
    with col2:
        monthly_obligations = st.number_input(
            "Existing Monthly Obligations (₹)",
            min_value=0,
            value=10000,
            step=1000,
            format="%d",
        )
        requested_loan_amount = st.number_input(
            "Requested Loan Amount (₹)",
            min_value=1,
            value=300000,
            step=25000,
            format="%d",
        )

    dti = (monthly_obligations / monthly_income) * 100 if monthly_income > 0 else 0.0

    st.divider()
    st.header("Indicative Customer Metrics")
    customer_metrics = st.columns(3)
    customer_metrics[0].metric("Monthly Income", inr(monthly_income))
    customer_metrics[1].metric("Existing Monthly Obligations", inr(monthly_obligations))
    customer_metrics[2].metric("Existing Obligations-to-Income", f"{dti:.1f}%")

    st.caption(
        "For this prototype, the ratio uses existing monthly obligations divided by monthly income. "
        "It does not include the EMI for the requested loan because the simulator does not calculate a proposed repayment schedule. "
        "The requested amount and employment type are captured for the concept but are not decision factors in the current illustrative rules."
    )

    if st.button("Check Indicative Eligibility", type="primary", use_container_width=True):
        status, explanation = get_preliminary_eligibility(int(credit_score), float(dti))
        st.divider()
        st.header("FastTrack Result")
        if status == "Eligible":
            st.success(f"Result: {status}")
        elif status == "Not Eligible":
            st.error(f"Result: {status} under the preliminary prototype rules")
        else:
            st.warning(f"Result: {status}")
        st.write(explanation)
        st.caption(
            "This result is indicative only. A real lender would apply its approved credit policy, verify customer information, "
            "complete required risk/compliance checks and make a formal decision."
        )

    st.divider()
    st.header("Illustrative Business Rules")
    rule_table = pd.DataFrame(
        [
            {
                "Condition": "Credit score at least 750 AND existing-obligation DTI at most 40%",
                "Prototype outcome": "Eligible",
            },
            {
                "Condition": "Credit score below 650 AND existing-obligation DTI above 40%",
                "Prototype outcome": "Not Eligible",
            },
            {
                "Condition": "All other combinations",
                "Prototype outcome": "Review Required",
            },
        ]
    )
    st.dataframe(rule_table, use_container_width=True, hide_index=True)
    st.caption(
        "The Not Eligible rule intentionally requires both risk flags, rather than either one alone, to avoid excluding a potential customer based on a single indicator. "
        "The thresholds are illustrative assumptions informed by portfolio patterns in this student dataset, not approved bank policy."
    )

    st.divider()
    st.header("Product Concept")
    st.markdown("**Current journey**")
    st.write("Customer -> Application -> KYC -> Credit Assessment -> Approval -> Disbursement")
    st.markdown("**Proposed journey**")
    st.write("Customer -> Basic Details -> FastTrack Pre-Qualification -> KYC -> Credit Assessment -> Approval -> Disbursement")
