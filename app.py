import streamlit as st
import pandas as pd
import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Digital Retail Loan Product",
    page_icon="၁၀၀",
    layout="wide"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("loan_applications.csv")
        return df
    except Exception as e:
        st.error(
            f"Error loading dataset: {e}. "
            "Please verify that loan_applications.csv is available."
        )
        return pd.DataFrame()


df = load_data()


# ============================================================
# DATA VALIDATION
# ============================================================

required_columns = [
    "Application_ID",
    "Age",
    "Gender",
    "City",
    "Employment_Type",
    "Monthly_Income",
    "Credit_Score",
    "Existing_Loans",
    "Monthly_Obligation",
    "Credit_Utilization",
    "Previous_Delinquency",
    "Loan_Amount",
    "Tenure_Months",
    "Interest_Rate",
    "Channel",
    "KYC_Status",
    "Approval_Status",
    "Approval_TAT_Hours",
    "Disbursement_Status",
    "Disbursement_TAT_Hours",
    "Dropoff_Stage",
    "First_EMI_Status",
    "DPD_30",
    "DPD_60",
    "DPD_90",
    "Collection_Status"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error(
        f"The following required columns are missing from the dataset: "
        f"{missing_columns}"
    )
    st.stop()


# ============================================================
# DATA PREPARATION
# ============================================================

if not df.empty:
    # Credit score segmentation
    df["Credit_Score_Band"] = pd.cut(
        df["Credit_Score"],
        bins=[0, 649, 699, 749, 900],
        labels=["<650", "650-699", "700-749", "750+"],
        include_lowest=True
    )

    # Income segmentation
    df["Income_Band"] = pd.cut(
        df["Monthly_Income"],
        bins=[0, 30000, 60000, 100000, float("inf")],
        labels=["<30K", "30K-59K", "60K-99K", "100K+"],
        include_lowest=True
    )

    # Approval flag
    df["Approved_Flag"] = (
        df["Approval_Status"]
        .astype(str)
        .str.lower()
        .eq("approved")
        .astype(int)
    )

    # Disbursement flag
    df["Disbursed_Flag"] = (
        df["Disbursement_Status"]
        .astype(str)
        .str.lower()
        .eq("disbursed")
        .astype(int)
    )

    # Debt/obligation-to-income ratio
    df["DTI"] = (
        df["Monthly_Obligation"] /
        df["Monthly_Income"].replace(0, pd.NA)
    )

    # Convert categorical values to strings where useful
    df["Channel"] = df["Channel"].astype(str)
    df["Employment_Type"] = df["Employment_Type"].astype(str)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_percentage(numerator, denominator):
    """Return percentage safely."""
    if denominator == 0:
        return 0
    return numerator / denominator


def calculate_kpis(data):
    """Calculate portfolio KPIs."""

    total_applications = len(data)

    approved = data["Approved_Flag"].sum() if "Approved_Flag" in data.columns else 0

    disbursed = data["Disbursed_Flag"].sum() if "Disbursed_Flag" in data.columns else 0

    approval_rate = safe_percentage(
        approved,
        total_applications
    )

    approval_to_disbursement = safe_percentage(
        disbursed,
        approved
    )

    overall_conversion = safe_percentage(
        disbursed,
        total_applications
    )

    dropoff_rate = 1 - overall_conversion

    approved_data = data[
        data["Approval_Status"]
        .astype(str)
        .str.lower()
        .eq("approved")
    ] if "Approval_Status" in data.columns else pd.DataFrame()

    disbursed_data = data[
        data["Disbursement_Status"]
        .astype(str)
        .str.lower()
        .eq("disbursed")
    ] if "Disbursement_Status" in data.columns else pd.DataFrame()

    avg_approval_tat = (
        approved_data["Approval_TAT_Hours"].mean()
        if not approved_data.empty and "Approval_TAT_Hours" in approved_data.columns
        else 0
    )

    avg_disbursement_tat = (
        disbursed_data["Disbursement_TAT_Hours"].mean()
        if not disbursed_data.empty and "Disbursement_TAT_Hours" in disbursed_data.columns
        else 0
    )

    dpd_rate = (
        disbursed_data["DPD_30"].mean()
        if not disbursed_data.empty and "DPD_30" in disbursed_data.columns
        else 0
    )

    return {
        "applications": total_applications,
        "approved": approved,
        "disbursed": disbursed,
        "approval_rate": approval_rate,
        "approval_to_disbursement": approval_to_disbursement,
        "conversion": overall_conversion,
        "dropoff": dropoff_rate,
        "approval_tat": avg_approval_tat,
        "disbursement_tat": avg_disbursement_tat,
        "dpd_rate": dpd_rate
    }


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Loan Product App")

st.sidebar.markdown(
    """
    **Digital Retail Loan Product Analytics**

    Use this prototype to analyse portfolio performance,
    identify funnel problems and test the FastTrack
    pre-qualification concept.
    """
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Portfolio Overview",
        "Funnel & Risk Analysis",
        "FastTrack Simulator"
    ]
)


# ============================================================
# SIDEBAR FILTERS
# ============================================================

if page in [
    "Portfolio Overview",
    "Funnel & Risk Analysis"
] and not df.empty:

    st.sidebar.divider()

    st.sidebar.subheader("Filters")

    channel_options = sorted(
        df["Channel"].dropna().unique().tolist()
    )

    selected_channels = st.sidebar.multiselect(
        "Channel",
        options=channel_options,
        default=channel_options
    )

    score_options = [
        "<650",
        "650-699",
        "700-749",
        "750+"
    ]

    selected_scores = st.sidebar.multiselect(
        "Credit Score Band",
        options=score_options,
        default=score_options
    )

    employment_options = sorted(
        df["Employment_Type"].dropna().unique().tolist()
    )

    selected_employment = st.sidebar.multiselect(
        "Employment Type",
        options=employment_options,
        default=employment_options
    )

    filtered_df = df[
        df["Channel"].isin(selected_channels) &
        df["Credit_Score_Band"].astype(str).isin(selected_scores) &
        df["Employment_Type"].isin(selected_employment)
    ]

else:
    filtered_df = df.copy()


# ============================================================
# PAGE 1 — PORTFOLIO OVERVIEW
# ============================================================

if page == "Portfolio Overview":

    st.title("၁၀၀ Digital Retail Loan Product Analytics")

    st.caption(
        "Interactive product and portfolio decision-support prototype"
    )

    st.divider()

    # --------------------------------------------------------
    # KPI CALCULATIONS
    # --------------------------------------------------------

    kpis = calculate_kpis(filtered_df)

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    st.subheader("Portfolio Overview")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Applications",
        f"{kpis['applications']:,}"
    )

    col2.metric(
        "Approval Rate",
        f"{kpis['approval_rate']:.1%}"
    )

    col3.metric(
        "Overall Conversion",
        f"{kpis['conversion']:.1%}"
    )

    col4.metric(
        "Avg Approval TAT",
        f"{kpis['approval_tat']:.1f} hrs"
    )

    col5.metric(
        "30+ DPD",
        f"{kpis['dpd_rate']:.1%}"
    )

    st.divider()

    # --------------------------------------------------------
    # LOAN FUNNEL
    # --------------------------------------------------------

    st.subheader("Loan Application Funnel")

    funnel_data = pd.DataFrame({
        "Stage": [
            "Applications",
            "Approved",
            "Disbursed"
        ],
        "Count": [
            kpis["applications"],
            kpis["approved"],
            kpis["disbursed"]
        ]
    })

    fig_funnel = px.funnel(
        funnel_data,
        x="Count",
        y="Stage",
        title="Application → Approval → Disbursement"
    )

    st.plotly_chart(
        fig_funnel,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # CHANNEL ANALYSIS
    # --------------------------------------------------------

    st.subheader("Channel Performance")

    if not filtered_df.empty:

        channel_analysis = (
            filtered_df
            .groupby("Channel")
            .agg(
                Applications=("Application_ID", "count"),
                Approved=("Approved_Flag", "sum"),
                Disbursed=("Disbursed_Flag", "sum"),
                Avg_Approval_TAT=("Approval_TAT_Hours", "mean")
            )
            .reset_index()
        )

        channel_analysis["Approval_Rate"] = (
            channel_analysis["Approved"] /
            channel_analysis["Applications"]
        )

        channel_analysis["Conversion_Rate"] = (
            channel_analysis["Disbursed"] /
            channel_analysis["Applications"]
        )

        col1, col2 = st.columns(2)

        with col1:

            fig_channel_conversion = px.bar(
                channel_analysis,
                x="Channel",
                y="Conversion_Rate",
                title="Conversion Rate by Channel",
                text_auto=".1%"
            )

            fig_channel_conversion.update_layout(
                yaxis_tickformat=".0%"
            )

            st.plotly_chart(
                fig_channel_conversion,
                use_container_width=True
            )

        with col2:

            fig_channel_tat = px.bar(
                channel_analysis,
                x="Channel",
                y="Avg_Approval_TAT",
                title="Average Approval TAT by Channel",
                text_auto=".1f"
            )

            st.plotly_chart(
                fig_channel_tat,
                use_container_width=True
            )

        st.dataframe(
            channel_analysis.style.format({
                "Approval_Rate": "{:.1%}",
                "Conversion_Rate": "{:.1%}",
                "Avg_Approval_TAT": "{:.1f}"
            }),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "No records match the selected filters."
        )


# ============================================================
# PAGE 2 — FUNNEL & RISK ANALYSIS
# ============================================================

elif page == "Funnel & Risk Analysis":

    st.title("၀၂ Funnel & Risk Analysis")

    st.caption(
        "Diagnose customer drop-offs, turnaround-time drivers "
        "and portfolio-risk patterns."
    )

    st.divider()

    if filtered_df.empty:

        st.warning(
            "No records match the selected filters."
        )
        st.stop()

    # --------------------------------------------------------
    # DROP-OFF ANALYSIS
    # --------------------------------------------------------

    st.subheader("Where Are Customers Dropping Off?")

    dropoff_analysis = (
        filtered_df
        .groupby("Dropoff_Stage")
        .size()
        .reset_index(name="Applications")
    )

    dropoff_analysis["Dropoff_Rate"] = (
        dropoff_analysis["Applications"] /
        len(filtered_df)
    )

    dropoff_analysis = dropoff_analysis.sort_values(
        "Applications",
        ascending=False
    )

    fig_dropoff = px.bar(
        dropoff_analysis,
        x="Dropoff_Stage",
        y="Applications",
        title="Applications by Drop-off Stage",
        text_auto=True
    )

    st.plotly_chart(
        fig_dropoff,
        use_container_width=True
    )

    st.dataframe(
        dropoff_analysis.style.format({
            "Dropoff_Rate": "{:.1%}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # --------------------------------------------------------
    # CREDIT SCORE RISK ANALYSIS
    # --------------------------------------------------------

    st.subheader("Risk–Conversion Analysis")

    risk_analysis = (
        filtered_df
        .groupby("Credit_Score_Band", observed=False)
        .agg(
            Applications=("Application_ID", "count"),
            Disbursed=("Disbursed_Flag", "sum"),
            DPD_30=("DPD_30", "mean")
        )
        .reset_index()
    )

    risk_analysis["Conversion_Rate"] = (
        risk_analysis["Disbursed"] /
        risk_analysis["Applications"]
    )

    col1, col2 = st.columns(2)

    with col1:

        fig_conversion = px.bar(
            risk_analysis,
            x="Credit_Score_Band",
            y="Conversion_Rate",
            title="Conversion by Credit Score Band",
            text_auto=".1%"
        )

        fig_conversion.update_layout(
            yaxis_tickformat=".0%"
        )

        st.plotly_chart( 
            fig_conversion,
            use_container_width=True
        )

    with col2:

        fig_dpd = px.bar(
            risk_analysis,
            x="Credit_Score_Band",
            y="DPD_30",
            title="30+ DPD by Credit Score Band",
            text_auto=".1%"
        )

        fig_dpd.update_layout(
            yaxis_tickformat=".0%"
        )

        st.plotly_chart(
            fig_dpd,
            use_container_width=True
        )

    st.dataframe(
        risk_analysis.style.format({
            "Conversion_Rate": "{:.1%}",
            "DPD_30": "{:.1%}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # --------------------------------------------------------
    # TAT ANALYSIS
    # --------------------------------------------------------

    st.subheader("Turnaround Time Analysis")

    approved_data = filtered_df[
        filtered_df["Approval_Status"]
        .astype(str)
        .str.lower()
        .eq("approved")
    ]

    if not approved_data.empty:

        tat_analysis = (
            approved_data
            .groupby("Channel")
            .agg(
                Avg_Approval_TAT=("Approval_TAT_Hours", "mean"),
                Applications=("Application_ID", "count")
            )
            .reset_index()
            .sort_values(
                "Avg_Approval_TAT",
                ascending=False
            )
        )

        fig_tat = px.bar(
            tat_analysis,
            x="Channel",
            y="Avg_Approval_TAT",
            title="Average Approval TAT by Channel",
            text_auto=".1f"
        )

        fig_tat.update_layout(
            yaxis_title="Hours"
        )

        st.plotly_chart(
            fig_tat,
            use_container_width=True
        )

    else:

        st.info(
            "No approved records available for TAT analysis."
        )


# ============================================================
# PAGE 3 — FASTTRACK PRE-QUALIFICATION
# ============================================================

elif page == "FastTrack Simulator":

    st.title("၀၄ FastTrack Pre-Qualification")

    st.caption(
        "Illustrative product prototype based on the PRD."
    )

    st.info(
        """
        **Purpose:** Provide an indicative eligibility result before
        the customer proceeds through the complete loan journey.

        This is a project prototype and does not represent actual
        bank underwriting policy.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # CUSTOMER INPUTS
    # --------------------------------------------------------

    st.subheader("Customer Information")

    col1, col2 = st.columns(2)

    with col1:

        income = st.number_input(
            "Monthly Income ($)",
            min_value=10000,
            max_value=1000000,
            value=50000,
            step=5000
        )

        credit_score = st.slider(
            "Credit Score",
            min_value=300,
            max_value=900,
            value=720,
            step=1
        )

        employment = st.selectbox(
            "Employment Type",
            sorted(
                df["Employment_Type"]
                .dropna()
                .unique()
                .tolist()
            ) if "Employment_Type" in df.columns else []
        )

    with col2:

        obligation = st.number_input(
            "Existing Monthly Obligations ($)",
            min_value=0,
            max_value=500000,
            value=10000,
            step=1000
        )

        loan_amount = st.number_input(
            "Requested Loan Amount ($)",
            min_value=50000,
            max_value=5000000,
            value=300000,
            step=25000
        )

    # --------------------------------------------------------
    # DERIVED METRIC
    # --------------------------------------------------------

    if income > 0:
        dti = obligation / income
    else:
        dti = 0

    st.divider()

    st.subheader("Indicative Customer Metrics")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Monthly Income",
        f"•{income:,.0f}"
    )

    c2.metric(
        "Monthly Obligations",
        f"•{obligation:,.0f}"
    )

    c3.metric(
        "Obligation-to-Income",
        f"{dti:.1%}"
    )

    st.divider()

    # --------------------------------------------------------
    # ELIGIBILITY BUTTON
    # --------------------------------------------------------

    if st.button(
        "၀၃ Check Indicative Eligibility",
        type="primary",
        use_container_width=True
    ):

        validation_errors = []

        if income <= 0:
            validation_errors.append(
                "Monthly income must be greater than zero."
            )

        if credit_score < 300 or credit_score > 900:
            validation_errors.append(
                "Credit score must be between 300 and 900."
            )

        if loan_amount <= 0:
            validation_errors.append(
                "Loan amount must be greater than zero."
            )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if validation_errors:

            st.error("Please correct the following:")

            for error in validation_errors:
                st.write(f"• {error}")

        else:

            # ------------------------------------------------
            # ILLUSTRATIVE PRODUCT RULES
            # ------------------------------------------------

            if (
                credit_score >= 750
                and dti <= 0.40
            ):

                result = "Eligible"

                reason = (
                    "High credit score and acceptable "
                    "obligation-to-income ratio."
                )

            elif (
                credit_score >= 700
                and dti <= 0.50
            ):

                result = "Review Required"

                reason = (
                    "Customer falls within the intermediate "
                    "risk band and requires additional assessment."
                )

            elif dti > 0.50:

                result = "Review Required"

                reason = (
                    "Obligation-to-income ratio exceeds "
                    "the primary eligibility threshold."
                )

            else:

                result = "Review Required"

                reason = (
                    "Credit profile requires further "
                    "assessment before proceeding."
                )

            # ------------------------------------------------
            # DISPLAY RESULT
            # ------------------------------------------------

            st.subheader("Pre-Qualification Result")

            if result == "Eligible":

                st.success(
                    "၁၀ ELIGIBLE"
                )

            else:

                st.warning(
                    "၁၁ REVIEW REQUIRED"
                )

            st.write(
                f"**Reason:** {reason}"
            )

            st.metric(
                "Indicative Loan Request",
                f"•{loan_amount:,.0f}"
            )

            st.caption(
                """
                This is an illustrative pre-qualification outcome.
                It is not a final credit decision. Final eligibility
                requires full credit assessment and applicable policies.
                """
            )

    # --------------------------------------------------------
    # PRODUCT EXPLANATION
    # --------------------------------------------------------

    st.divider()

    st.subheader("Product Concept")

    st.markdown(
        """
        ### FastTrack Pre-Qualification

        **Current journey**

        Customer → Application → KYC → Credit Assessment → Approval → Disbursement

        **Proposed journey**

        Customer → Basic Details → **FastTrack Pre-Qualification**
        → KYC → Credit Assessment → Approval → Disbursement

        The feature aims to provide customers with early indicative
        eligibility while retaining the final credit decision within
        the standard assessment process.
        """
    ) 


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Academic/Product Management Prototype | "
    "Synthetic retail-loan dataset | "
    "Illustrative rules only"
)
