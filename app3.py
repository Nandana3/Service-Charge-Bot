# app.py (Version 5.6 - All Employer Features Restored)

# --- 1. IMPORTS ---
import streamlit as st
import pandas as pd
import joblib
import pickle
from datetime import date, datetime
import plotly.express as px
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np

# --- 2. CONFIGURATION ---
DATA_FILE = 'kuwait_expat_labor_data.csv'
MODEL_FILE = 'dispute_model.joblib'
COLUMNS_FILE = 'model_columns.pkl'

# --- 3. DATA & MODEL LOADING ---
@st.cache_data
def load_data(filepath):
    """
    Loads data and correctly parses mixed DD-MM-YYYY and YYYY-MM-DD formats.
    """
    try:
        df = pd.read_csv(filepath)
        # Use format='mixed' to handle all date formats in the file
        df['start_date'] = pd.to_datetime(df['start_date'], format='mixed')
        df['end_date'] = pd.to_datetime(df['end_date'], format='mixed')
        df['tenure_years'] = (df['end_date'] - df['start_date']).dt.days / 365.25
        return df
    except FileNotFoundError:
        st.error(f"Error: The data file '{filepath}' was not found. Please make sure it's in the same folder as app.py.")
        return None

@st.cache_resource
def load_model(filepath):
    try:
        return joblib.load(filepath)
    except FileNotFoundError:
        return None

@st.cache_resource
def load_model_columns(filepath):
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

# --- 4. HELPER FUNCTIONS ---
def save_data(df, filepath):
    # Create a copy to avoid SettingWithCopyWarning
    df_copy = df.copy()
    # Ensure date columns are strings in a consistent format before saving
    df_copy.loc[:, 'start_date'] = pd.to_datetime(df_copy['start_date']).dt.strftime('%Y-%m-%d')
    df_copy.loc[:, 'end_date'] = pd.to_datetime(df_copy['end_date']).dt.strftime('%Y-%m-%d')
    df_copy.to_csv(filepath, index=False)
    st.cache_data.clear()


def calculate_indemnity_logic(years_of_service, monthly_salary, reason_for_leaving):
    if not monthly_salary or monthly_salary <= 0 or years_of_service <= 0:
        return {}
    daily_wage = monthly_salary / 26
    days_first_5 = min(years_of_service, 5) * 15
    days_after_5 = max(0, years_of_service - 5) * 30
    total_base_days = days_first_5 + days_after_5
    total_indemnity_base = total_base_days * daily_wage
    resignation_factor = 1.0
    resignation_text = "N/A (Termination)"
    if reason_for_leaving == "Resignation":
        if years_of_service < 3:
            resignation_factor = 0; resignation_text = "0% (Less than 3 years)"
        elif 3 <= years_of_service < 5:
            resignation_factor = 0.5; resignation_text = "50% (3 to 5 years)"
        elif 5 <= years_of_service < 10:
            resignation_factor = 2/3; resignation_text = "66.7% (5 to 10 years)"
        else:
            resignation_factor = 1.0; resignation_text = "100% (10+ years)"
    final_indemnity = total_indemnity_base * resignation_factor
    max_indemnity = 18 * monthly_salary
    was_capped = final_indemnity > max_indemnity
    final_indemnity = min(final_indemnity, max_indemnity)
    return {
        "final_indemnity": final_indemnity, "years_of_service": years_of_service,
        "daily_wage": daily_wage, "total_base_days": total_base_days,
        "total_indemnity_base": total_indemnity_base, "resignation_factor_text": resignation_text,
        "was_capped": was_capped, "max_indemnity": max_indemnity
    }

def predict_dispute(data, model, model_columns):
    if model is None or model_columns is None:
        st.error("Model not loaded.")
        return 0
    df_to_predict = pd.DataFrame([data])
    df_encoded = pd.get_dummies(df_to_predict)
    df_aligned = df_encoded.reindex(columns=model_columns, fill_value=0)
    prediction_proba = model.predict_proba(df_aligned)[0]
    return prediction_proba[1]

# --- 5. UI INTERFACE FUNCTIONS ---

def employee_interface(df):
    """Builds the UI for the Employee Portal."""
    st.title("Employee Portal")
    status = st.radio("What is your current employment status?", ("I am planning to leave my job (Calculator & Scenarios)", "I have already left my job (Data Entry & Calculation)"), key="employee_status")
    if "Calculator" in status:
        st.header("End-of-Service Indemnity Calculator")
        salary = st.number_input("Last Monthly Salary (KWD)", min_value=100.0, step=10.0, value=700.0)
        start_date_input = st.date_input("Employment Start Date", date(2020, 1, 1))
        end_date_input = st.date_input("Planned Employment End Date", date.today())
        reason = st.selectbox("Reason for Leaving", ['Resignation', 'Termination'])
        if st.button("Calculate Indemnity"):
            if end_date_input > start_date_input:
                years = (end_date_input - start_date_input).days / 365.25
                st.info(f"Calculated Years of Service: **{years:.3f} years**")
                result = calculate_indemnity_logic(years, salary, reason)
                st.success(f"Estimated Final Indemnity: **{result.get('final_indemnity', 0):.3f} KWD**")
                with st.expander("View Detailed Calculation Breakdown"):
                    st.subheader("Step-by-Step Breakdown")
                    st.markdown(f"""
                    - **Total Years of Service:** `{result.get('years_of_service', 0):.3f}`
                    - **Daily Wage Calculation:** `{salary:.2f} KWD / 26 days = {result.get('daily_wage', 0):.3f} KWD per day`
                    - **Base Indemnity (in KWD):** `{result.get('total_indemnity_base', 0):.3f} KWD`
                    - **Resignation Factor Applied:** `{result.get('resignation_factor_text', 'N/A')}`""")
                    if result.get('was_capped'):
                        st.warning(f"Note: The final indemnity was capped at the maximum of 18 months' salary ({result.get('max_indemnity', 0):.3f} KWD).")
            else:
                st.warning("Please ensure end date is after start date.")
        st.markdown("---")
        st.header("Scenario Analysis Tool")
        scenario_years = st.slider("Years of Service", 1.0, 25.0, 5.0, 0.5)
        scenario_salary = st.slider("Monthly Salary (KWD)", 100, 3000, 1000, 50)
        scenario_reason = st.selectbox("Scenario Reason", ['Resignation', 'Termination'], key="sc_reason")
        scenario_result = calculate_indemnity_logic(scenario_years, scenario_salary, scenario_reason)
        st.metric(label=f"Indemnity for a {scenario_reason.lower()} after {scenario_years} years", value=f"{scenario_result.get('final_indemnity', 0):.3f} KWD")
    else:
        st.header("Submit Your Details & Receive Final Calculation")
        with st.form("new_employee_form"):
            new_id = st.number_input("Your Employee ID", step=1, min_value=1, value=int(df['employee_id'].max() + 1))
            nationality = st.selectbox("Nationality Group", ['South Asian', 'Southeast Asian', 'Arab', 'African', 'Western'])
            industry = st.selectbox("Industry", ['Construction', 'Retail', 'IT', 'Hospitality', 'Health', 'Finance', 'Logistics', 'Oil & Gas'])
            job_category = st.selectbox("Job Category", ['Labourer', 'Helper', 'Driver', 'Electrician', 'Plumber', 'Cook', 'Clerk', 'Accountant', 'IT Professional', 'Nurse', 'Engineer', 'Manager'])
            salary_final = st.number_input("Final Monthly Salary (KWD)", min_value=100.0, step=10.0, value=750.0)
            start_date_final = st.date_input("Employment Start Date", date(2020, 1, 1))
            end_date_final = st.date_input("Employment End Date", date.today())
            reason_final = st.selectbox("Actual Reason for Leaving", ['Resignation', 'Termination', 'Contract End'])
            submitted = st.form_submit_button("Submit & Calculate My Indemnity")
            if submitted:
                if end_date_final > start_date_final:
                    years_final = (end_date_final - start_date_final).days / 365.25
                    result_final = calculate_indemnity_logic(years_final, salary_final, reason_final)
                    st.success(f"Calculated Indemnity: **{result_final.get('final_indemnity', 0):.3f} KWD**")
                    new_row = {'employee_id': new_id, 'nationality_group': nationality, 'industry': industry, 'job_category': job_category, 'monthly_salary_kwd': salary_final, 'start_date': start_date_final, 'end_date': end_date_final, 'reason_for_leaving': reason_final, 'filed_dispute': 0, 'tenure_years': years_final}
                    if new_id not in df['employee_id'].values:
                        new_df = pd.DataFrame([new_row])
                        updated_df = pd.concat([df, new_df], ignore_index=True)
                        save_data(updated_df, DATA_FILE)
                        st.info("Your anonymous data has been successfully saved.")
                    else:
                        st.warning(f"Employee ID {new_id} already exists.")
                else:
                    st.error("Error: The end date must be after the start date.")

def employer_interface(df, model, model_columns):
    """Builds the UI for the Employer Portal with all features integrated."""
    st.title("Employer Portal")
    
    st.header("Search or Add Employee")
    search_id = st.number_input("Enter Employee ID:", value=None, step=1)

    if search_id:
        employee_data = df[df['employee_id'] == search_id]
        
        if not employee_data.empty:
            # --- DISPLAY EXISTING EMPLOYEE ---
            employee_dict = employee_data.iloc[0].to_dict()
            
            st.subheader(f"Details for Employee ID: {search_id}")
            display_df = employee_data.drop(columns=['start_date', 'end_date']).T.copy()
            display_df.columns = ["Details"]
            st.table(display_df)
            
            st.subheader("Financial Liability")
            indemnity_result = calculate_indemnity_logic(employee_dict['tenure_years'], employee_dict['monthly_salary_kwd'], employee_dict['reason_for_leaving'])
            st.metric(label="Calculated End-of-Service Indemnity Owed", value=f"{indemnity_result.get('final_indemnity', 0):.3f} KWD")

            st.subheader("Dispute Analysis")
            dispute_prob = predict_dispute(employee_dict, model, model_columns)
            st.warning(f"Predicted Risk of Dispute: **{dispute_prob*100:.2f}%**")
            
            st.subheader("Update Dispute Status")
            current_status = "Yes" if employee_dict['filed_dispute'] == 1 else "No"
            
            dispute_filed = st.radio(
                f"Has this employee actually filed a formal dispute? (Current status: {current_status})",
                ("No", "Yes"),
                index=1 if current_status == "Yes" else 0,
                key=f"dispute_update_{search_id}"
            )
            
            if st.button("Update Status"):
                new_status = 1 if dispute_filed == "Yes" else 0
                row_index = df[df['employee_id'] == search_id].index[0]
                df.loc[row_index, 'filed_dispute'] = new_status
                save_data(df, DATA_FILE)
                st.success(f"Dispute status for employee {search_id} updated successfully!")
                st.rerun()

        else:
            # --- ADD NEW EMPLOYEE FORM (IF ID NOT FOUND) ---
            st.info(f"Employee ID {search_id} not found. You can add a new record below.")
            with st.form(f"new_employee_form_{search_id}"):
                st.subheader(f"Create New Employee Record for ID: {search_id}")
                
                nationality = st.selectbox("Nationality Group", ['South Asian', 'Southeast Asian', 'Arab', 'African', 'Western'])
                industry = st.selectbox("Industry", ['Construction', 'Retail', 'IT', 'Hospitality', 'Health', 'Finance', 'Logistics', 'Oil & Gas'])
                job_category = st.selectbox("Job Category", ['Labourer', 'Helper', 'Driver', 'Electrician', 'Plumber', 'Cook', 'Clerk', 'Accountant', 'IT Professional', 'Nurse', 'Engineer', 'Manager'])
                salary_final = st.number_input("Final Monthly Salary (KWD)", min_value=100.0, step=10.0)
                start_date_final = st.date_input("Employment Start Date")
                end_date_final = st.date_input("Employment End Date")
                reason_final = st.selectbox("Actual Reason for Leaving", ['Resignation', 'Termination', 'Contract End'])
                
                submitted = st.form_submit_button("Save New Employee")
                
                if submitted:
                    if end_date_final > start_date_final:
                        years_final = (end_date_final - start_date_final).days / 365.25
                        new_row = {
                            'employee_id': search_id,
                            'nationality_group': nationality,
                            'industry': industry,
                            'job_category': job_category,
                            'monthly_salary_kwd': salary_final,
                            'start_date': start_date_final,
                            'end_date': end_date_final,
                            'reason_for_leaving': reason_final,
                            'filed_dispute': 0,
                            'tenure_years': years_final
                        }
                        
                        new_df = pd.DataFrame([new_row])
                        updated_df = pd.concat([df, new_df], ignore_index=True)
                        save_data(updated_df, DATA_FILE)
                        st.success(f"Employee {search_id} has been added to the dataset successfully!")
                        st.rerun()
                    else:
                        st.error("Error: The end date must be after the start date.")
    
    st.markdown("---")
    # --- HYPOTHETICAL EMPLOYEE ANALYSIS ---
    st.header("Analyze a New or Hypothetical Employee")
    with st.form("hypothetical_employee_form"):
        st.write("Enter the details to see a dispute risk prediction and indemnity calculation.")
        
        hypo_years = st.slider("Years of Service", 1.0, 25.0, 5.0, 0.5)
        hypo_salary = st.number_input("Monthly Salary (KWD)", min_value=100.0, step=10.0, value=1000.0)
        hypo_reason = st.selectbox("Reason for Leaving", ['Resignation', 'Termination', 'Contract End'], key="h_reason")
        hypo_industry = st.selectbox("Industry", ['Construction', 'Retail', 'IT', 'Hospitality', 'Health', 'Finance', 'Logistics', 'Oil & Gas'], key="h_ind")
        hypo_job = st.selectbox("Job Category", ['Labourer', 'Helper', 'Driver', 'Electrician', 'Plumber', 'Cook', 'Clerk', 'Accountant', 'IT Professional', 'Nurse', 'Engineer', 'Manager'], key="h_job")
        
        predict_button = st.form_submit_button("Analyze Employee")
        
        if predict_button:
            # Calculate Indemnity
            hypo_indemnity_result = calculate_indemnity_logic(hypo_years, hypo_salary, hypo_reason)
            st.metric(label="Calculated Indemnity", value=f"{hypo_indemnity_result.get('final_indemnity', 0):.3f} KWD")

            # Predict Dispute Risk
            hypo_data = {'reason_for_leaving': hypo_reason, 'industry': hypo_industry, 'job_category': hypo_job, 'monthly_salary_kwd': hypo_salary}
            hypo_risk = predict_dispute(hypo_data, model, model_columns)
            st.metric(label="Predicted Dispute Risk", value=f"{hypo_risk*100:.2f}%")

def dashboard_interface(df):
    """Builds the UI for the Market Dashboard with an enhanced donut chart."""
    # This function remains unchanged.
    st.title("📊 Market Dashboard")
    st.subheader("Employee Distribution Across Industries")
    industry_dist = df['industry'].value_counts().reset_index()
    industry_dist.columns = ['Industry', 'Count']
    appealing_palette = ['#8ecae6', '#219ebc', '#023047', '#ffb703', '#fb8500', '#bde0fe', '#ffc8dd', '#cdb4db']
    fig_pie = px.pie(industry_dist, names='Industry', values='Count', color_discrete_sequence=appealing_palette, hole=0.5)
    fig_pie.update_traces(textposition='outside', textinfo='percent+label', textfont_size=14, marker=dict(line=dict(color='#FFFFFF', width=4)), pull=[0.05] * len(industry_dist))
    fig_pie.update_layout(title_text='Industry Sector Breakdown', title_x=0.5, title_font_size=24, height=600, showlegend=False, font=dict(family="Inter, sans-serif", color="#444"), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_pie, use_container_width=True)
    st.markdown("---")
    st.subheader("Average Salary Analysis")
    top_5_jobs = df['job_category'].value_counts().nlargest(5).index
    df_top_jobs = df[df['job_category'].isin(top_5_jobs)]
    salary_analysis = df_top_jobs.groupby(['job_category', 'reason_for_leaving'])['monthly_salary_kwd'].mean().reset_index()
    fig_bar = px.bar(salary_analysis, x='job_category', y='monthly_salary_kwd', color='reason_for_leaving', barmode='group', title='Average Salary by Job Role and Reason for Leaving', labels={'job_category': 'Job Category', 'monthly_salary_kwd': 'Average Monthly Salary (KWD)', 'reason_for_leaving': 'Reason for Leaving'}, text_auto='.3s')
    fig_bar.update_layout(title_x=0.5)
    st.plotly_chart(fig_bar, use_container_width=True)

@st.cache_data
def run_segmentation(_df):
    """
    Performs data preprocessing and clustering. Caching this heavy computation.
    """
    # This function remains unchanged.
    features = ['monthly_salary_kwd', 'tenure_years', 'industry', 'job_category']
    cluster_df = _df[features].dropna()
    numeric_features = ['monthly_salary_kwd', 'tenure_years']
    categorical_features = ['industry', 'job_category']
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)])
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('clusterer', kmeans)])
    cluster_labels = pipeline.fit_predict(cluster_df)
    cluster_df['cluster'] = cluster_labels
    preprocessed_data = pipeline.named_steps['preprocessor'].transform(cluster_df)
    pca = PCA(n_components=2, random_state=42)
    principal_components = pca.fit_transform(preprocessed_data)
    cluster_df['pca1'] = principal_components[:, 0]
    cluster_df['pca2'] = principal_components[:, 1]
    return cluster_df

def segmentation_interface(df):
    """Builds the UI for the Workforce Segmentation feature."""
    # This function remains unchanged.
    st.title("🤖 Workforce Segmentation")
    st.write("This page uses K-Means clustering to automatically discover segments of employees based on their salary, tenure, industry, and job role.")
    segmented_df = run_segmentation(df)
    st.subheader("Interactive Cluster Visualization")
    fig = px.scatter(segmented_df, x='pca1', y='pca2', color='cluster', color_continuous_scale=px.colors.qualitative.Vivid, hover_data=['job_category', 'industry', 'monthly_salary_kwd', 'tenure_years'])
    fig.update_layout(title="Employee Segments Visualized in 2D", xaxis_title="Principal Component 1", yaxis_title="Principal Component 2", legend_title_text='Cluster', title_x=0.5)
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Cluster Profiles")
    st.write("Summary statistics for each discovered employee segment.")
    for cluster_num in sorted(segmented_df['cluster'].unique()):
        st.write(f"**Cluster {cluster_num}**")
        cluster_data = segmented_df[segmented_df['cluster'] == cluster_num]
        description = cluster_data[['monthly_salary_kwd', 'tenure_years']].describe().loc[['mean', 'std', 'min', 'max']]
        st.table(description.style.format("{:.2f}"))
        st.write(f"**Most Common Job Category:** {cluster_data['job_category'].mode().iloc[0]}")
        st.write(f"**Most Common Industry:** {cluster_data['industry'].mode().iloc[0]}")
        st.markdown("---")

# --- 6. MAIN APP EXECUTION ---
def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(page_title="Kuwait Labor Law Advisor", layout="wide", initial_sidebar_state="expanded")
    
    main_df = load_data(DATA_FILE)
    dispute_model = load_model(MODEL_FILE)
    model_cols = load_model_columns(COLUMNS_FILE)

    st.sidebar.title("AIHR Kuwait Service Charge Bot - App Navigation")
    interface = st.sidebar.radio(
        "Select Your Interface", 
        ["Employee Portal", "Employer Portal", "Market Dashboard", "Workforce Segmentation"]
    )

    if main_df is None:
        st.warning("Please ensure the data file is available to use the app.")
        return

    if interface == "Employee Portal":
        employee_interface(main_df)
    elif interface == "Employer Portal":
        employer_interface(main_df, dispute_model, model_cols)
    elif interface == "Market Dashboard":
        dashboard_interface(main_df)
    elif interface == "Workforce Segmentation":
        segmentation_interface(main_df)

if __name__ == "__main__":
    main()
