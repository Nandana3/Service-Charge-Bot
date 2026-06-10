import pandas as pd
import numpy as np
import random
from datetime import date, timedelta

# --- Configuration & Data Definitions ---

# Based on Embassy of India, Kuwait Minimum Wage Schedule
JOB_DATA = {
    'Labourer': {'min_wage': 100, 'salary_range': (100, 140)},
    'Helper': {'min_wage': 100, 'salary_range': (100, 150)},
    'Driver': {'min_wage': 120, 'salary_range': (120, 200)},
    'Electrician': {'min_wage': 125, 'salary_range': (125, 250)},
    'Plumber': {'min_wage': 125, 'salary_range': (125, 250)},
    'Cook': {'min_wage': 120, 'salary_range': (120, 220)},
    'Clerk': {'min_wage': 150, 'salary_range': (150, 250)},
    'Accountant': {'min_wage': 200, 'salary_range': (200, 450)},
    'IT Professional': {'min_wage': 250, 'salary_range': (250, 800)},
    'Nurse': {'min_wage': 330, 'salary_range': (330, 600)},
    'Engineer': {'min_wage': 450, 'salary_range': (450, 1200)},
    'Manager': {'min_wage': 500, 'salary_range': (500, 2000)},
}

INDUSTRIES = ['Construction', 'Retail', 'IT', 'Hospitality', 'Health', 'Finance', 'Logistics', 'Oil & Gas']
NATIONALITY_GROUPS = ['South Asian', 'Southeast Asian', 'Arab', 'African', 'Western']
REASONS_FOR_LEAVING = ['Resignation', 'Termination', 'Contract End']
NUMBER_OF_RECORDS = 5000

# --- Data Generation Logic ---

def create_synthetic_dataset(num_records):
    """Generates a synthetic dataset of expat labor data in Kuwait."""
    records = []
    for i in range(num_records):
        employee_id = 1001 + i
        
        # Weighted choices for realism
        nationality = np.random.choice(NATIONALITY_GROUPS, p=[0.55, 0.20, 0.15, 0.05, 0.05])
        industry = np.random.choice(INDUSTRIES, p=[0.3, 0.15, 0.1, 0.1, 0.1, 0.05, 0.1, 0.1])
        job_category = random.choice(list(JOB_DATA.keys()))
        
        # Generate realistic salary
        min_sal, max_sal = JOB_DATA[job_category]['salary_range']
        monthly_salary = round(random.uniform(min_sal, max_sal), 2)
        
        # Generate realistic employment dates
        end_date = date(2025, 1, 1) - timedelta(days=random.randint(30, 365*15))
        tenure_days = random.randint(365, 365*12) # Tenure between 1 and 12 years
        start_date = end_date - timedelta(days=tenure_days)
        
        reason_for_leaving = np.random.choice(REASONS_FOR_LEAVING, p=[0.6, 0.3, 0.1])
        
        # --- Logic for the target variable 'filed_dispute' ---
        # Disputes are more likely in certain conditions
        dispute_probability = 0.05 # Base probability
        if reason_for_leaving == 'Termination':
            dispute_probability += 0.25
        # Higher chance of dispute if salary is in the bottom 20% of the possible range
        if monthly_salary < (min_sal + 0.2 * (max_sal - min_sal)):
            dispute_probability += 0.15
        if (end_date - start_date).days < 365 * 2: # Higher chance for shorter tenures
            dispute_probability += 0.05
            
        filed_dispute = 1 if random.random() < dispute_probability else 0
        
        records.append({
            'employee_id': employee_id,
            'nationality_group': nationality,
            'industry': industry,
            'job_category': job_category,
            'monthly_salary_kwd': monthly_salary,
            'start_date': start_date,
            'end_date': end_date,
            'reason_for_leaving': reason_for_leaving,
            'filed_dispute': filed_dispute
        })
        
    return pd.DataFrame(records)

# --- Main execution block ---
if __name__ == "__main__":
    print("Generating synthetic dataset...")
    df = create_synthetic_dataset(NUMBER_OF_RECORDS)
    
    # Save to CSV
    file_path = 'kuwait_expat_labor_data.csv'
    df.to_csv(file_path, index=False)
    
    print(f"\nSuccessfully generated dataset with {len(df)} records.")
    print(f"File saved as: {file_path}")
    print("\nFirst 5 rows of your new dataset:")
    print(df.head())