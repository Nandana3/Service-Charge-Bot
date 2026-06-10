# create_flowchart.py

from graphviz import Digraph

# Create a new directed graph
dot = Digraph('AppFlowchart', comment='Application Workflow')
dot.attr(rankdir='TB', splines='ortho', nodesep='0.6', ranksep='0.8')
dot.attr('node', shape='box', style='rounded,filled', fillcolor='#EFEFEF', fontname='Helvetica')
dot.attr('edge', fontname='Helvetica', fontsize='10')

# --- Main Flow ---
with dot.subgraph(name='cluster_main') as c:
    c.attr(style='invis')
    c.node('start', 'Start app.py', shape='ellipse', fillcolor='#A9D18E')
    c.node('load_files', 'Load Data, Model, & Columns')
    c.node('sidebar', 'Display Sidebar Navigation')
    c.node('select_interface', 'User Selects Interface', shape='diamond', fillcolor='#F4D03F')
    
    c.edge('start', 'load_files')
    c.edge('load_files', 'sidebar')
    c.edge('sidebar', 'select_interface')

# --- Employee Portal Path ---
with dot.subgraph(name='cluster_employee') as c:
    c.attr(label='Employee Portal', style='rounded', color='blue')
    c.node('employee_ui', 'Display Employee UI')
    c.node('employee_choice', 'Planning to Leave or Already Left?', shape='diamond', fillcolor='#F4D03F')
    c.node('calculator', 'Show Indemnity Calculator\n& Scenario Sliders')
    c.node('data_entry', 'Show Data Entry Form')
    c.node('save_new_employee', 'Save New Record to CSV', shape='cylinder', fillcolor='#AED6F1')

    c.edge('employee_ui', 'employee_choice')
    c.edge('employee_choice', 'calculator', label='Planning to Leave')
    c.edge('employee_choice', 'data_entry', label='Already Left')
    c.edge('data_entry', 'save_new_employee')

# --- Employer Portal Path ---
with dot.subgraph(name='cluster_employer') as c:
    c.attr(label='Employer Portal', style='rounded', color='red')
    c.node('employer_ui', 'Display Employer UI\n& Search Box')
    c.node('id_found', 'Employee ID Found?', shape='diamond', fillcolor='#F4D03F')
    c.node('display_data', '1. Display Data Table\n2. Calculate Indemnity\n3. Predict Dispute Risk')
    c.node('update_status_option', 'Offer Option to\nUpdate Dispute Status')
    c.node('update_choice', 'User Updates?', shape='diamond', fillcolor='#F4D03F')
    c.node('save_update', 'Save Update to CSV', shape='cylinder', fillcolor='#AED6F1')

    c.edge('employer_ui', 'id_found')
    c.edge('id_found', 'display_data', label='Yes')
    c.edge('id_found', 'employer_ui', label='No, Search Again')
    c.edge('display_data', 'update_status_option')
    c.edge('update_status_option', 'update_choice')
    c.edge('update_choice', 'save_update', label='Yes')

# --- Dashboard Path ---
with dot.subgraph(name='cluster_dashboard') as c:
    c.attr(label='Market Dashboard', style='rounded', color='green')
    c.node('dashboard_ui', 'Display Dashboard UI')
    c.node('gen_charts', 'Generate & Display All Charts')
    
    c.edge('dashboard_ui', 'gen_charts')

# --- Connecting Main Flow to Branches ---
dot.edge('select_interface', 'employee_ui', label='Employee')
dot.edge('select_interface', 'employer_ui', label='Employer')
dot.edge('select_interface', 'dashboard_ui', label='Dashboard')

# Render the graph to a file
dot.render('app_flowchart', format='png', view=False, cleanup=True)

print("Flowchart successfully generated as 'app_flowchart.png'")