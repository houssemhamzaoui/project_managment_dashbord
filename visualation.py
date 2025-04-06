import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import psycopg2
import pandas as pd
import plotly.express as px
import re
from datetime import datetime

# Security enhancements
def sanitize_query(query):
    """Basic SQL injection protection"""
    forbidden_patterns = [
        r";\s*--", r";\s*#", r"\/\*", r"\*\/", 
        r"\b(drop|alter|truncate|grant|revoke|shutdown)\b",
        r"\b(insert|update|delete|create|replace)\b"
    ]
    for pattern in forbidden_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return None
    return query

def get_schema_info(_conn):
    """Get read-only schema information"""
    schema = {}
    try:
        # Create a new cursor for this operation
        with _conn.cursor() as cur:
            try:
                cur.execute("""
                    SELECT table_name, column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
                    ORDER BY table_name, ordinal_position
                """)
                # Explicitly commit if autocommit is False
                if not _conn.autocommit:
                    _conn.commit()
                
                for table, column, dtype in cur.fetchall():
                    if table not in schema:
                        schema[table] = []
                    schema[table].append(f"{column} ({dtype})")
            except psycopg2.Error as e:
                _conn.rollback()
                st.error(f"Schema query failed: {str(e)}")
                return {}
    except Exception as e:
        st.error(f"Schema inspection failed: {str(e)}")
        return {}
    return schema

def execute_select_only(_conn, query):
    """Execute only SELECT queries with row limit"""
    try:
        # Security checks
        query = sanitize_query(query)
        if not query:
            st.error("Query blocked by security policy")
            return None
            
        if not query.lower().strip().startswith(('select', 'with')):
            st.error("Only SELECT queries are allowed")
            return None
            
        # Remove any existing semicolons and whitespace
        query = query.rstrip(';').strip()
            
        # Add LIMIT if not present (safety measure)
        if "limit" not in query.lower():
            query += " LIMIT 1000"
            
        with _conn.cursor() as cur:
            try:
                start_time = datetime.now()
                cur.execute(query)
                
                if not _conn.autocommit:
                    _conn.commit()
                
                if not cur.description:
                    st.error("No results returned - not a SELECT query?")
                    return None
                    
                columns = [desc[0] for desc in cur.description]
                data = cur.fetchall()
                elapsed = (datetime.now() - start_time).total_seconds()
                
                df = pd.DataFrame(data, columns=columns)
                st.info(f"Query executed in {elapsed:.2f} seconds. Returned {len(df)} rows.")
                return df
            except psycopg2.Error as e:
                _conn.rollback()
                st.error(f"PostgreSQL error: {str(e)}")
                return None
                
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return None

@st.cache_resource
def get_db_connection():
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="project_managment",
            user="postgres",
            password="hosshoss",
            port="5432",
            connect_timeout=5
        )
        # Set autocommit to True to avoid transaction issues
        conn.autocommit = True
        return conn
    except Exception as e:
        st.error(f"PostgreSQL connection failed: {str(e)}")
        return None

def determine_best_chart(df):
    """Automatically determine the best chart type based on data characteristics"""
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Default values
    chart_type = "Bar Chart"
    x_col = df.columns[0]
    y_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    
    # Rule-based chart selection
    if len(date_cols) >= 1 and len(numeric_cols) >= 1:
        # Time series data - use line chart
        chart_type = "Line Chart"
        x_col = date_cols[0]
        y_col = numeric_cols[0]
    elif len(cat_cols) >= 1 and len(numeric_cols) >= 1:
        # Categorical vs numeric - use bar chart
        chart_type = "Bar Chart"
        x_col = cat_cols[0]
        y_col = numeric_cols[0]
    elif len(numeric_cols) >= 2:
        # Two numeric variables - use scatter plot
        chart_type = "Scatter Plot"
        x_col = numeric_cols[0]
        y_col = numeric_cols[1]
    elif len(numeric_cols) >= 1:
        # Single numeric variable - use histogram
        chart_type = "Histogram"
        x_col = numeric_cols[0]
    elif len(cat_cols) >= 1:
        # Single categorical variable - use pie chart
        chart_type = "Pie Chart"
        x_col = cat_cols[0]
        if len(df.columns) > 1:
            y_col = df.columns[1]  # Use second column as values if available
    
    return chart_type, x_col, y_col

def show_visualizations(df):
    """Create interactive Plotly charts based on dataframe"""
    st.subheader("Data Visualizations")
    
    # Auto-detect column types
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Determine best initial chart
    if 'chart_type' not in st.session_state:
        st.session_state.chart_type, st.session_state.x_col, st.session_state.y_col = determine_best_chart(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Chart type selection
        chart_types = ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Pie Chart"]
        new_chart_type = st.selectbox(
            "Chart type", 
            chart_types,
            index=chart_types.index(st.session_state.chart_type),
            key='viz_chart_type'
        )
        if new_chart_type != st.session_state.chart_type:
            st.session_state.chart_type = new_chart_type
            # Reset to default columns when chart type changes
            st.session_state.x_col = df.columns[0]
            st.session_state.y_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    
    with col2:
        if st.session_state.chart_type in ["Bar Chart", "Line Chart", "Scatter Plot"]:
            # Convert columns to list for index() method
            columns_list = df.columns.tolist()
            
            # X-axis selection
            new_x_col = st.selectbox(
                "X-axis", 
                columns_list,
                index=columns_list.index(st.session_state.x_col) if st.session_state.x_col in columns_list else 0,
                key='viz_x_axis'
            )
            
            # Y-axis selection (only numeric for these chart types)
            y_options = numeric_cols if numeric_cols else columns_list
            new_y_col = st.selectbox(
                "Y-axis", 
                y_options,
                index=y_options.index(st.session_state.y_col) if st.session_state.y_col in y_options else 0,
                key='viz_y_axis'
            )
            
            if new_x_col != st.session_state.x_col:
                st.session_state.x_col = new_x_col
            if new_y_col != st.session_state.y_col:
                st.session_state.y_col = new_y_col
                
        elif st.session_state.chart_type == "Histogram":
            # Column selection (prefer numeric)
            x_options = numeric_cols if numeric_cols else df.columns.tolist()
            new_x_col = st.selectbox(
                "Column", 
                x_options,
                index=x_options.index(st.session_state.x_col) if st.session_state.x_col in x_options else 0,
                key='viz_hist_col'
            )
            if new_x_col != st.session_state.x_col:
                st.session_state.x_col = new_x_col
                
        elif st.session_state.chart_type == "Pie Chart":
            # Category selection (prefer categorical)
            x_options = cat_cols if cat_cols else df.columns.tolist()
            new_x_col = st.selectbox(
                "Category", 
                x_options,
                index=x_options.index(st.session_state.x_col) if st.session_state.x_col in x_options else 0,
                key='viz_pie_cat'
            )
            
            # Value selection (prefer numeric)
            y_options = numeric_cols if numeric_cols else df.columns.tolist()
            new_y_col = st.selectbox(
                "Value", 
                y_options,
                index=y_options.index(st.session_state.y_col) if st.session_state.y_col in y_options else 0,
                key='viz_pie_val'
            )
            
            if new_x_col != st.session_state.x_col:
                st.session_state.x_col = new_x_col
            if new_y_col != st.session_state.y_col:
                st.session_state.y_col = new_y_col
    
    try:
        if st.session_state.chart_type == "Bar Chart":
            fig = px.bar(df, x=st.session_state.x_col, y=st.session_state.y_col, 
                         title=f"{st.session_state.y_col} by {st.session_state.x_col}")
        elif st.session_state.chart_type == "Line Chart":
            fig = px.line(df, x=st.session_state.x_col, y=st.session_state.y_col, 
                          title=f"{st.session_state.y_col} over {st.session_state.x_col}")
        elif st.session_state.chart_type == "Scatter Plot":
            fig = px.scatter(df, x=st.session_state.x_col, y=st.session_state.y_col, 
                             title=f"{st.session_state.y_col} vs {st.session_state.x_col}")
        elif st.session_state.chart_type == "Histogram":
            fig = px.histogram(df, x=st.session_state.x_col, 
                               title=f"Distribution of {st.session_state.x_col}")
        elif st.session_state.chart_type == "Pie Chart":
            fig = px.pie(df, names=st.session_state.x_col, values=st.session_state.y_col, 
                         title=f"{st.session_state.y_col} by {st.session_state.x_col}")
            
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Could not generate visualization: {str(e)}")

class ProjectQueryInterface:
    def __init__(self, conn, project_id):
        self.conn = conn
        self.project_id = project_id
        self.schema = get_schema_info(conn)
        
    def generate_project_context(self):
        """Generate context that includes the project ID constraint"""
        return f"""
        All queries should be filtered to only include data for project_id = {self.project_id}.
        The project ID must appear in all WHERE clauses for queries about project-specific data.
        """
    
    def generate_query(self, question):
        """Generate a query scoped to the specific project"""
        with st.spinner("Generating project-specific query..."):
            try:
                prompt = f"""### Database Schema:
{self.schema}

### Special Instructions:
{self.generate_project_context()}

### Question:
{question}

### Safe SELECT Query:"""
                
                inputs = st.session_state.tokenizer(
                    prompt, 
                    return_tensors="pt"
                ).to(st.session_state.model.device)
                
                outputs = st.session_state.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=st.session_state.tokenizer.eos_token_id
                )
                
                query = st.session_state.tokenizer.decode(
                    outputs[0], 
                    skip_special_tokens=True
                ).split("### Safe SELECT Query:")[-1].strip()
                
                return query
                
            except Exception as e:
                st.error(f"Query generation failed: {str(e)}")
                return None

def main():
    st.set_page_config(
        page_title="Project Analytics",
        page_icon="🔍",
        layout="wide"
    )
    
    # Check for project_id in URL parameters using the new st.query_params
    project_id = st.query_params.get("project_id", None)
    
    # Initialize database connection
    conn = get_db_connection()
    if conn is None:
        st.stop()
    
    # If no project ID in URL, show project selection
    if not project_id:
        show_project_selection(conn)
        return
    
    # Otherwise show the query interface for the specific project
    show_project_query_interface(conn, project_id)

def show_project_selection(conn):
    """Show project list with links to query interface"""
    st.title("Select a Project")
    
    try:
        with conn.cursor() as cur:
            # Updated to use your actual table name and columns
            cur.execute("SELECT projectid, title FROM project LIMIT 100")
            projects = cur.fetchall()
            
            if not projects:
                st.warning("No projects found in database")
                return
                
            for project_id, project_title in projects:
                # Create a button that sets the project_id in URL
                if st.button(
                    f"{project_title} (ID: {project_id})",
                    key=f"project_{project_id}"
                ):
                    st.query_params["project_id"] = project_id
                    st.rerun()
                    
    except psycopg2.Error as e:
        st.error(f"Failed to load projects: {str(e)}")

def show_project_query_interface(conn, project_id):
    """Show the query interface for a specific project"""
    # Initialize services
    query_interface = ProjectQueryInterface(conn, project_id)
    
    # First get project title for display
    project_title = "Unknown Project"
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT title FROM project WHERE projectid = %s", (project_id,))
            result = cur.fetchone()
            if result:
                project_title = result[0]
    except psycopg2.Error as e:
        st.error(f"Database error: {str(e)}")
    
    st.title(f"Project Analytics: {project_title}")
    st.markdown(f"**Project ID:** {project_id}")
    
    # Display schema information
    with st.expander("Database Schema for This Project"):
        st.json(query_interface.schema)
    
    # Add button to return to project selection
    if st.button("← Back to Project Selection"):
        st.query_params.clear()
        st.rerun()
    
    # Load model (required for generation)
    if st.checkbox("Load AI Model", key="load_model"):
        with st.spinner("Loading AI model..."):
            try:
                st.session_state.tokenizer = AutoTokenizer.from_pretrained(
                    "motherduckdb/DuckDB-NSQL-7B-v0.1"
                )
                st.session_state.model = AutoModelForCausalLM.from_pretrained(
                    "motherduckdb/DuckDB-NSQL-7B-v0.1",
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
                st.session_state.model_loaded = True
                st.success("Model loaded successfully!")
            except Exception as e:
                st.error(f"Model loading failed: {str(e)}")
                return
    
    # Only show query interface if model is loaded
    if st.session_state.get('model_loaded', False):
        question = st.text_area(
            f"Ask about project {project_title}:",
            placeholder=f"e.g. Show all tasks for this project with their status",
            height=100
        )
        
        if st.button("Generate Query"):
            if not question.strip():
                st.warning("Please enter a question")
            else:
                query = query_interface.generate_query(question)
                
                if query:
                    if not query.lower().strip().startswith('select'):
                        st.error("Generated query is not a SELECT statement")
                        st.code(query, language="sql")
                    else:
                        st.session_state.generated_query = query
                        st.subheader("Generated Query")
                        st.code(query, language="sql")
                
        if st.session_state.get('generated_query'):
            if st.button("Execute Query"):
                with st.spinner("Executing query..."):
                    results_df = execute_select_only(conn, st.session_state.generated_query)
                    st.session_state.results_df = results_df
            
            if st.session_state.get('results_df') is not None:
                st.subheader("Query Results")
                st.dataframe(st.session_state.results_df)
                
                if not st.session_state.results_df.empty:
                    show_visualizations(st.session_state.results_df)

if __name__ == "__main__":
    main()