import yaml
import google.generativeai as genai
import mysql.connector
from tabulate import tabulate
import re
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory
import threading
import itertools
import time
import sys
import json


# Load config from YAML
def load_config():
    with open("config.yml", "r") as f:
        return yaml.safe_load(f)

def start_spinner(message="Working..."):
    done = threading.Event()

    def spinner():
        for char in itertools.cycle("|/-\\"):
            if done.is_set():
                break
            sys.stdout.write(f"\r{message} {char}")
            sys.stdout.flush()
            time.sleep(0.1)
        # Clear the spinner line completely and add newline
        sys.stdout.write("\r" + " " * (len(message) + 2) + "\r")
        sys.stdout.flush()

    thread = threading.Thread(target=spinner)
    thread.start()
    
    def stop_spinner():
        done.set()
        thread.join()  # Wait for spinner thread to finish
        # Ensure clean line after spinner
        print()  # This will add a proper newline
    
    # Return the stop function instead of done event
    stop_spinner.done = done
    return stop_spinner


def test_db_connection(db_conf):
    try:
        conn = mysql.connector.connect(
            host=db_conf["host"],
            user=db_conf["user"],
            password=db_conf["password"],
            database=db_conf["name"],
            port=db_conf.get("port", 3306),
        )
        print(f"✅ Connected to {db_conf['name']} on {db_conf['host']}:{db_conf.get('port', 3306)} as {db_conf['user']}")
    except mysql.connector.Error as e:
        print(f"❌ Failed to connect to database: {e}")
        exit(1)
    finally:
        conn.close()


# Extract all SQL code blocks from Gemini response
def extract_all_sql_blocks(text):
    blocks = re.findall(r"```sql\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return [b.strip() for b in blocks] if blocks else []


def extract_json_blocks(text):
    """Extract JSON blocks from response"""
    blocks = re.findall(r"```json\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return [b.strip() for b in blocks] if blocks else []


def run_sql(sql, db_conf):
    try:
        conn = mysql.connector.connect(
            host=db_conf["host"],
            user=db_conf["user"],
            password=db_conf["password"],
            database=db_conf["name"],
            port=db_conf.get("port", 3306),
        )
        cursor = conn.cursor()

        # Split multiple statements by ';' if exist, remove empty parts
        statements = [stmt.strip() for stmt in sql.split(';') if stmt.strip()]

        last_cols = None
        last_rows = None

        for stmt in statements:
            cursor.execute(stmt)
            if cursor.description:
                last_cols = [desc[0] for desc in cursor.description]
                last_rows = cursor.fetchall()
            else:
                conn.commit()
                last_cols = None
                last_rows = f"Affected rows: {cursor.rowcount}"

        return last_cols, last_rows

    except mysql.connector.Error as e:
        return None, f"MySQL Error: {e}"

    finally:
        cursor.close()
        conn.close()


def format_result_for_ai(cols, rows):
    """Format query results in a concise way for AI processing"""
    if not cols:
        return str(rows)
    
    # Limit to first 10 rows for AI analysis
    limited_rows = rows[:10] if isinstance(rows, list) else rows
    
    # Create a structured representation
    result_data = []
    if isinstance(limited_rows, list):
        for row in limited_rows:
            result_data.append(dict(zip(cols, row)))
    
    return {
        "columns": cols,
        "row_count": len(rows) if isinstance(rows, list) else "unknown",
        "sample_data": result_data,
        "preview": tabulate(limited_rows[:5] if isinstance(limited_rows, list) else limited_rows, 
                          headers=cols, tablefmt="grid") if cols else str(limited_rows)
    }


def create_multistep_prompt(user_request, execution_history=None):
    """Create prompt for multi-step operation planning"""
    history_context = ""
    if execution_history:
        history_context = "\nPrevious steps executed:\n"
        for i, step in enumerate(execution_history, 1):
            history_context += f"Step {i}: {step['description']}\n"
            history_context += f"SQL: {step['sql']}\n"
            if step.get('result'):
                history_context += f"Result summary: {step['result_summary']}\n"
            history_context += "\n"

    prompt = f"""
You are a MySQL database expert that can perform multi-step operations.

User request: "{user_request}"
{history_context}

Analyze this request and determine if it requires multiple steps. 

If it's a simple single-step operation, respond with:
```json
{{"type": "single", "sql": "YOUR_SQL_QUERY"}}
```

If it requires multiple steps, respond with:
```json
{{
    "type": "multistep",
    "plan": [
        {{"step": 1, "description": "What this step does", "sql": "FIRST_SQL_QUERY", "analysis_needed": true/false}},
        {{"step": 2, "description": "What this step does", "sql": "SECOND_SQL_QUERY or 'DEPENDS_ON_PREVIOUS'", "analysis_needed": true/false}}
    ],
    "total_steps": 2
}}
```

Use "DEPENDS_ON_PREVIOUS" for SQL that needs to be generated based on previous results.
Set "analysis_needed": true if the step results need to be analyzed before proceeding.

Examples of multi-step operations:
- Database optimization (analyze tables first, then create indexes)
- Performance analysis (check slow queries, then analyze specific tables)
- Data migration (check data integrity, then perform migration)
- Complex reporting (gather metrics, then generate summary)
"""
    return prompt


def create_dependent_step_prompt(step_info, previous_results, user_request):
    """Create prompt for generating SQL based on previous results"""
    prompt = f"""
You are a MySQL expert generating the next step in a multi-step operation.

Original user request: "{user_request}"
Current step: {step_info['description']}

Previous step results:
{json.dumps(previous_results, indent=2, default=str)}

Based on the previous results, generate the appropriate SQL query for this step.
Respond with ONLY the SQL query in triple backticks:

```sql
YOUR_SQL_QUERY_HERE
```
"""
    return prompt


def analyze_step_results(step_info, result_data, user_request, remaining_steps):
    """Create prompt for analyzing step results and potentially modifying the plan"""
    prompt = f"""
You are a MySQL expert analyzing the results of a database operation step.

Original user request: "{user_request}"
Current step: {step_info['description']}

Step results:
{json.dumps(result_data, indent=2, default=str)}

Remaining planned steps:
{json.dumps(remaining_steps, indent=2)}

Based on these results, should we:
1. Continue with the planned steps as-is
2. Modify the remaining steps
3. Add additional steps
4. Skip some steps

Respond in JSON format:
```json
{{
    "action": "continue|modify|add|skip",
    "explanation": "Why this action is recommended",
    "modified_steps": [/* only if action is modify or add */]
}}
```
"""
    return prompt


def execute_multistep_operation(plan, user_request, chat, db_conf):
    """Execute a multi-step operation plan"""
    execution_history = []
    current_plan = plan.copy()
    
    print(f"\n🚀 Starting multi-step operation with {len(current_plan)} planned steps")
    
    for step_idx, step_info in enumerate(current_plan):
        print(f"\n📋 Step {step_info['step']}: {step_info['description']}")
        
        # Generate SQL if it depends on previous results
        if step_info['sql'] == 'DEPENDS_ON_PREVIOUS':
            if not execution_history:
                print("❌ Cannot depend on previous results - no previous steps executed")
                continue
                
            print("🔄 Generating SQL based on previous results...")
            dependent_prompt = create_dependent_step_prompt(
                step_info, 
                execution_history[-1].get('result_data', {}), 
                user_request
            )
            
            spinner_done = start_spinner("🧠 Generating dependent SQL")
            response = chat.send_message(dependent_prompt)
            spinner_done()
            
            sql_blocks = extract_all_sql_blocks(response.text)
            if not sql_blocks:
                print("❌ Failed to generate SQL for dependent step")
                continue
            
            step_info['sql'] = sql_blocks[0]
        
        print(f"🔹 Executing SQL:\n{step_info['sql']}")
        
        # Execute the SQL
        cols, result = run_sql(step_info['sql'], db_conf)
        
        if isinstance(result, str) and result.startswith("MySQL Error"):
            print(f"❌ SQL error: {result}")
            # Try to fix the SQL
            fix_prompt = f"""
Fix this MySQL query that caused an error:

Query: {step_info['sql']}
Error: {result}

Provide ONLY the corrected SQL in triple backticks.
"""
            print("🔧 Trying to fix SQL...")
            fix_response = chat.send_message(fix_prompt)
            fixed_sql_blocks = extract_all_sql_blocks(fix_response.text)
            
            if fixed_sql_blocks:
                step_info['sql'] = fixed_sql_blocks[0]
                print(f"🛠  Retrying with fixed query:\n{step_info['sql']}")
                cols, result = run_sql(step_info['sql'], db_conf)
            
            if isinstance(result, str) and result.startswith("MySQL Error"):
                print(f"❌ Still failed after fix attempt: {result}")
                continue
        
        # Display results
        if cols:
            print(tabulate(result[:5], headers=cols, tablefmt="grid"))
            if len(result) > 5:
                print(f"... and {len(result) - 5} more rows")
        else:
            print(result)
        
        # Format results for AI processing
        result_data = format_result_for_ai(cols, result)
        
        # Store execution history
        execution_step = {
            'step': step_info['step'],
            'description': step_info['description'],
            'sql': step_info['sql'],
            'result_data': result_data,
            'result_summary': f"Returned {result_data.get('row_count', 'unknown')} rows" if cols else str(result)
        }
        execution_history.append(execution_step)
        
        # Analyze results if needed
        if step_info.get('analysis_needed', False) and step_idx < len(current_plan) - 1:
            print("🔍 Analyzing results for next steps...")
            
            remaining_steps = current_plan[step_idx + 1:]
            analysis_prompt = analyze_step_results(step_info, result_data, user_request, remaining_steps)
            
            spinner_done = start_spinner("🧠 Analyzing results")
            analysis_response = chat.send_message(analysis_prompt)
            spinner_done()
            
            json_blocks = extract_json_blocks(analysis_response.text)
            if json_blocks:
                try:
                    analysis = json.loads(json_blocks[0])
                    print(f"💡 Analysis: {analysis['explanation']}")
                    
                    if analysis['action'] == 'modify' and 'modified_steps' in analysis:
                        print("🔄 Modifying remaining steps based on analysis...")
                        # Replace remaining steps with modified ones
                        current_plan = current_plan[:step_idx + 1] + analysis['modified_steps']
                    elif analysis['action'] == 'skip':
                        print("⏭ Skipping remaining steps based on analysis")
                        break
                        
                except json.JSONDecodeError:
                    print("⚠ Could not parse analysis results, continuing with original plan")
    
    return execution_history


def main():
    config = load_config()
    genai.configure(api_key=config["google"]["api_key"])
    model = genai.GenerativeModel(config["google"].get("model", "gemini-1.5-flash"))

    # Persistent chat session for single run
    chat = model.start_chat()

    # DB check
    test_db_connection(config["database"])

    history = InMemoryHistory()
    conversation_history = []

    print("\nAI MySQL Multi-Step Agent (type 'exit' to quit)")
    print("Supports both simple queries and complex multi-step operations!")

    while True:
        try:
            user_input = prompt("Enter your request > ", history=history).strip()
            if not user_input:
                continue
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if user_input.lower() in ["exit", "quit"]:
            print("Bye!")
            break

        # Determine if this is a single or multi-step operation
        planning_prompt = create_multistep_prompt(user_input)
        
        stop_spinner = start_spinner("🧠 Analyzing request")
        response = chat.send_message(planning_prompt)
        stop_spinner()
        
        # Try to extract JSON plan
        json_blocks = extract_json_blocks(response.text)
        
        if json_blocks:
            try:
                plan_data = json.loads(json_blocks[0])
                
                if plan_data["type"] == "single":
                    # Handle single-step operation (original behavior)
                    print("\n🔹 Single-step operation")
                    sql_query = plan_data["sql"]
                    print(f"SQL: {sql_query}")
                    
                    cols, result = run_sql(sql_query, config["database"])
                    
                    if isinstance(result, str) and result.startswith("MySQL Error"):
                        print(f"❌ SQL error: {result}")
                        continue
                    
                    if cols:
                        print(tabulate(result[:10], headers=cols, tablefmt="grid"))
                        if len(result) > 10:
                            print(f"... and {len(result) - 10} more rows")
                    else:
                        print(result)
                    
                    # Get explanation
                    explain_prompt = f"""
                    Explain the results of this SQL query in Czech:
                    
                    User request: {user_input}
                    SQL: {sql_query}
                    Result preview: {tabulate(result[:3], headers=cols, tablefmt="grid") if cols else str(result)}
                    """
                    
                    spinner_done = start_spinner("📖 Explaining results")
                    explain_response = chat.send_message(explain_prompt)
                    spinner_done()
                    
                    print(f"\n💬 Vysvětlení:\n{explain_response.text.strip()}")
                    print()  # Add extra newline for better spacing
                    
                else:
                    # Handle multi-step operation
                    execution_history = execute_multistep_operation(
                        plan_data["plan"], user_input, chat, config["database"]
                    )
                    
                    # Provide final summary
                    if execution_history:
                        summary_prompt = f"""
                        Provide a final summary in Czech of this multi-step database operation:
                        
                        Original request: {user_input}
                        
                        Steps executed:
                        {json.dumps(execution_history, indent=2, default=str)}
                        
                        Summarize what was accomplished and any key findings.
                        """
                        
                        spinner_done = start_spinner("📋 Generating summary")
                        summary_response = chat.send_message(summary_prompt)
                        spinner_done()
                        
                        print(f"\n📋 Celkové shrnutí:\n{summary_response.text.strip()}")
                
                conversation_history.append({"user": user_input, "type": plan_data["type"]})
                print()  # Add extra newline for better spacing
                
            except json.JSONDecodeError:
                print("❌ Could not parse operation plan, trying simple SQL generation...")
                # Fallback to original single-query behavior
                sql_blocks = extract_all_sql_blocks(response.text)
                if sql_blocks:
                    for sql_query in sql_blocks:
                        print(f"\n🔹 Executing SQL:\n{sql_query}")
                        cols, result = run_sql(sql_query, config["database"])
                        
                        if isinstance(result, str) and result.startswith("MySQL Error"):
                            print(f"❌ SQL error: {result}")
                            continue
                        
                        if cols:
                            print(tabulate(result[:5], headers=cols, tablefmt="grid"))
                        else:
                            print(result)
                        print()  # Add extra newline for better spacing
        else:
            print("❌ Could not understand the request format")
            print()  # Add extra newline for better spacing


if __name__ == "__main__":
    main()
    