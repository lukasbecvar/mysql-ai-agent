import yaml
import google.generativeai as genai
import mysql.connector
from tabulate import tabulate
import re

# Load config from YAML
def load_config():
    with open("config.yml", "r") as f:
        return yaml.safe_load(f)

# Extract SQL from LLM response (in triple backticks)
def extract_sql(text):
    m = re.search(r"```sql\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()

# Run SQL query and return columns+rows or error string
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
        cursor.execute(sql)
        if cursor.description:
            cols = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            return cols, rows
        else:
            conn.commit()
            return None, f"Affected rows: {cursor.rowcount}"
    except mysql.connector.Error as e:
        return None, f"MySQL Error: {e}"
    finally:
        cursor.close()
        conn.close()

def prompt_for_sql_with_history(user_prompt, history):
    history_text = ""
    for i, entry in enumerate(history[-10:]):  # keep last 10 for context
        history_text += f"User: {entry['user']}\nSQL: {entry['sql']}\n\n"
    return f'''
You are a professional MySQL database assistant.
You have the following conversation history with the user:
{history_text}
Now, convert the user's new request into a safe, syntactically correct MySQL query.
Respond with ONLY the SQL query inside triple backticks, no explanations.

Request:
""" {user_prompt} """
'''

# Build prompt for fixing SQL on error
def prompt_fix_sql(sql, error, history):
    history_text = "\n".join(
        f"Attempt {i+1} SQL:\n{h['sql']}\nError:\n{h['error']}" for i, h in enumerate(history)
    )
    return f"""
You are a professional MySQL assistant helping to fix errors in SQL statements.
The user gave this SQL query, which caused an error:

{sql}

Error message:
{error}

Past attempts and errors:
{history_text}

Please provide a corrected SQL query wrapped in triple backticks, no explanations.
"""

# Build prompt for result explanation
def prompt_explain(sql, cols, rows, user_lang):
    preview = tabulate(rows[:5], headers=cols, tablefmt="grid") if cols else str(rows)
    return f"""
You are a helpful assistant that explains SQL query results in the user's language ({user_lang}).

The SQL query executed was:

```sql
{sql}
```

Here is a preview of the result (max 5 rows):

{preview}

Please provide a short, clear summary of what this result means.
"""

def main():
    config = load_config()
    genai.configure(api_key=config["google"]["api_key"])
    model = genai.GenerativeModel(config["google"].get("model", "gemini-1.5-flash"))

    conversation_history = []
    error_history = []

    print("AI MySQL Natural Language Agent (type 'exit' to quit)")

    while True:
        user_input = input("Enter your request > ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Bye!")
            break

        # Generate SQL query with context
        prompt_sql = prompt_for_sql_with_history(user_input, conversation_history)
        response = model.generate_content(prompt_sql)
        sql_query = extract_sql(response.text)

        print(f"""Generated SQL: 
{sql_query} 
""")
        attempt = 0
        while attempt < 5:
            cols, result = run_sql(sql_query, config["database"])

            if isinstance(result, str) and result.startswith("MySQL Error"):
                print(f"SQL error: {result}")

                # Check repeated errors
                if any(h["sql"] == sql_query and h["error"] == result for h in error_history):
                    print("Repeated error detected, stopping retries.")
                    break

                error_history.append({"sql": sql_query, "error": result})
                fix_prompt = prompt_fix_sql(sql_query, result, error_history)
                fix_response = model.generate_content(fix_prompt)
                fixed_sql = extract_sql(fix_response.text)

                print(f"""Trying to fix SQL, new query:
{fixed_sql}
""")

                sql_query = fixed_sql
                attempt += 1
                continue
            else:
                if cols:
                    print(tabulate(result[:5], headers=cols, tablefmt="grid"))
                else:
                    print(result)

                user_lang = "English"

                explain_prompt = prompt_explain(sql_query, cols, result, user_lang)
                explain_response = model.generate_content(explain_prompt)
                print(f"""Explanation:
{explain_response.text.strip()}""")

                # Add to conversation history for context next round
                conversation_history.append({"user": user_input, "sql": sql_query})

                break

if __name__ == "__main__":
    main()
