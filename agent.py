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
        sys.stdout.write("\r" + " " * (len(message) + 2) + "\r")

    thread = threading.Thread(target=spinner)
    thread.start()
    return done


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
    return [b.strip() for b in blocks] if blocks else [text.strip()]


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
def prompt_explain(sql, cols, rows, user_prompt):
    preview = tabulate(rows[:5], headers=cols, tablefmt="grid") if cols else str(rows)
    return f"""
You are a helpful assistant that explains SQL query results.

The user's original request was:
\"\"\"{user_prompt}\"\"\"

The SQL query executed was:

```sql
{sql}

```

Here is a preview of the result (max 5 rows):

{preview}

Please provide a short, clear summary of what this result means in user's language.
"""

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
    error_history = []

    print("\nAI MySQL Natural Language Agent (type 'exit' to quit)")

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

        # Instruction to generate possibly multiple SQL blocks
        prompt_sql = f"""
You are a MySQL expert. Convert the following user request into safe, valid MySQL query or multiple queries (if needed).
Respond ONLY with SQL blocks in triple backticks. Do NOT add explanations or comments outside the code.

User request:
\"""{user_input}\"""
"""

        spinner_done = start_spinner("🧠 Creating SQL query")
        response = chat.send_message(prompt_sql)
        spinner_done.set()
        sql_blocks = extract_all_sql_blocks(response.text)

        for i, sql_query in enumerate(sql_blocks, 1):
            print(f"""\n🔹 SQL Block {i}:
{sql_query}
""")

            attempt = 0
            while attempt < 5:
                cols, result = run_sql(sql_query, config["database"])

                if isinstance(result, str) and result.startswith("MySQL Error"):
                    print(f"❌ SQL error: {result}")

                    # Avoid retrying same known-bad SQL+error combo
                    if any(h["sql"] == sql_query and h["error"] == result for h in error_history):
                        print("⚠️ Repeated error detected, skipping to next block.")
                        break

                    error_history.append({"sql": sql_query, "error": result})

                    fix_prompt = prompt_fix_sql(sql_query, result, error_history)
                    print("🔧 Trying to fix SQL...")
                    fix_response = chat.send_message(fix_prompt)
                    fixed_sql_blocks = extract_all_sql_blocks(fix_response.text)
                    fixed_sql = fixed_sql_blocks[0] if fixed_sql_blocks else ""

                    print(f"""🛠️  New fixed query:
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

                    # Explanation
                    explain_prompt = prompt_explain(sql_query, cols, result, user_input)
                    spinner_done = start_spinner("📖 Explaining results")
                    explain_response = chat.send_message(explain_prompt)
                    spinner_done.set()

                    print(f"""\nExplanation:
{explain_response.text.strip()}""")

                    conversation_history.append({"user": user_input, "sql": sql_query})
                    break


if __name__ == "__main__":
    main()
