import psycopg2
from psycopg2 import sql


# --- Database Connection and Operations ---

def connect():
    """Establishes a connection to the PostgreSQL database."""
    conn = None
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT
        )
        print("Connection to Supabase PostgreSQL successful.")
        return conn
    except psycopg2.OperationalError as e:
        print(f"Error connecting to the database: {e}")
        return None

def insert_rule(conn, session_id, expert_name, expertise_area, rule_text):
    """Inserts a new rule into the interview_rules table."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO interview_rules (session_id, expert_name, expertise_area, rule_text)
                VALUES (%s, %s, %s, %s)
                RETURNING id;
                """,
                (session_id, expert_name, expertise_area, rule_text)
            )
            rule_id = cur.fetchone()[0]
            conn.commit()
            print(f"Successfully inserted rule with ID: {rule_id}")
            return rule_id
    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error inserting data: {error}")
        if conn:
            conn.rollback()

def get_rule_by_session_id(conn, session_id):
    """Retrieves a rule by its session_id."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT * FROM interview_rules WHERE session_id = %s;
                """,
                (session_id,)
            )
            row = cur.fetchone()
            if row:
                print(f"Found rule for session ID '{session_id}': {row}")
                return row
            else:
                print(f"No rule found for session ID '{session_id}'.")
                return None
    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error retrieving data: {error}")
        return None

def update_rule_status(conn, session_id, completed):
    """Updates the completed status of a rule."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE interview_rules SET completed = %s WHERE session_id = %s;
                """,
                (completed, session_id)
            )
            conn.commit()
            if cur.rowcount > 0:
                print(f"Successfully updated completed status for session ID '{session_id}' to {completed}.")
            else:
                print(f"No rule found with session ID '{session_id}' to update.")
    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error updating data: {error}")
        if conn:
            conn.rollback()

def delete_rule(conn, session_id):
    """Deletes a rule by its session_id."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM interview_rules WHERE session_id = %s;
                """,
                (session_id,)
            )
            conn.commit()
            if cur.rowcount > 0:
                print(f"Successfully deleted rule with session ID '{session_id}'.")
            else:
                print(f"No rule found with session ID '{session_id}' to delete.")
    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error deleting data: {error}")
        if conn:
            conn.rollback()

# --- Main Execution Block ---

if __name__ == "__main__":
    # --- Step 1: Connect to the database ---
    conn = connect()

    if conn:
        try:
            # --- Step 2: Insert a new rule ---
            print("\n--- Inserting a new rule ---")
            my_session_id = "test_session_123"
            insert_rule(
                conn,
                session_id=my_session_id,
                expert_name="John Doe",
                expertise_area="Database Systems",
                rule_text="The expert must verify all connections."
            )

            # --- Step 3: Read the newly created rule ---
            print("\n--- Retrieving the inserted rule ---")
            retrieved_rule = get_rule_by_session_id(conn, my_session_id)

            # --- Step 4: Update the rule's status ---
            if retrieved_rule:
                print("\n--- Updating the rule's status to completed ---")
                update_rule_status(conn, my_session_id, True)
                get_rule_by_session_id(conn, my_session_id)

            # --- Step 5: Delete the rule ---
            print("\n--- Deleting the rule ---")
            delete_rule(conn, my_session_id)
            
            # --- Verify deletion ---
            print("\n--- Verifying deletion ---")
            get_rule_by_session_id(conn, my_session_id)

        finally:
            if conn:
                conn.close()
                print("\nDatabase connection closed.")
