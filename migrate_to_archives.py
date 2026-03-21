#!/usr/bin/env python3
"""
Migration Helper Script:
1. Connects to your existing PostgreSQL database.
2. Ensures the schema matches the new multi-archive requirements (adds missing columns).
3. Generates the YAML configuration snippet for your existing database.
"""

import argparse
import sys
import os

try:
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
except ImportError:
    print("Error: psycopg2 is required. Install it with: pip install psycopg2-binary")
    sys.exit(1)

def update_schema(conn_str):
    """Add missing columns to the existing photos table if they don't exist."""
    print(f"Connecting to database to verify/update schema...")
    
    try:
        conn = psycopg2.connect(conn_str)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        
        # List of new columns added for the multi-archive/LLM features
        columns_to_add = {
            "is_black_and_white": "BOOLEAN DEFAULT FALSE",
            "caption_model_version": "VARCHAR(50)",
            "embedding_model_version": "VARCHAR(50)"
        }
        
        # Check table existence first
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'photos'
            );
        """)
        if not cur.fetchone()[0]:
            print("Warning: 'photos' table not found. This might be a fresh database.")
            return True

        for col_name, col_type in columns_to_add.items():
            # Check if column exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.columns 
                    WHERE table_name = 'photos' AND column_name = %s
                );
            """, (col_name,))
            
            exists = cur.fetchone()[0]
            if not exists:
                print(f"  - Adding missing column: {col_name}")
                cur.execute(f"ALTER TABLE photos ADD COLUMN IF NOT EXISTS {col_name} {col_type};")
            else:
                print(f"  - Column {col_name} already exists.")
        
        cur.close()
        conn.close()
        print("Schema verification/update complete.")
        return True
        
    except Exception as e:
        print(f"Error updating schema: {e}")
        return False

def generate_yaml_snippet(db_name, user, host, port, root_dir, archive_name="Existing Production"):
    """Generate the YAML snippet for the config file."""
    
    # Mask password if present in a full URI, but here we assume components
    yaml_content = f"""
  - name: "{archive_name}"
    id: "prod_existing"
    db_path: "postgresql://{user}:YOUR_PASSWORD@{host}:{port}/{db_name}"
    root_dir: "{root_dir}"
    description: "Migrated existing PostgreSQL database. Update password in YAML."
    embedding_model: "all-MiniLM-L6-v2"
    llm_model: "local-llama3"
"""
    return yaml_content

def parse_connection_string(conn_str):
    """Simple parser to extract components from a postgresql:// URL."""
    # Format: postgresql://user:password@host:port/dbname
    if not conn_str.startswith("postgresql://"):
        return None
    
    body = conn_str.replace("postgresql://", "")
    
    # Split auth and rest
    if "@" in body:
        auth, rest = body.split("@", 1)
        if ":" in auth:
            user, password = auth.split(":", 1)
        else:
            user, password = auth, ""
    else:
        rest = body
        user = "postgres" # default
        password = ""

    # Split host/port/db
    if "/" in rest:
        host_port, db_name = rest.split("/", 1)
    else:
        host_port = rest
        db_name = "postgres"

    if ":" in host_port:
        host, port = host_port.split(":", 1)
    else:
        host = host_port
        port = "5432"
        
    return {
        "db_name": db_name,
        "user": user,
        "password": password,
        "host": host,
        "port": port
    }

def main():
    parser = argparse.ArgumentParser(description="Migrate existing Postgres DB to Archive Config")
    parser.add_argument("--db", type=str, help="Existing PostgreSQL connection string (postgresql://user:pass@host:port/db)")
    parser.add_argument("--root", type=str, required=True, help="Root directory path for this archive (e.g., ~/Documents/photos1)")
    parser.add_argument("--name", type=str, default="Existing Production", help="Friendly name for this archive")
    
    args = parser.parse_args()

    if not args.db:
        print("\n=== Manual Configuration Mode ===")
        print("Please provide your connection string with --db to auto-generate config.")
        print("Example: --db 'postgresql://myuser:mypass@localhost:5432/myphotodb'")
        return

    # Parse connection string
    creds = parse_connection_string(args.db)
    if not creds:
        print("Invalid connection string format. Must start with postgresql://")
        return

    # 1. Update Schema
    if not update_schema(args.db):
        print("Failed to update schema. Please check connection.")
        return

    # 2. Generate YAML
    print("\n=== Generated Configuration ===")
    print("Copy the following block into your 'archives_config.yaml' under the 'archives:' list:")
    print("-" * 40)
    
    yaml_block = generate_yaml_snippet(
        creds['db_name'], 
        creds['user'], 
        creds['host'], 
        creds['port'], 
        os.path.abspath(os.path.expanduser(args.root)),
        args.name
    )
    print(yaml_block)
    print("-" * 40)
    print("\nIMPORTANT: Replace 'YOUR_PASSWORD' in the generated YAML with your actual database password.")

if __name__ == "__main__":
    main()
