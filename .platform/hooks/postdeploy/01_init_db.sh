#!/bin/bash
echo "Running SQLite DB init..."

# Run the Python DB init script
/var/app/venv/*/bin/python3 
/var/app/current/manage_init_db.py

echo "DB init complete."

