"""Tiny synthetic text-to-SQL set for bring-up (no 50 GB OmniSQL download).

Creates <db_root>/toy/<db_id>/<db_id>.sqlite for three small databases and
writes veRL parquet files with a handful of questions each.

    python -m rsched.data.make_toy_sql --db-root $RSCHED_DATA/sql_db --out $RSCHED_DATA/sql_toy
"""

from __future__ import annotations

import argparse
import os
import random
import sqlite3

import pandas as pd

from rsched.data.sql_prompt import build_row

DBS = {
    "company": {
        "ddl": [
            "CREATE TABLE departments (dept_id INTEGER PRIMARY KEY, name TEXT, city TEXT)",
            "CREATE TABLE employees (emp_id INTEGER PRIMARY KEY, name TEXT, dept_id INTEGER, salary INTEGER, "
            "hire_year INTEGER, FOREIGN KEY (dept_id) REFERENCES departments(dept_id))",
        ],
        "fill": lambda cur, rng: (
            cur.executemany(
                "INSERT INTO departments VALUES (?,?,?)",
                [(1, "Engineering", "Ithaca"), (2, "Sales", "NYC"), (3, "Research", "Ithaca"), (4, "HR", "Boston")],
            ),
            cur.executemany(
                "INSERT INTO employees VALUES (?,?,?,?,?)",
                [
                    (i, f"emp{i:03d}", rng.randint(1, 4), rng.randrange(50_000, 200_000, 1000), rng.randint(2010, 2026))
                    for i in range(1, 201)
                ],
            ),
        ),
        "qs": [
            ("How many employees work in the Engineering department?",
             "SELECT COUNT(*) FROM employees e JOIN departments d ON e.dept_id = d.dept_id WHERE d.name = 'Engineering'"),
            ("What is the average salary in each department? Return the department name and the average salary.",
             "SELECT d.name, AVG(e.salary) FROM employees e JOIN departments d ON e.dept_id = d.dept_id GROUP BY d.name"),
            ("List the names of employees hired in 2026 who earn more than 150000.",
             "SELECT name FROM employees WHERE hire_year = 2026 AND salary > 150000"),
            ("Which city has the most employees? Return the city name.",
             "SELECT d.city FROM employees e JOIN departments d ON e.dept_id = d.dept_id GROUP BY d.city "
             "ORDER BY COUNT(*) DESC LIMIT 1"),
            ("What is the highest salary among employees based in Ithaca?",
             "SELECT MAX(e.salary) FROM employees e JOIN departments d ON e.dept_id = d.dept_id WHERE d.city = 'Ithaca'"),
            ("How many employees were hired each year? Return the year and the count.",
             "SELECT hire_year, COUNT(*) FROM employees GROUP BY hire_year"),
        ],
    },
    "library": {
        "ddl": [
            "CREATE TABLE authors (author_id INTEGER PRIMARY KEY, name TEXT, country TEXT)",
            "CREATE TABLE books (book_id INTEGER PRIMARY KEY, title TEXT, author_id INTEGER, year INTEGER, pages INTEGER)",
            "CREATE TABLE loans (loan_id INTEGER PRIMARY KEY, book_id INTEGER, member TEXT, days INTEGER)",
        ],
        "fill": lambda cur, rng: (
            cur.executemany(
                "INSERT INTO authors VALUES (?,?,?)",
                [(i, f"author{i}", rng.choice(["US", "UK", "FR", "JP"])) for i in range(1, 21)],
            ),
            cur.executemany(
                "INSERT INTO books VALUES (?,?,?,?,?)",
                [(i, f"book{i}", rng.randint(1, 20), rng.randint(1950, 2025), rng.randint(80, 900)) for i in range(1, 151)],
            ),
            cur.executemany(
                "INSERT INTO loans VALUES (?,?,?,?)",
                [(i, rng.randint(1, 150), f"m{rng.randint(1, 40)}", rng.randint(1, 60)) for i in range(1, 501)],
            ),
        ),
        "qs": [
            ("How many books were written by authors from Japan?",
             "SELECT COUNT(*) FROM books b JOIN authors a ON b.author_id = a.author_id WHERE a.country = 'JP'"),
            ("What is the title of the most borrowed book?",
             "SELECT b.title FROM loans l JOIN books b ON l.book_id = b.book_id GROUP BY b.book_id "
             "ORDER BY COUNT(*) DESC, b.book_id LIMIT 1"),
            ("List the titles of books published before 1960.", "SELECT title FROM books WHERE year < 1960"),
            ("What is the average number of pages per author country? Return the country and average pages.",
             "SELECT a.country, AVG(b.pages) FROM books b JOIN authors a ON b.author_id = a.author_id GROUP BY a.country"),
            ("Which members have borrowed more than 15 books? Return member ids.",
             "SELECT member FROM loans GROUP BY member HAVING COUNT(*) > 15"),
        ],
    },
    "store": {
        "ddl": [
            "CREATE TABLE products (product_id INTEGER PRIMARY KEY, name TEXT, category TEXT, price REAL)",
            "CREATE TABLE orders (order_id INTEGER PRIMARY KEY, product_id INTEGER, quantity INTEGER, month INTEGER)",
        ],
        "fill": lambda cur, rng: (
            cur.executemany(
                "INSERT INTO products VALUES (?,?,?,?)",
                [(i, f"prod{i}", rng.choice(["toys", "books", "garden", "tools"]), round(rng.uniform(2, 90), 2))
                 for i in range(1, 61)],
            ),
            cur.executemany(
                "INSERT INTO orders VALUES (?,?,?,?)",
                [(i, rng.randint(1, 60), rng.randint(1, 5), rng.randint(1, 12)) for i in range(1, 801)],
            ),
        ),
        "qs": [
            ("What is the total revenue (price times quantity) for the 'garden' category?",
             "SELECT SUM(p.price * o.quantity) FROM orders o JOIN products p ON o.product_id = p.product_id "
             "WHERE p.category = 'garden'"),
            ("Which month had the largest number of units sold? Return the month number.",
             "SELECT month FROM orders GROUP BY month ORDER BY SUM(quantity) DESC LIMIT 1"),
            ("List the names of products that were never ordered.",
             "SELECT name FROM products WHERE product_id NOT IN (SELECT product_id FROM orders)"),
            ("How many distinct products were ordered in December?",
             "SELECT COUNT(DISTINCT product_id) FROM orders WHERE month = 12"),
            ("What is the most expensive product in each category? Return category and the maximum price.",
             "SELECT category, MAX(price) FROM products GROUP BY category"),
        ],
    },
}


def schema_text(conn: sqlite3.Connection) -> str:
    rows = conn.execute("SELECT sql FROM sqlite_master WHERE type='table'").fetchall()
    return "\n\n".join(r[0] + ";" for r in rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-frac", type=float, default=0.25)
    args = ap.parse_args()

    rng = random.Random(0)
    rows = []
    for db_id, spec in DBS.items():
        d = os.path.join(args.db_root, "toy", db_id)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{db_id}.sqlite")
        if os.path.exists(path):
            os.remove(path)
        conn = sqlite3.connect(path)
        cur = conn.cursor()
        for stmt in spec["ddl"]:
            cur.execute(stmt)
        spec["fill"](cur, rng)
        conn.commit()
        schema = schema_text(conn)
        for q, gold in spec["qs"]:
            conn.execute(gold).fetchall()  # gold must run
            rows.append((db_id, schema, q, gold))
        conn.close()

    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_frac))
    os.makedirs(args.out, exist_ok=True)
    for split, chunk in (("validation", rows[:n_val]), ("train", rows[n_val:])):
        recs = [
            build_row(source="toy", db_id=db, idx=i, schema=s, question=q, gold_sql=g, split=split)
            for i, (db, s, q, g) in enumerate(chunk)
        ]
        pd.DataFrame(recs).to_parquet(os.path.join(args.out, f"{split}.parquet"))
        print(f"{split}: {len(recs)} rows -> {args.out}/{split}.parquet")


if __name__ == "__main__":
    main()
