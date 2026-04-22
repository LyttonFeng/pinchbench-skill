"""task_18_spreadsheet_summary variant generator.

Generates synthetic quarterly_sales.csv and company_expenses.xlsx files
with varied but consistent data so the agent must actually parse the files.

Requires: openpyxl (pip install openpyxl)
"""
from __future__ import annotations
import csv
import io
import random
import zipfile
from collections import defaultdict
from html import escape
from typing import Iterator
from . import BaseVariantGenerator, TaskVariant

_REGIONS = ["North", "South", "East", "West"]
_PRODUCTS = ["Widget A", "Widget B", "Widget C"]
_DEPARTMENTS = ["Engineering", "Marketing", "Sales", "Operations"]

# (first_name, last_name)
_EMPLOYEES = [
    ("Alice", "Chen"), ("Bob", "Martinez"), ("Carol", "Johnson"),
    ("David", "Kim"), ("Eve", "Thompson"), ("Frank", "Garcia"),
    ("Grace", "Lee"), ("Henry", "Wilson"), ("Iris", "Brown"),
    ("Jack", "Davis"), ("Kate", "Miller"), ("Liam", "Taylor"),
]


def _col_name(idx: int) -> str:
    name = ""
    while idx:
        idx, rem = divmod(idx - 1, 26)
        name = chr(65 + rem) + name
    return name


def _cell_xml(row_idx: int, col_idx: int, value: str | int | float) -> str:
    ref = f"{_col_name(col_idx)}{row_idx}"
    if isinstance(value, (int, float)):
        return f'<c r="{ref}"><v>{value}</v></c>'
    return f'<c r="{ref}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>'


def _sheet_xml(rows: list[list[str | int | float]]) -> str:
    row_xml = []
    for r_idx, row in enumerate(rows, start=1):
        cells = "".join(_cell_xml(r_idx, c_idx, value) for c_idx, value in enumerate(row, start=1))
        row_xml.append(f'<row r="{r_idx}">{cells}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(row_xml)}</sheetData>'
        '</worksheet>'
    )


def _build_xlsx_minimal(sheets: dict[str, list[list[str | int | float]]]) -> bytes:
    """Create a minimal XLSX workbook using only the Python standard library."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            + "".join(
                f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                for i in range(1, len(sheets) + 1)
            ) +
            '</Types>'
        ))
        zf.writestr("_rels/.rels", (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            '</Relationships>'
        ))
        workbook_sheets = []
        workbook_rels = []
        for i, name in enumerate(sheets, start=1):
            workbook_sheets.append(f'<sheet name="{escape(name)}" sheetId="{i}" r:id="rId{i}"/>')
            workbook_rels.append(
                f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>'
            )
            zf.writestr(f"xl/worksheets/sheet{i}.xml", _sheet_xml(sheets[name]))
        zf.writestr("xl/workbook.xml", (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            f'<sheets>{"".join(workbook_sheets)}</sheets>'
            '</workbook>'
        ))
        zf.writestr("xl/_rels/workbook.xml.rels", (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f'{"".join(workbook_rels)}'
            '</Relationships>'
        ))
    return buf.getvalue()


def _generate_csv(rng: random.Random) -> tuple[str, list[dict]]:
    """Generate quarterly_sales.csv with 24 rows."""
    rows = []
    records = []
    months = ["2024-01", "2024-02", "2024-03"]
    for month in months:
        for region in _REGIONS:
            for product in _PRODUCTS:
                day = rng.randint(1, 28)
                date = f"{month}-{day:02d}"
                units = rng.randint(50, 300)
                unit_price = rng.choice([10, 15, 20, 25, 30])
                revenue = units * unit_price
                cost_pct = rng.uniform(0.3, 0.6)
                cost = round(revenue * cost_pct, 2)
                rows.append([date, region, product, units, unit_price, revenue, cost])
                records.append({
                    "date": date,
                    "region": region,
                    "product": product,
                    "units_sold": units,
                    "unit_price": unit_price,
                    "revenue": revenue,
                    "cost": cost,
                })

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Date", "Region", "Product", "Units_Sold", "Unit_Price", "Revenue", "Cost"])
    writer.writerows(rows)
    return buf.getvalue(), records


def _generate_xlsx(rng: random.Random) -> tuple[bytes, dict]:
    """Generate company_expenses.xlsx with Q1_Expenses and Budgets sheets."""
    employees_sample = rng.sample(_EMPLOYEES, 12)
    dept_assignments = {emp: rng.choice(_DEPARTMENTS) for emp in employees_sample}

    expense_categories = ["Travel", "Software", "Hardware", "Training", "Meals"]
    expense_records = []
    expenses_rows: list[list[str | int | float]] = [["Employee", "Department", "Category", "Amount", "Date"]]
    for emp in employees_sample:
        first, last = emp
        dept = dept_assignments[emp]
        category = rng.choice(expense_categories)
        amount = rng.randint(200, 1500)
        month = rng.randint(1, 3)
        day = rng.randint(1, 28)
        date_str = f"2024-0{month}-{day:02d}"
        employee = f"{first} {last}"
        expenses_rows.append([employee, dept, category, amount, date_str])
        expense_records.append({
            "employee": employee,
            "department": dept,
            "category": category,
            "amount": amount,
            "date": date_str,
        })

    budgets = {}
    budget_rows: list[list[str | int | float]] = [["Department", "Q1_Budget", "Q2_Budget", "Q3_Budget", "Q4_Budget"]]
    for dept in _DEPARTMENTS:
        q1 = rng.randint(5000, 15000)
        q2 = rng.randint(5000, 15000)
        q3 = rng.randint(5000, 15000)
        q4 = rng.randint(5000, 15000)
        budget_rows.append([dept, q1, q2, q3, q4])
        budgets[dept] = {
            "q1_budget": q1,
            "q2_budget": q2,
            "q3_budget": q3,
            "q4_budget": q4,
        }

    try:
        import openpyxl
    except ImportError:
        xlsx = _build_xlsx_minimal({"Q1_Expenses": expenses_rows, "Budgets": budget_rows})
    else:
        wb = openpyxl.Workbook()
        ws1 = wb.active
        ws1.title = "Q1_Expenses"
        for row in expenses_rows:
            ws1.append(row)
        ws2 = wb.create_sheet("Budgets")
        for row in budget_rows:
            ws2.append(row)
        buf = io.BytesIO()
        wb.save(buf)
        xlsx = buf.getvalue()
    return xlsx, {"expenses": expense_records, "budgets": budgets}


def _top_key(values: dict[str, float]) -> str:
    """Deterministic max helper for stable expected answers."""
    return sorted(values.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _expected(csv_records: list[dict], workbook: dict) -> dict:
    region_revenue: dict[str, float] = defaultdict(float)
    product_revenue: dict[str, float] = defaultdict(float)
    total_revenue = 0.0
    total_cost = 0.0
    total_units = 0
    for row in csv_records:
        revenue = float(row["revenue"])
        cost = float(row["cost"])
        total_revenue += revenue
        total_cost += cost
        total_units += int(row["units_sold"])
        region_revenue[row["region"]] += revenue
        product_revenue[row["product"]] += revenue

    dept_expenses: dict[str, float] = defaultdict(float)
    employee_expenses: dict[str, float] = defaultdict(float)
    total_q1_expenses = 0.0
    for row in workbook["expenses"]:
        amount = float(row["amount"])
        total_q1_expenses += amount
        dept_expenses[row["department"]] += amount
        employee_expenses[row["employee"]] += amount

    budget_comparison = {}
    for dept in _DEPARTMENTS:
        actual = dept_expenses.get(dept, 0.0)
        budget = float(workbook["budgets"][dept]["q1_budget"])
        variance = budget - actual
        budget_comparison[dept] = {
            "actual_expenses": round(actual, 2),
            "q1_budget": round(budget, 2),
            "variance": round(variance, 2),
            "status": "under_budget" if variance >= 0 else "over_budget",
        }

    return {
        "target_file": "data_summary.md",
        "csv": {
            "total_revenue": round(total_revenue, 2),
            "total_profit": round(total_revenue - total_cost, 2),
            "total_units_sold": total_units,
            "top_region_by_revenue": _top_key(region_revenue),
            "top_region_revenue": round(region_revenue[_top_key(region_revenue)], 2),
            "top_product_by_revenue": _top_key(product_revenue),
            "top_product_revenue": round(product_revenue[_top_key(product_revenue)], 2),
        },
        "excel": {
            "total_q1_expenses": round(total_q1_expenses, 2),
            "department_highest_expenses": _top_key(dept_expenses),
            "department_highest_expenses_amount": round(dept_expenses[_top_key(dept_expenses)], 2),
            "employee_highest_expenses": _top_key(employee_expenses),
            "employee_highest_expenses_amount": round(employee_expenses[_top_key(employee_expenses)], 2),
            "budget_comparison": budget_comparison,
        },
    }


def _prompt_for_variant(variant_seed: int) -> tuple[str, str]:
    """Return a prompt variant without changing data-generation randomness."""
    variants = [
        (
            "canonical",
            (
                "I have two data files in my workspace that have been provided for you to analyze:\n\n"
                "1. `quarterly_sales.csv` — a CSV file with sales transactions containing columns: "
                "Date, Region, Product, Units_Sold, Unit_Price, Revenue, Cost (24 rows of data)\n"
                "2. `company_expenses.xlsx` — an Excel workbook with two sheets: "
                "\"Q1_Expenses\" (employee expense reports with 12 records) and "
                "\"Budgets\" (departmental budget allocations)\n\n"
                "Please read and analyze both files, then write a summary report to `data_summary.md` that includes:\n\n"
                "- **CSV Analysis**: Total revenue, total profit (revenue minus cost), total units sold, "
                "the top-performing region by revenue, and the top-selling product by revenue.\n"
                "- **Excel Analysis**: Total Q1 expenses, the department with the highest expenses, "
                "the employee with the highest total expenses, and a comparison of Q1 actual expenses "
                "vs Q1 budgets by department.\n"
                "- A brief overall insights section combining findings from both files."
            ),
        ),
        (
            "manager_report",
            (
                "Please prepare a short business report from the files in this workspace. "
                "Use `quarterly_sales.csv` for sales performance and `company_expenses.xlsx` for Q1 expenses and budgets.\n\n"
                "Create the final deliverable as `data_summary.md`. The report must include total revenue, profit, "
                "units sold, the leading region by revenue, the leading product by revenue, total Q1 expenses, "
                "the highest-spend department, the highest-spend employee, and actual-vs-budget by department. "
                "Do the calculations from the files rather than guessing."
            ),
        ),
        (
            "ops_finance_brief",
            (
                "I need an operations/finance brief in markdown. There are two source files:\n\n"
                "- `quarterly_sales.csv`: sales rows with Date, Region, Product, Units_Sold, Unit_Price, Revenue, Cost\n"
                "- `company_expenses.xlsx`: workbook containing `Q1_Expenses` and `Budgets`\n\n"
                "Analyze both sources and write `data_summary.md`. Include sales totals, profit, units, top region, "
                "top product, total Q1 expenses, highest spending department, highest spending employee, "
                "and each department's Q1 actual expense compared with its Q1 budget."
            ),
        ),
        (
            "file_delivery",
            (
                "Read the spreadsheet data in this directory and produce the requested output file. "
                "The output file must be named exactly `data_summary.md`.\n\n"
                "Use `quarterly_sales.csv` to calculate revenue, profit, units sold, top region, and top product. "
                "Use `company_expenses.xlsx` to calculate Q1 expense totals, highest department spend, highest employee spend, "
                "and department-level budget variance. Format the result as a clear markdown summary with a brief insight section."
            ),
        ),
        (
            "analyst_task",
            (
                "Act as a data analyst. Combine the CSV sales file and the Excel expense workbook into one markdown summary.\n\n"
                "Required output: `data_summary.md`.\n"
                "Required sales metrics: total revenue, total profit, total units sold, best region by revenue, best product by revenue.\n"
                "Required expense metrics: total Q1 expenses, top spending department, top spending employee, and Q1 actual vs budget by department.\n"
                "Make sure the numbers are computed from the files."
            ),
        ),
    ]
    return variants[variant_seed % len(variants)]


class Generator(BaseVariantGenerator):
    task_id = "task_18_spreadsheet_summary"

    def sample(self, n: int = 1, seed: int | None = None) -> Iterator[TaskVariant]:
        rng = random.Random(seed)
        for _ in range(n):
            # Use a per-variant seed so data is deterministic per call
            variant_seed = rng.randint(0, 2**31)
            vrng = random.Random(variant_seed)

            csv_data, csv_records = _generate_csv(vrng)

            # Reset vrng for xlsx so it's independent from csv generation
            vrng2 = random.Random(variant_seed + 1)
            xlsx_data, workbook = _generate_xlsx(vrng2)
            expected = _expected(csv_records, workbook)
            prompt_variant, prompt = _prompt_for_variant(variant_seed)

            yield TaskVariant(
                task_id=self.task_id,
                prompt=prompt,
                workspace_files={
                    "quarterly_sales.csv": csv_data,
                    "company_expenses.xlsx": xlsx_data,
                },
                metadata={
                    "variant_seed": variant_seed,
                    "source": "synthetic_spreadsheet_v1",
                    "prompt_variant": prompt_variant,
                    "expected_target_file": expected["target_file"],
                },
                expected=expected,
            )
