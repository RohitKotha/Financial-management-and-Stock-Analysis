from datetime import datetime, timedelta
import random

from core.database import Database

USERNAME = "testuser"
PASSWORD = "Test@123"

random.seed(42)


def main():
    db = Database()

    created, msg = db.create_user(USERNAME, PASSWORD)
    if not created and "already exists" not in msg.lower():
        raise RuntimeError(msg)

    ok, auth_msg, user = db.authenticate_user(USERNAME, PASSWORD)
    if not ok:
        raise RuntimeError(auth_msg)

    user_id = user["id"]
    db.set_current_user(user_id)

    conn = db.get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM transactions WHERE user_id = ?", (user_id,))
    cur.execute("DELETE FROM budgets WHERE user_id = ?", (user_id,))
    cur.execute("DELETE FROM portfolio WHERE user_id = ?", (user_id,))
    cur.execute("DELETE FROM financial_goals WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()

    expense_templates = {
        "Food & Dining": ["Grocery Store", "Restaurant Dinner", "Coffee Shop", "Lunch Meal"],
        "Transportation": ["Fuel Station", "Cab Ride", "Bus Pass", "Parking Fee"],
        "Shopping": ["Online Shopping", "Clothing Purchase", "Electronics Accessory", "Home Item"],
        "Entertainment": ["Movie Ticket", "Streaming Subscription", "Game Purchase", "Concert Pass"],
        "Bills & Utilities": ["Electricity Bill", "Internet Bill", "Water Bill", "Mobile Recharge"],
        "Healthcare": ["Pharmacy", "Doctor Visit", "Lab Test", "Health Checkup"],
        "Education": ["Course Subscription", "Book Purchase", "Training Fee", "Learning App"],
    }

    income_templates = [
        ("Salary", "Salary", 65000),
        ("Freelance", "Freelance", 12000),
    ]

    start_date = datetime.now().date() - timedelta(days=95)
    num_days = 96

    expense_count = 0
    income_count = 0

    for i in range(num_days):
        day = start_date + timedelta(days=i)
        day_str = day.strftime("%Y-%m-%d")

        daily_expenses = 2 if i % 3 == 0 else 1
        categories = list(expense_templates.keys())
        random.shuffle(categories)

        for j in range(daily_expenses):
            category = categories[j]
            description = random.choice(expense_templates[category])
            amount = round(random.uniform(120, 2400), 2)
            db.add_transaction(day_str, description, amount, category, "expense")
            expense_count += 1

        if i % 30 == 0:
            db.add_transaction(day_str, income_templates[0][0], income_templates[0][2], income_templates[0][1], "income")
            income_count += 1
        if i % 14 == 0:
            db.add_transaction(day_str, income_templates[1][0], income_templates[1][2], income_templates[1][1], "income")
            income_count += 1

    month = datetime.now().strftime("%Y-%m")
    db.set_budget("Food & Dining", 12000, month)
    db.set_budget("Transportation", 5000, month)
    db.set_budget("Shopping", 7000, month)
    db.set_budget("Bills & Utilities", 9000, month)

    db.add_financial_goal("Emergency Fund", 200000, (datetime.now().date() + timedelta(days=365)).strftime("%Y-%m-%d"), 45000)
    db.add_financial_goal("Vacation Fund", 80000, (datetime.now().date() + timedelta(days=210)).strftime("%Y-%m-%d"), 12000)

    db.add_portfolio_item("AAPL", 8, 172.40, (datetime.now().date() - timedelta(days=120)).strftime("%Y-%m-%d"))
    db.add_portfolio_item("MSFT", 5, 338.10, (datetime.now().date() - timedelta(days=90)).strftime("%Y-%m-%d"))
    db.add_portfolio_item("TSLA", 4, 216.75, (datetime.now().date() - timedelta(days=75)).strftime("%Y-%m-%d"))

    transactions = db.get_transactions()
    expenses = transactions[transactions["type"] == "expense"]

    print(f"User ready: {USERNAME}")
    print(f"Transactions: {len(transactions)} (expenses={expense_count}, income={income_count})")
    print(f"Expense days span: {expenses['date'].min()} to {expenses['date'].max()}")
    print(f"Categories present: {', '.join(sorted(expenses['category'].dropna().unique().tolist()))}")


if __name__ == "__main__":
    main()
