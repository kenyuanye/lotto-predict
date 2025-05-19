import pandas as pd

def load_draw_history(path="data/draw_history.xlsx"):
    try:
        df = pd.read_excel(path, parse_dates=["Draw Date"])
        df.columns = df.columns.astype(str).str.strip()
        df = df.dropna(subset=["Draw Date", "1", "2", "3", "4", "5", "6", "Power Ball"])
        df = df.sort_values("Draw Number").reset_index(drop=True)
        df["DrawIndex"] = df.index
        return df
    except Exception as e:
        print(f"❌ Failed to load draw history: {e}")
        return pd.DataFrame()

def load_active_tickets(path="data/active_tickets.xlsx"):
    try:
        df = pd.read_excel(path)
        df.columns = df.columns.astype(str).str.strip()
        expected_cols = [str(i) for i in range(1, 7)]
        if not all(col in df.columns for col in expected_cols):
            raise ValueError("Missing expected ticket columns.")
        ticket_sets = df[expected_cols].dropna(how="any").values.tolist()
        ticket_excludes = [set(map(int, row)) for row in ticket_sets]
        return ticket_excludes
    except Exception as e:
        print(f"⚠️ Failed to load active tickets: {e}")
        return []
