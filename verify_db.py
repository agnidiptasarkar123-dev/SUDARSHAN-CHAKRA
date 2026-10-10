import os
from main import engine, Base, SessionLocal, Customer, seed_deep_history, ml_predictor

def verify():
    # Delete DB if exists
    if os.path.exists("greenpulse.db"):
        os.remove("greenpulse.db")
        
    # Recreate tables and seed
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    
    print("Seeding Deep History Data...")
    seed_deep_history(db)
    ml_predictor.train_mock_models(db)
    
    # Query one customer
    customer = db.query(Customer).first()
    if not customer:
        print("No customers generated.")
        return
        
    print(f"\n--- Customer Profile ---")
    print(f"Name: {customer.name}, Age: {customer.age}, CIBIL: {customer.cibil_score}")
    print(f"Active Loans: {customer.active_loans}")
    print(f"Existing Products: {customer.existing_products}")
    
    print("\n--- 10-Year Transaction Timeline (Sample) ---")
    txs = customer.transactions
    print(f"Total Transactions: {len(txs)}")
    
    # Print the oldest 5 and newest 5
    txs_sorted = sorted(txs, key=lambda t: t.date)
    print("Oldest:")
    for t in txs_sorted[:5]:
        print(f"  {t.date}: £{t.amount} at {t.merchant_name} ({t.category})")
        
    print("...")
    print("Newest:")
    for t in txs_sorted[-5:]:
        print(f"  {t.date}: £{t.amount} at {t.merchant_name} ({t.category})")
        
    db.close()
    print("Verification complete.")

if __name__ == "__main__":
    verify()
