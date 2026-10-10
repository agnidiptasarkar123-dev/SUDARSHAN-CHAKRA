"""
Green Banking Platform - Module 1: Foundation & Data Pipeline
Engineered for rapid hackathon execution, strictly typed, zero fluff.
"""

import json
import random
from contextlib import asynccontextmanager
from datetime import date, timedelta
from typing import List, Optional

from faker import Faker
from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, field_validator
from sqlalchemy import Column, Date as SQLDate, Float, ForeignKey, Integer, String, create_engine
from sqlalchemy.orm import Session, declarative_base, relationship, sessionmaker

from logic_core import CarbonEngine, FeatureAggregator
from ml_engine import EligibilityGatekeeper, MLPredictor
from llm_orchestrator import LLMInsightGenerator, LLMInsightSchema

ml_predictor = MLPredictor()
llm_generator = LLMInsightGenerator()

# -----------------------------------------------------------------------------
# 1. Database Setup (SQLite: greenpulse.db)
# -----------------------------------------------------------------------------

DATABASE_URL = "sqlite:///./greenpulse.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# -----------------------------------------------------------------------------
# 2. Database Models
# -----------------------------------------------------------------------------

class Merchant(Base):
    __tablename__ = "merchants"

    id: int = Column(Integer, primary_key=True, index=True)
    name: str = Column(String, nullable=False)
    merchant_type: str = Column(String, nullable=False)


class Customer(Base):
    __tablename__ = "customers"

    id: int = Column(Integer, primary_key=True, index=True)
    name: str = Column(String, nullable=False)
    age: int = Column(Integer, nullable=False)
    cibil_score: int = Column(Integer, nullable=False)
    active_loans: str = Column(String, default="[]", nullable=False)  # JSON string
    existing_products: str = Column(String, default="[]", nullable=False)  # JSON string

    transactions = relationship(
        "Transaction",
        back_populates="customer",
        cascade="all, delete-orphan",
        order_by="desc(Transaction.date)"
    )


class Transaction(Base):
    __tablename__ = "transactions"

    id: int = Column(Integer, primary_key=True, index=True)
    customer_id: int = Column(Integer, ForeignKey("customers.id"), nullable=False, index=True)
    merchant_id: int = Column(Integer, ForeignKey("merchants.id"), nullable=False, index=True)
    amount: float = Column(Float, nullable=False)
    date: date = Column(SQLDate, nullable=False)

    customer = relationship("Customer", back_populates="transactions")
    merchant = relationship("Merchant", lazy="joined")

    @property
    def merchant_name(self) -> str:
        return self.merchant.name if self.merchant else ""

    @property
    def category(self) -> str:
        return self.merchant.merchant_type if self.merchant else ""


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# -----------------------------------------------------------------------------
# 3. Pydantic Schemas
# -----------------------------------------------------------------------------

class TransactionResponse(BaseModel):
    id: int
    customer_id: int
    merchant_id: int
    amount: float
    merchant_name: str
    date: date
    category: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class CustomerResponse(BaseModel):
    id: int
    name: str
    age: int
    cibil_score: int
    active_loans: List[str]
    existing_products: List[str]

    model_config = ConfigDict(from_attributes=True)

    @field_validator("existing_products", "active_loans", mode="before")
    @classmethod
    def parse_json_arrays(cls, v):
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                return parsed if isinstance(parsed, list) else [parsed]
            except Exception:
                return [v] if v else []
        elif isinstance(v, list):
            return v
        return []


class Customer360Response(BaseModel):
    id: int
    name: str
    age: int
    cibil_score: int
    active_loans: List[str]
    existing_products: List[str]
    total_transactions: int
    total_spend: float
    transactions: List[TransactionResponse]

    model_config = ConfigDict(from_attributes=True)

    @field_validator("existing_products", "active_loans", mode="before")
    @classmethod
    def parse_json_arrays(cls, v):
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                return parsed if isinstance(parsed, list) else [parsed]
            except Exception:
                return [v] if v else []
        elif isinstance(v, list):
            return v
        return []


class CarbonImpactResponse(BaseModel):
    customer_id: int
    monthly_fuel_spend: float
    monthly_electricity_spend: float
    annual_fuel_spend: float
    annual_electricity_spend: float
    ev_avoided_emissions_kg_co2e: float
    solar_avoided_emissions_kg_co2e: float
    total_avoided_emissions_kg_co2e: float

    model_config = ConfigDict(from_attributes=True)


class Opportunity(BaseModel):
    product_name: str
    propensity: float
    confidence: str
    is_eligible: bool
    reason_codes: List[str]
    carbon_savings_kg: float

    model_config = ConfigDict(from_attributes=True)


class OpportunitiesResponse(BaseModel):
    customer_id: int
    opportunities: List[Opportunity]

    model_config = ConfigDict(from_attributes=True)


class CustomerInsightResponse(BaseModel):
    customer_id: int
    product_name: str
    llm_insight: LLMInsightSchema

    model_config = ConfigDict(from_attributes=True)


# -----------------------------------------------------------------------------
# 4. Synthetic Data Generation (Faker with Deep Historical Patterns)
# -----------------------------------------------------------------------------

GREEN_PRODUCT_OPTIONS = [
    [],
    ["EV Loan"],
    ["Green Mortgage"],
    ["Solar Panel Financing"],
    ["EV Loan", "Solar Panel Financing"],
    ["Eco Home Improvement"],
]

def seed_deep_history(db: Session) -> None:
    customer_count = db.query(Customer).count()
    if customer_count > 0:
        return

    fake = Faker("en_GB")
    Faker.seed(42)
    random.seed(42)

    today = date.today()

    # 1. Create 20 Fixed Merchants
    merchants = []
    merchant_data = [
        ("Shell", "Fuel"), ("BP", "Fuel"), ("Texaco", "Fuel"),
        ("British Gas", "Utilities"), ("EDF", "Utilities"),
        ("Tesco", "Groceries"), ("Sainsbury's", "Groceries"), ("Marks & Spencer", "Groceries"),
        ("Transport for London", "Public Transport"), ("National Rail", "Public Transport"),
        ("Amazon UK", "Retail"), ("Costa Coffee", "Dining"),
        ("Starbucks", "Dining"), ("ASDA", "Groceries"), ("Morrisons", "Groceries"),
        ("Uber", "Transport"), ("Deliveroo", "Dining"), ("Apple Store", "Retail"),
        ("John Lewis", "Retail"), ("Argos", "Retail")
    ]
    for name, mtype in merchant_data:
        m = Merchant(name=name, merchant_type=mtype)
        db.add(m)
        merchants.append(m)
    db.flush()

    fuel_merchants = [m for m in merchants if m.merchant_type == "Fuel"]
    utility_merchants = [m for m in merchants if m.merchant_type == "Utilities"]
    other_merchants = [m for m in merchants if m.merchant_type not in ("Fuel", "Utilities")]

    # 2. Generate 50 Customers with correlated ages, CIBIL and active loans
    customers = []
    for _ in range(50):
        age = random.randint(22, 75)
        cibil = random.randint(300, 900)
        
        active_loans = []
        if cibil > 750:
            active_loans = random.choice([["Home Loan"], ["Home Loan", "Auto Loan"], ["Personal Loan"]])
        elif cibil > 600:
            active_loans = random.choice([["Personal Loan"], ["Auto Loan"], []])
        else:
            active_loans = random.choice([[], ["Payday Loan"]])
            
        cust = Customer(
            name=fake.name(),
            age=age,
            cibil_score=cibil,
            active_loans=json.dumps(active_loans),
            existing_products=json.dumps(random.choice(GREEN_PRODUCT_OPTIONS))
        )
        db.add(cust)
        customers.append(cust)
    db.flush()

    # 3. Generate Deep Transaction History
    historical_years = [10, 8, 7, 6, 5, 4, 3, 2, 1]
    recent_months = list(range(1, 13))

    for cust in customers:
        txs = []
        
        # Yearly checkpoints
        for y in historical_years:
            tx_date = today - timedelta(days=y * 365 + random.randint(-15, 15))
            for _ in range(random.randint(2, 5)):
                m = random.choice(merchants)
                amt = round(random.uniform(10.0, 150.0), 2)
                txs.append(Transaction(customer_id=cust.id, merchant_id=m.id, amount=amt, date=tx_date))
                
        # Monthly checkpoints
        for m_ago in recent_months:
            tx_date = today - timedelta(days=m_ago * 30 + random.randint(-5, 5))
            # Force high carbon & utility signals
            for _ in range(random.randint(1, 2)):
                m = random.choice(fuel_merchants)
                amt = round(random.uniform(45.0, 110.0), 2)
                txs.append(Transaction(customer_id=cust.id, merchant_id=m.id, amount=amt, date=tx_date))
                
            for _ in range(random.randint(1, 2)):
                m = random.choice(utility_merchants)
                amt = round(random.uniform(80.0, 240.0), 2)
                txs.append(Transaction(customer_id=cust.id, merchant_id=m.id, amount=amt, date=tx_date))
                
            for _ in range(random.randint(2, 5)):
                m = random.choice(other_merchants)
                amt = round(random.uniform(10.0, 150.0), 2)
                txs.append(Transaction(customer_id=cust.id, merchant_id=m.id, amount=amt, date=tx_date))
                
        txs.sort(key=lambda t: t.date, reverse=True)
        db.add_all(txs)
        
    db.commit()


# -----------------------------------------------------------------------------
# 5. FastAPI Lifespan & Application
# -----------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        seed_deep_history(db)
        ml_predictor.train_mock_models(db)
    finally:
        db.close()
    yield


app = FastAPI(
    title="GreenPulse Platform API",
    description="Green Banking Customer 360 & ESG Data Pipeline",
    version="1.0.0",
    lifespan=lifespan
)


# -----------------------------------------------------------------------------
# 6. Endpoints
# -----------------------------------------------------------------------------

@app.get("/", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "service": "GreenPulse Platform",
        "docs": "/docs"
    }


@app.get(
    "/api/customers/{customer_id}",
    response_model=Customer360Response,
    summary="Customer 360 View",
    tags=["Customer 360"]
)
def get_customer_360(
    customer_id: int,
    db: Session = Depends(get_db)
):
    customer = db.query(Customer).filter(Customer.id == customer_id).first()
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customer with ID {customer_id} not found"
        )

    txs = customer.transactions
    total_spend = round(sum(t.amount for t in txs), 2)

    return Customer360Response(
        id=customer.id,
        name=customer.name,
        age=customer.age,
        cibil_score=customer.cibil_score,
        active_loans=customer.active_loans,
        existing_products=customer.existing_products,
        total_transactions=len(txs),
        total_spend=total_spend,
        transactions=txs
    )


@app.get(
    "/api/customers/{customer_id}/carbon-impact",
    response_model=CarbonImpactResponse,
    summary="Customer Carbon Impact & Avoided Emissions",
    tags=["Carbon Engine"]
)
def get_customer_carbon_impact(
    customer_id: int,
    db: Session = Depends(get_db)
):
    customer = db.query(Customer).filter(Customer.id == customer_id).first()
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customer with ID {customer_id} not found"
        )

    # 1. Feature Aggregation
    features = FeatureAggregator.aggregate(customer.transactions)

    # 2. GHG Protocol Carbon Engine calculations
    ev_avoided = CarbonEngine.calculate_ev_avoided_emissions(features.annual_fuel_spend)
    solar_avoided = CarbonEngine.calculate_solar_avoided_emissions(features.annual_electricity_spend)
    total_avoided = round(ev_avoided + solar_avoided, 2)

    return CarbonImpactResponse(
        customer_id=customer.id,
        monthly_fuel_spend=features.monthly_fuel_spend,
        monthly_electricity_spend=features.monthly_electricity_spend,
        annual_fuel_spend=features.annual_fuel_spend,
        annual_electricity_spend=features.annual_electricity_spend,
        ev_avoided_emissions_kg_co2e=ev_avoided,
        solar_avoided_emissions_kg_co2e=solar_avoided,
        total_avoided_emissions_kg_co2e=total_avoided
    )


@app.get(
    "/api/customers/{customer_id}/opportunities",
    response_model=OpportunitiesResponse,
    summary="Customer Green Product Opportunities",
    tags=["Opportunities"]
)
def get_customer_opportunities(
    customer_id: int,
    db: Session = Depends(get_db)
):
    customer = db.query(Customer).filter(Customer.id == customer_id).first()
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customer with ID {customer_id} not found"
        )

    # 1. Feature Aggregation
    features = FeatureAggregator.aggregate(customer.transactions)
    cust_feature_dict = {
        "annual_fuel_spend": features.annual_fuel_spend,
        "annual_electricity_spend": features.annual_electricity_spend,
        "age": float(customer.age),
        "cibil_score": float(customer.cibil_score),
    }

    # 2. ML Propensity Scores & SHAP-mimic Reason Codes
    propensities = ml_predictor.predict_propensities(cust_feature_dict)
    reason_codes_dict = ml_predictor.get_reason_codes(cust_feature_dict)

    # 3. GHG Protocol Avoided Emissions
    ev_savings = CarbonEngine.calculate_ev_avoided_emissions(features.annual_fuel_spend)
    solar_savings = CarbonEngine.calculate_solar_avoided_emissions(features.annual_electricity_spend)

    # 4. Product Opportunities & Hard Gatekeeper Validation
    products_info = [
        ("EV Finance", propensities.get("EV Finance", 0.0), reason_codes_dict.get("EV Finance", []), ev_savings),
        ("Solar Finance", propensities.get("Solar Finance", 0.0), reason_codes_dict.get("Solar Finance", []), solar_savings),
    ]

    opportunities: List[Opportunity] = []
    for prod_name, score, reasons, carbon in products_info:
        is_eligible = EligibilityGatekeeper.check_eligibility(customer, prod_name)

        if score >= 0.70:
            confidence = "HIGH"
        elif score >= 0.40:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        opportunities.append(
            Opportunity(
                product_name=prod_name,
                propensity=score,
                confidence=confidence,
                is_eligible=is_eligible,
                reason_codes=reasons,
                carbon_savings_kg=carbon
            )
        )

    return OpportunitiesResponse(
        customer_id=customer.id,
        opportunities=opportunities
    )


@app.get(
    "/api/customers/{customer_id}/insights",
    response_model=CustomerInsightResponse,
    summary="Neuro-Symbolic AI Green Advisory Insights",
    tags=["AI Advisory"]
)
def get_customer_insights(
    customer_id: int,
    db: Session = Depends(get_db)
):
    customer = db.query(Customer).filter(Customer.id == customer_id).first()
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customer with ID {customer_id} not found"
        )

    # 1. Feature Aggregation
    features = FeatureAggregator.aggregate(customer.transactions)
    cust_feature_dict = {
        "annual_fuel_spend": features.annual_fuel_spend,
        "annual_electricity_spend": features.annual_electricity_spend,
        "age": float(customer.age),
        "cibil_score": float(customer.cibil_score),
    }

    # 2. ML Propensity Scores & SHAP Reason Codes
    propensities = ml_predictor.predict_propensities(cust_feature_dict)
    reason_codes_dict = ml_predictor.get_reason_codes(cust_feature_dict)

    # 3. GHG Protocol Avoided Emissions
    ev_savings = CarbonEngine.calculate_ev_avoided_emissions(features.annual_fuel_spend)
    solar_savings = CarbonEngine.calculate_solar_avoided_emissions(features.annual_electricity_spend)

    # 4. Opportunity Candidates
    candidates = [
        {
            "product_name": "EV Finance",
            "propensity": propensities.get("EV Finance", 0.0),
            "is_eligible": EligibilityGatekeeper.check_eligibility(customer, "EV Finance"),
            "reason_codes": reason_codes_dict.get("EV Finance", []),
            "carbon_savings_kg": ev_savings
        },
        {
            "product_name": "Solar Finance",
            "propensity": propensities.get("Solar Finance", 0.0),
            "is_eligible": EligibilityGatekeeper.check_eligibility(customer, "Solar Finance"),
            "reason_codes": reason_codes_dict.get("Solar Finance", []),
            "carbon_savings_kg": solar_savings
        }
    ]

    # Filter for eligible products first
    eligible_candidates = [c for c in candidates if c["is_eligible"]]
    target_list = eligible_candidates if eligible_candidates else candidates

    # Rank by propensity descending
    target_list.sort(key=lambda x: x["propensity"], reverse=True)
    top_opportunity = target_list[0]

    # 5. Generate LLM Advisory Insight (Gemini with deterministic fail-safe)
    insight = llm_generator.generate_insight(top_opportunity)

    return CustomerInsightResponse(
        customer_id=customer.id,
        product_name=top_opportunity["product_name"],
        llm_insight=insight
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
