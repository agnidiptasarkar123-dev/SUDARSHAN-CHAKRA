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

class Customer(Base):
    __tablename__ = "customers"

    id: int = Column(Integer, primary_key=True, index=True)
    name: str = Column(String, nullable=False)
    age: int = Column(Integer, nullable=False)
    existing_products: str = Column(String, default="[]", nullable=False)  # JSON string
    credit_score: int = Column(Integer, nullable=False)

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
    amount: float = Column(Float, nullable=False)
    merchant_name: str = Column(String, nullable=False)
    date: date = Column(SQLDate, nullable=False)
    category: Optional[str] = Column(String, nullable=True)

    customer = relationship("Customer", back_populates="transactions")


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
    amount: float
    merchant_name: str
    date: date
    category: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class CustomerResponse(BaseModel):
    id: int
    name: str
    age: int
    existing_products: List[str]
    credit_score: int

    model_config = ConfigDict(from_attributes=True)

    @field_validator("existing_products", mode="before")
    @classmethod
    def parse_existing_products(cls, v):
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
    existing_products: List[str]
    credit_score: int
    total_transactions: int
    total_spend: float
    transactions: List[TransactionResponse]

    model_config = ConfigDict(from_attributes=True)

    @field_validator("existing_products", mode="before")
    @classmethod
    def parse_existing_products(cls, v):
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
# 4. Synthetic Data Generation (Faker with High-Carbon & High-Utility Signals)
# -----------------------------------------------------------------------------

HIGH_CARBON_MERCHANTS = ["Shell", "BP", "Texaco"]
HIGH_UTILITY_MERCHANTS = ["British Gas", "EDF"]
GENERAL_MERCHANTS = [
    ("Tesco", "Groceries", (15.0, 120.0)),
    ("Sainsbury's", "Groceries", (12.0, 110.0)),
    ("Marks & Spencer", "Groceries", (20.0, 130.0)),
    ("Transport for London", "Public Transport", (3.5, 30.0)),
    ("National Rail", "Public Transport", (12.0, 75.0)),
    ("Amazon UK", "Retail", (15.0, 180.0)),
    ("Costa Coffee", "Dining", (3.5, 25.0)),
]

GREEN_PRODUCT_OPTIONS = [
    [],
    ["EV Loan"],
    ["Green Mortgage"],
    ["Solar Panel Financing"],
    ["EV Loan", "Solar Panel Financing"],
    ["Eco Home Improvement"],
]


def seed_database_if_empty(db: Session) -> None:
    customer_count = db.query(Customer).count()
    if customer_count > 0:
        return

    fake = Faker("en_GB")
    Faker.seed(42)
    random.seed(42)

    today = date.today()
    created_customers: List[Customer] = []

    # 1. Generate 50 Customers
    for _ in range(50):
        cust = Customer(
            name=fake.name(),
            age=random.randint(22, 70),
            credit_score=random.randint(550, 850),
            existing_products=json.dumps(random.choice(GREEN_PRODUCT_OPTIONS))
        )
        db.add(cust)
        created_customers.append(cust)

    db.flush()

    # 2. Generate exactly 20 transactions per customer with guaranteed carbon & utility signals
    for cust in created_customers:
        transactions: List[Transaction] = []

        # Forced high-carbon fuel signals (3-5 transactions)
        carbon_count = random.randint(3, 5)
        for _ in range(carbon_count):
            tx_date = today - timedelta(days=random.randint(1, 90))
            transactions.append(
                Transaction(
                    customer_id=cust.id,
                    amount=round(random.uniform(45.0, 110.0), 2),
                    merchant_name=random.choice(HIGH_CARBON_MERCHANTS),
                    date=tx_date,
                    category="Fuel"
                )
            )

        # Forced high-utility electricity/gas signals (2-4 transactions)
        utility_count = random.randint(2, 4)
        for _ in range(utility_count):
            tx_date = today - timedelta(days=random.randint(1, 90))
            transactions.append(
                Transaction(
                    customer_id=cust.id,
                    amount=round(random.uniform(80.0, 240.0), 2),
                    merchant_name=random.choice(HIGH_UTILITY_MERCHANTS),
                    date=tx_date,
                    category="Utilities"
                )
            )

        # Remaining transactions to reach exactly 20 transactions
        remaining_count = 20 - (carbon_count + utility_count)
        for _ in range(remaining_count):
            tx_date = today - timedelta(days=random.randint(1, 90))
            merchant_name, category, (min_amt, max_amt) = random.choice(GENERAL_MERCHANTS)
            transactions.append(
                Transaction(
                    customer_id=cust.id,
                    amount=round(random.uniform(min_amt, max_amt), 2),
                    merchant_name=merchant_name,
                    date=tx_date,
                    category=category
                )
            )

        transactions.sort(key=lambda t: t.date, reverse=True)
        db.add_all(transactions)

    db.commit()


# -----------------------------------------------------------------------------
# 5. FastAPI Lifespan & Application
# -----------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        seed_database_if_empty(db)
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
        existing_products=customer.existing_products,
        credit_score=customer.credit_score,
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
        "credit_score": float(customer.credit_score),
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
        "credit_score": float(customer.credit_score),
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
