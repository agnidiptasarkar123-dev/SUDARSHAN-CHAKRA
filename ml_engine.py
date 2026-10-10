"""
GreenPulse Platform - Module 3: ML Propensity Models & Hard Eligibility Gatekeeper
Strictly typed, zero fluff.
"""

import json
from typing import Any, Dict, List, Union

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from logic_core import FeatureAggregator


class MLPredictor:
    """
    Random Forest Propensity Predictor for Green Financial Products.
    """
    def __init__(self) -> None:
        self.ev_model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        self.solar_model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        self.is_trained: bool = False

    def train_mock_models(self, db_session: Any) -> None:
        """
        Fetches all customers and their aggregated spends, training in-memory
        Random Forest models for EV Finance and Solar Finance.
        """
        from main import Customer  # Local import to prevent circular dependency

        customers = db_session.query(Customer).all()
        if not customers:
            return

        features_list: List[List[float]] = []
        y_ev_list: List[int] = []
        y_solar_list: List[int] = []

        for cust in customers:
            agg = FeatureAggregator.aggregate(cust.transactions)
            vector = [
                float(agg.annual_fuel_spend),
                float(agg.annual_electricity_spend),
                float(cust.age),
                float(cust.credit_score),
            ]
            features_list.append(vector)

            # Domain heuristic training labels
            ev_label = 1 if (agg.annual_fuel_spend > 2500.0 and cust.credit_score >= 600) else 0
            solar_label = 1 if (agg.annual_electricity_spend > 3500.0 and cust.credit_score >= 580) else 0

            y_ev_list.append(ev_label)
            y_solar_list.append(solar_label)

        X = np.array(features_list)
        y_ev = np.array(y_ev_list)
        y_solar = np.array(y_solar_list)

        # Ensure at least two classes exist for probability calibration
        if len(np.unique(y_ev)) < 2:
            median_fuel = float(np.median(X[:, 0]))
            y_ev = (X[:, 0] >= median_fuel).astype(int)

        if len(np.unique(y_solar)) < 2:
            median_elec = float(np.median(X[:, 1]))
            y_solar = (X[:, 1] >= median_elec).astype(int)

        self.ev_model.fit(X, y_ev)
        self.solar_model.fit(X, y_solar)
        self.is_trained = True

    def predict_propensities(self, customer_features: Dict[str, float]) -> Dict[str, float]:
        """
        Returns probability score (0.0 to 1.0) for both products.
        """
        if not self.is_trained:
            fuel = customer_features.get("annual_fuel_spend", 0.0)
            elec = customer_features.get("annual_electricity_spend", 0.0)
            return {
                "EV Finance": round(min(1.0, fuel / 4500.0), 2),
                "Solar Finance": round(min(1.0, elec / 7000.0), 2),
            }

        vector = np.array([[
            float(customer_features.get("annual_fuel_spend", 0.0)),
            float(customer_features.get("annual_electricity_spend", 0.0)),
            float(customer_features.get("age", 35.0)),
            float(customer_features.get("credit_score", 700.0)),
        ]])

        ev_proba = self.ev_model.predict_proba(vector)[0]
        solar_proba = self.solar_model.predict_proba(vector)[0]

        ev_score = float(ev_proba[1]) if len(ev_proba) > 1 else float(ev_proba[0])
        solar_score = float(solar_proba[1]) if len(solar_proba) > 1 else float(solar_proba[0])

        return {
            "EV Finance": round(ev_score, 2),
            "Solar Finance": round(solar_score, 2),
        }

    @staticmethod
    def get_reason_codes(customer_features: Dict[str, float]) -> Dict[str, List[str]]:
        """
        Extracts top interpretability signals mimicking SHAP explanations.
        """
        ev_reasons: List[str] = []
        solar_reasons: List[str] = []

        fuel_spend = customer_features.get("annual_fuel_spend", 0.0)
        elec_spend = customer_features.get("annual_electricity_spend", 0.0)
        credit_score = customer_features.get("credit_score", 0.0)

        # EV Signals
        if fuel_spend > 2000.0:
            ev_reasons.append("HIGH_FUEL_SPEND")
        if fuel_spend > 3800.0:
            ev_reasons.append("EXCESSIVE_ICE_CARBON_INTENSITY")
        if credit_score >= 700.0:
            ev_reasons.append("EXCELLENT_CREDIT_PROFILE")
        if not ev_reasons:
            ev_reasons.append("STANDARD_COMMUTE_PROFILE")

        # Solar Signals
        if elec_spend > 3000.0:
            solar_reasons.append("HIGH_GRID_ELECTRICITY_COST")
        if elec_spend > 6000.0:
            solar_reasons.append("HIGH_DOMESTIC_POWER_CONSUMPTION")
        if credit_score >= 680.0:
            solar_reasons.append("QUALIFIED_HOME_OWNER_PROFILE")
        if not solar_reasons:
            solar_reasons.append("STANDARD_RESIDENTIAL_LOAD")

        return {
            "EV Finance": ev_reasons,
            "Solar Finance": solar_reasons,
        }


class EligibilityGatekeeper:
    """
    Hard business rule eligibility checks.
    """
    @staticmethod
    def check_eligibility(customer_data: Any, product_name: str) -> bool:
        """
        Rules:
        - If product_name == 'EV Finance' and 'EV' in existing_products: False
        - If product_name == 'Solar Finance' and 'Solar' in existing_products: False
        """
        existing_products = getattr(customer_data, "existing_products", "[]")
        if isinstance(customer_data, dict):
            existing_products = customer_data.get("existing_products", "[]")

        if not isinstance(existing_products, str):
            existing_products = json.dumps(existing_products)

        if product_name == "EV Finance" and "EV" in existing_products:
            return False

        if product_name == "Solar Finance" and "Solar" in existing_products:
            return False

        return True
