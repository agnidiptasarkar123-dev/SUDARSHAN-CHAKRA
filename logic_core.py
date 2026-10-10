"""
GreenPulse Platform - Module 2: The Logic Core
Feature Aggregation & Deterministic GHG Protocol Carbon Engine.
Strictly typed, zero fluff.
"""

from dataclasses import dataclass
from typing import Any, Iterable, List, Set


@dataclass(frozen=True)
class AggregatedFeatures:
    monthly_fuel_spend: float
    monthly_electricity_spend: float
    annual_fuel_spend: float
    annual_electricity_spend: float


class FeatureAggregator:
    """
    Deterministic feature aggregation for customer banking transactions.
    """
    FUEL_MERCHANTS: Set[str] = {"Shell", "BP", "Texaco"}
    UTILITY_MERCHANTS: Set[str] = {"British Gas", "EDF"}

    @classmethod
    def aggregate(cls, transactions: Iterable[Any]) -> AggregatedFeatures:
        monthly_fuel: float = 0.0
        monthly_elec: float = 0.0

        for tx in transactions:
            category = getattr(tx, "category", "") or ""
            merchant = getattr(tx, "merchant_name", "") or ""
            amount = float(getattr(tx, "amount", 0.0) or 0.0)

            # Fuel spend check
            if category.lower() == "fuel" or merchant in cls.FUEL_MERCHANTS:
                monthly_fuel += amount

            # Electricity / Utility spend check
            if category.lower() == "utilities" or merchant in cls.UTILITY_MERCHANTS:
                monthly_elec += amount

        monthly_fuel_spend = round(monthly_fuel, 2)
        monthly_electricity_spend = round(monthly_elec, 2)
        annual_fuel_spend = round(monthly_fuel_spend * 12.0, 2)
        annual_electricity_spend = round(monthly_electricity_spend * 12.0, 2)

        return AggregatedFeatures(
            monthly_fuel_spend=monthly_fuel_spend,
            monthly_electricity_spend=monthly_electricity_spend,
            annual_fuel_spend=annual_fuel_spend,
            annual_electricity_spend=annual_electricity_spend
        )


class CarbonEngine:
    """
    Deterministic GHG Protocol carbon calculation engine.
    Calculates avoided emissions (kg CO2e) for clean technology transitions.
    """
    PETROL_CO2_PER_LITRE: float = 2.31
    GRID_CO2_PER_KWH: float = 0.207
    AVG_FUEL_PRICE: float = 1.50
    AVG_ELEC_TARIFF: float = 0.28
    EV_EFFICIENCY_KWH_PER_MILE: float = 0.25
    AVG_ANNUAL_MILEAGE: float = 8000.0
    SOLAR_REPLACEMENT_RATIO: float = 0.40

    @classmethod
    def calculate_ev_avoided_emissions(cls, annual_fuel_spend: float) -> float:
        """
        Formula:
        ((annual_fuel_spend / AVG_FUEL_PRICE) * PETROL_CO2_PER_LITRE)
        - (AVG_ANNUAL_MILEAGE * EV_EFFICIENCY_KWH_PER_MILE * GRID_CO2_PER_KWH)
        """
        if annual_fuel_spend <= 0.0:
            return 0.0

        ice_emissions = (annual_fuel_spend / cls.AVG_FUEL_PRICE) * cls.PETROL_CO2_PER_LITRE
        ev_grid_emissions = (
            cls.AVG_ANNUAL_MILEAGE * cls.EV_EFFICIENCY_KWH_PER_MILE * cls.GRID_CO2_PER_KWH
        )
        avoided = ice_emissions - ev_grid_emissions
        rounded = round(avoided, 2)
        return max(0.0, rounded)

    @classmethod
    def calculate_solar_avoided_emissions(cls, annual_electricity_spend: float) -> float:
        """
        Formula:
        (annual_electricity_spend / AVG_ELEC_TARIFF) * SOLAR_REPLACEMENT_RATIO * GRID_CO2_PER_KWH
        """
        if annual_electricity_spend <= 0.0:
            return 0.0

        avoided = (
            (annual_electricity_spend / cls.AVG_ELEC_TARIFF)
            * cls.SOLAR_REPLACEMENT_RATIO
            * cls.GRID_CO2_PER_KWH
        )
        return round(avoided, 2)
