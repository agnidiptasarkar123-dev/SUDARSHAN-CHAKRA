import httpx
import pytest

BASE_URL = "http://127.0.0.1:8000"

def test_data_integrity():
    """Test 1: Data Integrity - Customer360 schema."""
    response = httpx.get(f"{BASE_URL}/api/customers/1")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    
    # Assert expected keys exist
    expected_keys = {
        "id", "name", "age", "existing_products", "credit_score",
        "total_transactions", "total_spend", "transactions"
    }
    assert expected_keys.issubset(data.keys()), f"Missing keys in response: {expected_keys - data.keys()}"
    
    # Check types
    assert isinstance(data["id"], int)
    assert isinstance(data["name"], str)
    assert isinstance(data["total_transactions"], int)
    assert isinstance(data["transactions"], list)

def test_math_stability():
    """Test 2: Math Stability - No negative carbon savings, safe division."""
    response = httpx.get(f"{BASE_URL}/api/customers/1/carbon-impact")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    
    # Check keys
    expected_keys = {
        "customer_id", "annual_fuel_spend", "annual_electricity_spend",
        "ev_avoided_emissions_kg_co2e", "solar_avoided_emissions_kg_co2e",
        "total_avoided_emissions_kg_co2e"
    }
    assert expected_keys.issubset(data.keys()), f"Missing keys in response: {expected_keys - data.keys()}"
    
    # Check non-negative
    assert data["total_avoided_emissions_kg_co2e"] >= 0
    assert data["ev_avoided_emissions_kg_co2e"] >= 0
    assert data["solar_avoided_emissions_kg_co2e"] >= 0

def test_hard_gatekeeper():
    """Test 3: Hard Gatekeeper - Customer with EV doesn't get EV Finance."""
    # Find a customer with EV in existing products by iterating 1 to 20
    ev_customer_id = None
    for cid in range(1, 21):
        c_resp = httpx.get(f"{BASE_URL}/api/customers/{cid}")
        if c_resp.status_code == 200:
            c = c_resp.json()
            products = c.get("existing_products", [])
            if any("EV" in p for p in products):
                ev_customer_id = cid
                break
                
    assert ev_customer_id is not None, "Could not find a synthetic customer with EV product for testing."

    
    # Check their opportunities
    opp_resp = httpx.get(f"{BASE_URL}/api/customers/{ev_customer_id}/opportunities")
    assert opp_resp.status_code == 200
    opp_data = opp_resp.json()
    
    for opp in opp_data["opportunities"]:
        if opp["product_name"] == "EV Finance":
            assert opp["is_eligible"] is False, "Gatekeeper failed: EV Finance should be ineligible for existing EV owner"

def test_llm_fail_safe():
    """Test 4: LLM Fail-Safe - Insights schema matches."""
    response = httpx.get(f"{BASE_URL}/api/customers/1/insights")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    
    # Check outer keys
    assert "llm_insight" in data
    insight = data["llm_insight"]
    
    # Check inner keys (LLMInsightSchema)
    expected_keys = {
        "headline", "why_this_recommendation", 
        "estimated_impact_explained", "primary_call_to_action"
    }
    assert expected_keys.issubset(insight.keys()), f"Missing keys in insight: {expected_keys - insight.keys()}"
    
    # Check types
    assert isinstance(insight["headline"], str)
    assert isinstance(insight["why_this_recommendation"], str)
    assert isinstance(insight["estimated_impact_explained"], str)
    assert isinstance(insight["primary_call_to_action"], str)
    
    # Ensure they are not empty strings
    assert len(insight["headline"]) > 0
    assert len(insight["why_this_recommendation"]) > 0
