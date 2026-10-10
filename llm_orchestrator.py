"""
GreenPulse Platform - Module 4: The Neuro-Symbolic LLM Layer (Gemini Orchestration)
Strictly typed, zero fluff, deterministic fallback fail-safe.
"""

import json
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

# Load environment variables (.env) if present
load_dotenv()


class LLMInsightSchema(BaseModel):
    """
    Strict schema for green banking advisory recommendations.
    """
    headline: str
    why_this_recommendation: str
    estimated_impact_explained: str
    primary_call_to_action: str

    model_config = ConfigDict(from_attributes=True)


class LLMInsightGenerator:
    """
    Orchestrates Google Gemini LLM generation with strict JSON schema
    and deterministic fallback mechanism for hackathon reliability.
    """

    SYSTEM_INSTRUCTION = (
        "You are an expert green banking advisor. You will receive structured JSON containing "
        "a customer's propensity score, reason codes, and exact carbon savings for a green product. "
        "Your ONLY job is to translate this data into a friendly, professional advisory message fitting the schema. "
        "DO NOT invent any numbers. DO NOT calculate anything. If the data says 7009 kg CO2e, say exactly that."
    )

    def __init__(self) -> None:
        self.api_key: Optional[str] = os.environ.get("GEMINI_API_KEY")
        self._genai_client = None

        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._genai_client = genai
            except Exception as e:
                print(f"[LLMInsightGenerator] Gemini initialization warning: {e}")
                self._genai_client = None

    def _generate_fallback(self, opportunity_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deterministic, audit-compliant fallback used if GEMINI_API_KEY is unset or API call fails.
        """
        product = opportunity_data.get("product_name", "Green Product")
        carbon = opportunity_data.get("carbon_savings_kg", 0.0)
        reasons = opportunity_data.get("reason_codes", [])
        reasons_str = ", ".join(reasons) if reasons else "identified consumption profile"

        if product == "EV Finance":
            return {
                "headline": f"Transition to Clean Driving & Avoid {carbon} kg CO2e Annually",
                "why_this_recommendation": (
                    f"Your spending profile indicates significant fuel expenditure ({reasons_str}). "
                    "Switching from internal combustion to electric mobility provides immediate carbon and operational savings."
                ),
                "estimated_impact_explained": (
                    f"Based on your transaction history, driving an EV eliminates approximately {carbon} kg CO2e "
                    "in tailpipe emissions every year after accounting for UK grid intensity."
                ),
                "primary_call_to_action": "Apply for GreenPulse EV Finance with preferential green interest rates."
            }
        elif product == "Solar Finance":
            return {
                "headline": f"Generate On-Site Clean Solar & Offset {carbon} kg CO2e Annually",
                "why_this_recommendation": (
                    f"Your account activity shows substantial utility expenses ({reasons_str}). "
                    "Installing rooftop solar panels replaces high-tariff grid draw with clean, self-generated energy."
                ),
                "estimated_impact_explained": (
                    f"Adopting residential solar is projected to prevent {carbon} kg CO2e in annual grid emissions "
                    "while lowering long-term electricity bills."
                ),
                "primary_call_to_action": "Check your rooftop solar eligibility with GreenPulse Solar Financing."
            }
        else:
            return {
                "headline": f"Unlock Sustainable Savings & Reduce {carbon} kg CO2e",
                "why_this_recommendation": f"Tailored recommendation based on your verified account patterns ({reasons_str}).",
                "estimated_impact_explained": f"This sustainable transition prevents an estimated {carbon} kg CO2e emissions annually.",
                "primary_call_to_action": f"Explore {product} options in your GreenPulse banking portal."
            }

    def generate_insight(self, opportunity_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates structured green banking advice using Gemini or returns deterministic fallback.
        """
        if not self._genai_client or not self.api_key:
            return self._generate_fallback(opportunity_data)

        try:
            genai = self._genai_client
            # Recommended fast model
            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                generation_config={
                    "temperature": 0.0,
                    "response_mime_type": "application/json",
                },
                system_instruction=self.SYSTEM_INSTRUCTION
            )

            prompt = (
                f"Customer Opportunity Data:\n{json.dumps(opportunity_data, indent=2)}\n\n"
                f"Generate a JSON object matching this exact schema:\n"
                f"{{\n"
                f'  "headline": "Concise compelling advisory title",\n'
                f'  "why_this_recommendation": "Rationale grounded directly in the provided reason codes and data",\n'
                f'  "estimated_impact_explained": "Carbon impact explanation stating the exact carbon_savings_kg provided",\n'
                f'  "primary_call_to_action": "Clear next action step for the customer"\n'
                f"}}"
            )

            response = model.generate_content(prompt)
            raw_text = response.text.strip()
            parsed = json.loads(raw_text)

            # Validate against strict Pydantic schema
            validated = LLMInsightSchema(**parsed)
            return validated.model_dump()

        except Exception as e:
            print(f"[LLMInsightGenerator] API execution exception (using fallback): {e}")
            return self._generate_fallback(opportunity_data)
