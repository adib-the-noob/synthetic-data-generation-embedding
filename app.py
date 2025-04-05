import json
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

llm = ChatOllama(
    model="llama3.2",
    temperature=0.3,  # Adjust the temperature for creativity vs. accuracy
)

PROMPT_TEMPLATE = """
Generate 3 unique and realistic FMCG (Fast-Moving Consumer Goods) product entries as a JSON list.

Each item in the list must follow this format:
{
  "id": "string (unique alphanumeric product ID)",
  "title": "string (brief, market-ready product title)",
  "description": "string (a richly detailed product description including its use-case, key features, and where applicable: nutritional information for food/beverage or ingredients for personal care and cleaning products)",
  "category": "string (one of: Beverages, Snacks, Personal Care, Cleaning, Household, Dairy, Bakery, Frozen Foods, etc.)",
  "brand": "string (realistic-sounding or fictional brand name)",
  "price": float (reasonable price in USD, e.g. 3.99)",
  "tags": ["string", ...] (array of relevant keywords such as 'organic', 'sugar-free', 'vegan', 'eco-friendly', etc.)
}

Guidelines:
- For **food or beverage products**, include a short nutrition facts section in the description (e.g., calories, fats, proteins, sugars, etc.).
- For **personal care, household, or cleaning products**, include typical **ingredients or active compounds** (e.g., aloe vera, sodium lauryl sulfate).
- Ensure diversity across categories, brands, and price points.
- Descriptions should read like they are from an online store or product catalog – informative, persuasive, and realistic.
- Make sure each product is **unique**, and the tone stays consistent.

Output only valid JSON. Do not include any additional text or explanations. The JSON should be well-structured and formatted.
"""

# save the response to a file
with open("fmcg_data.json", "w") as f:
    # Generate the FMCG product data
    response = json.loads(llm.invoke([
        HumanMessage(content=PROMPT_TEMPLATE)
    ]).content)
    f.write(json.dumps(response, indent=2))
    print(json.dumps(response, indent=2))
