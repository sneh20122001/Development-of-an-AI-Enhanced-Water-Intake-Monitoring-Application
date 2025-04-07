import os 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001",
    temperature=0,
    api_key=GOOGLE_API_KEY
)

class WaterIntakeAgent:


    def __init__(self):
        self.history = []

    def analyze_intake(self, intake_ml):


        prompt = f"""
        You are a hydration assistant. The user has consumed {intake_ml} ml of water today.
        Provide a hydration status and suggest if they need to drink more water.
        """

        response = llm.invoke([HumanMessage(content=prompt)])

        return response.content
    

if __name__ == "__main__":
    agent = WaterIntakeAgent()
    intake = 1500
    feedback = agent.analyze_intake(intake)
    print(f"Hydration Analysis: {feedback}")