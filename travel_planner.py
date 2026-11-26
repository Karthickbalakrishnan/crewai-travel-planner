import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew

# Load OPENAI_API_KEY and MODEL from your .env file
load_dotenv()

# Ask the user for a city when the script starts
city = input("Which city? ").strip()

# [R] ROLE & [C] CONTEXT inside the Agent
planner = Agent(
    role="Zero-Waste Micro-Trip Strategist",  # [R] Role
    goal="Curate a seamless 3-stop walking loop in {city} for travelers with limited time.",
    backstory=(  # [C] Context
        "You are an urban mobility and cultural experience designer writing for busy "
        "solo travelers who only have half a day to explore {city}. You value spatial "
        "logic, walkable routes, and authentic local atmosphere over long cross-city "
        "commutes or generic sightseeing lists."
    ),
    verbose=True
)

# [T] TASK, [F] FORMAT & [R] RULES inside the Task
plan_trip_task = Task(
    description=(
        # [T] Task
        "Select 3 iconic yet geographically close places to visit in {city} that form "
        "a logical half-day walking route.\n\n"
        # [R] Rules / Constraints
        "Rules:\n"
        "- Do NOT pick locations on opposite sides of the city.\n"
        "- Do NOT include introductory greetings, filler text, or closing summaries.\n"
        "- Keep each description under 12 words, focusing on why it fits the route or the best time to go."
    ),
    expected_output=(
        # [F] Format
        "A clean, numbered list of exactly 3 stops formatted strictly as:\n"
        "1. Stop Name - concise spatial or timing highlight\n"
        "2. Stop Name - concise spatial or timing highlight\n"
        "3. Stop Name - concise spatial or timing highlight"
    ),
    agent=planner
)

# Assemble into a Crew
crew = Crew(
    agents=[planner],
    tasks=[plan_trip_task]
)

# Run and print the result
result = crew.kickoff(inputs={"city": city})
print(result)