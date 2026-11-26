from crewai import Agent, Task, Crew, LLM


def build_crew() -> Crew:
    local_llm = LLM(
        model="ollama/qwen2.5:latest",
        base_url="http://localhost:11434"
    )

    planner = Agent(
        role="Zero-Waste Micro-Trip Strategist",
        goal="Recommend the top 3 walkable places to visit in {city} for time-conscious travelers.",
        backstory=(
            "You are an urban experience designer writing for busy solo travelers "
            "who want to see the best of {city} without wasting time in transit. "
            "You skip generic tourist fluff and deliver crisp, actionable picks."
        ),
        llm=local_llm,
        verbose=True,
    )

    task = Task(
        description=(
            "Identify the 3 best places to visit in {city}.\n\n"
            "Rules:\n"
            "- Suggest exactly 3 places—no more, no less.\n"
            "- Do NOT include intro text, restaurant lists, transit essays, or closing summaries.\n"
            "- Keep each highlight under 10 words, noting why to go or the best time to visit."
        ),
        expected_output=(
            "A numbered list of exactly 3 places in {city}, formatted strictly as:\n"
            "1. Place Name - concise highlight or best time to visit\n"
            "2. Place Name - concise highlight or best time to visit\n"
            "3. Place Name - concise highlight or best time to visit"
        ),
        agent=planner,
    )

    return Crew(agents=[planner], tasks=[task])


def main() -> None:
    city = input("Which city? ").strip()
    if not city:
        print("Please enter a city name.")
        return

    crew = build_crew()
    result = crew.kickoff(inputs={"city": city})
    print(f"\n{result.raw}")


if __name__ == "__main__":
    main()
