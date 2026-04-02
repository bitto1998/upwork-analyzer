import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from crewai import Agent, Task, Crew, Process

# Creates the Webpage Look
st.set_page_config(page_title="Upwork AI Optimizer", layout="wide")
st.title("🚀 Upwork Competitor Analyzer (3.1 PRO VERSION)")
st.markdown("This system secretly browses top Upwork profiles for you and finds out what words make them rank high.")

# Loads your Google VIP Pass (API Key) safely
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("Missing Google API Key in secrets! Please add it in Streamlit settings.")

# Website text boxes
skill = st.text_input("What is your core freelance skill?", value="Social Media Marketing, Meta Ads")
client = st.text_input("Who is your ideal client?", value="E-commerce stores")

# When the user clicks the button, do this:
if st.button("Start Scraping and Analyzing"):
    with st.spinner("AI is thinking... Give it 1 to 3 minutes. Do not close the page."):
        
        # ---> UPDATED TO USE YOUR NEW 3.1 PRO MODEL <---
        gemini = ChatGoogleGenerativeAI(model="gemini-3.1-pro")
        search_tool = DuckDuckGoSearchRun()

        # Agent 1: The Client
        agent_buyer = Agent(
            role='Upwork Client Simulator',
            goal=f'Act like an upwork client hiring for {skill}. Guess exactly what they type into search.',
            backstory='You know exactly what wealthy business owners search for.',
            llm=gemini
        )

        # Agent 2: The Scraper
        agent_scraper = Agent(
            role='Profile Scraper',
            goal='Search Upwork for those queries using Google, and extract their data.',
            backstory='You use search tricks (like site:upwork.com/freelancers/) to find out what competitors are writing.',
            tools=[search_tool],
            llm=gemini,
            verbose=True
        )

        # Agent 3: The Strategist
        agent_seo = Agent(
            role='SEO Profile Master',
            goal='Tell the user what keywords to copy and how to write their profile to beat competitors.',
            backstory='You reverse-engineer winning Upwork formulas.',
            llm=gemini
        )

        # Giving Tasks to the Agents
        t1 = Task(
            description=f'List 5 queries a {client} uses to find a {skill} expert.', 
            agent=agent_buyer, 
            expected_output="A list of 5 search queries."
        )
        
        t2 = Task(
            description='Use the search tool on the queries from task 1 to read snippets of high ranking profiles. Look for patterns in how they write.', 
            agent=agent_scraper, 
            expected_output="A summary of what the top competitor profiles look like and say."
        )
        
        t3 = Task(
            description='Create a Master Blueprint for the user with: 1) Top 10 Keywords, 2) Formatting tips (paragraphs vs bullets), 3) A copy-paste example intro paragraph based ONLY on the competitors.', 
            agent=agent_seo, 
            expected_output="Final SEO Upwork Guide."
        )

        # Let the Agents do the work
        crew = Crew(
            agents=[agent_buyer, agent_scraper, agent_seo], 
            tasks=[t1, t2, t3], 
            process=Process.sequential
        )
        
        # Kick off the process
        result = crew.kickoff()

        # Display final result on the website
        st.success("✅ Analysis Complete! See your customized blueprint below:")
        st.markdown(result)
