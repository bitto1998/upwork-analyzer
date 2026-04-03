import streamlit as st
import os
from langchain_community.tools import DuckDuckGoSearchRun
from crewai import Agent, Task, Crew, Process

st.set_page_config(page_title="Upwork AI Optimizer", layout="wide")
st.title("🚀 Upwork Competitor Analyzer (PRO VERSION)")
st.markdown("This system secretly browses top Upwork profiles for you and finds out what words make them rank high.")

if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    # CrewAI's internal system strictly looks for GEMINI_API_KEY, so we duplicate it here
    os.environ["GEMINI_API_KEY"] = st.secrets["GOOGLE_API_KEY"] 
else:
    st.error("Missing Google API Key in secrets! Please add it in Streamlit settings.")

skill = st.text_input("What is your core freelance skill?", value="Social Media Marketing, Meta Ads")
client = st.text_input("Who is your ideal client?", value="E-commerce stores")

if st.button("Start Scraping and Analyzing"):
    with st.spinner("AI is thinking... Give it 1 to 3 minutes. Do not close the page."):
        try:
            search_tool = DuckDuckGoSearchRun()
            
            # Feed the model name as a direct string, bypassing LangChain entirely
            model_name = "gemini/gemini-3.1-flash-lite"

            agent_buyer = Agent(
                role='Upwork Client Simulator',
                goal=f'Act like an upwork client hiring for {skill}. Guess exactly what they type into search.',
                backstory='You know exactly what wealthy business owners search for.',
                llm=model_name
            )

            agent_scraper = Agent(
                role='Profile Scraper',
                goal='Search Upwork for those queries using Google, and extract their data.',
                backstory='You use search tricks (like site:upwork.com/freelancers/) to find out what competitors are writing.',
                tools=[search_tool],
                llm=model_name,
                verbose=True
            )

            agent_seo = Agent(
                role='SEO Profile Master',
                goal='Tell the user what keywords to copy and how to write their profile to beat competitors.',
                backstory='You reverse-engineer winning Upwork formulas.',
                llm=model_name
            )

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
                description='Create a Master Blueprint for the user with: Top 10 Keywords, Formatting tips, A copy-paste example intro paragraph based ONLY on the competitors.', 
                agent=agent_seo, 
                expected_output="Final SEO Upwork Guide."
            )

            crew = Crew(
                agents=[agent_buyer, agent_scraper, agent_seo], 
                tasks=[t1, t2, t3], 
                process=Process.sequential
            )
            
            result = crew.kickoff()

            st.success("✅ Analysis Complete! See your customized blueprint below:")
            # Updated to handle CrewAI's newest output formatting
            st.markdown(result.raw if hasattr(result, 'raw') else str(result))
            
        except Exception as e:
            st.error("The system hit a roadblock while building your agents. Here is the exact problem:")
            st.error(str(e))
