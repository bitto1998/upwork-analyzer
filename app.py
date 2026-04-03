import streamlit as st
import os
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from duckduckgo_search import DDGS

# --- PAGE CONFIG ---
st.set_page_config(page_title="Upwork AI Optimizer PRO", layout="wide")
st.title("🚀 Upwork Competitor Analyzer & Profile Writer (PRO VERSION)")
st.markdown("Reverse-engineer top freelancers and generate an algorithm-beating profile for yourself.")

# --- SECRETS & API ---
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    os.environ["GEMINI_API_KEY"] = st.secrets["GOOGLE_API_KEY"] 
else:
    st.error("Missing Google API Key in secrets! Please add it in Streamlit settings.")

# --- ENHANCED USER INPUTS ---
st.sidebar.header("🎯 Your Freelance Profile")
skill = st.sidebar.text_input("Main Skill/Niche", value="Digital Marketing, Meta Ads & Funnels")
client = st.sidebar.text_input("Ideal Client / Industry", value="7-Figure E-commerce brands")
experience = st.sidebar.text_input("Years of Experience / Top Stat", value="5 years, generated over $1M in ad revenue")
tone = st.sidebar.selectbox("Profile Tone", ["Professional & Authoritative", "Conversational & Results-Driven", "Aggressive & Direct"])

# --- SMARTER TOOL ---
@tool("Web Search Tool")
def search_tool(query: str) -> str:
    """Search DuckDuckGo strictly for Upwork Freelancer profiles based on keywords."""
    # Force the search engine to ONLY look at upwork profiles
    boolean_query = f'site:upwork.com/freelancers/ {query}'
    results = DDGS().text(boolean_query, max_results=5)
    return str(results)

# --- APP EXECUTION ---
if st.button("🔥 Analyze Competitors & Write My Profile"):
    with st.spinner("Executing multi-agent Upwork workflow... Analyzing intent, searching competitors, and drafting copy... (Give it 3-5 minutes)."):
        try:
            model_name = "gemini/gemini-2.5-flash"

            # --- AGENT 1: The Client Psychologist ---
            agent_buyer = Agent(
                role='Target Audience Psychologist',
                goal=f'Understand the specific pain points and search behaviors of a {client} looking for a {skill} expert.',
                backstory="You have hired hundreds of freelancers on Upwork. You know exactly what wealthy business owners search for and the words that catch their attention.",
                llm=model_name
            )

            # --- AGENT 2: The Upwork SEO Analyst ---
            agent_scraper = Agent(
                role='Upwork Algorithm & Intelligence Analyst',
                goal='Analyze the search terms provided, scrape top-tier Upwork profiles, and identify the recurring keywords in their Profile Titles and Bios.',
                backstory="You are an SEO hacker specializing in Upwork's internal search algorithm. You use exact-match Boolean searches to read the snippets of top-earning competitors to see what keywords get them ranked on page 1.",
                tools=[search_tool],
                llm=model_name,
                verbose=True
            )

            # --- AGENT 3: The Conversion Copywriter ---
            agent_copywriter = Agent(
                role='Direct-Response Upwork Profile Copywriter',
                goal='Write a highly-optimized, high-converting Upwork profile title and description tailored for the user.',
                backstory=f"You are an elite Upwork profile writer. You blend competitor data with the freelancer's actual experience. Your focus is strictly on Title SEO, an undeniable first-2-sentence Hook, and persuasive bio copy. Your tone is {tone}.",
                llm=model_name
            )

            # --- TASKS (A Sequential Workflow) ---
            t1_buyer_intent = Task(
                description=f'Identify 5 exact-match search queries that a {client} will type into Upwork to hire someone for {skill}. List their top 3 biggest business pain points.', 
                agent=agent_buyer, 
                expected_output="A list of 5 search queries and 3 core pain points."
            )
            
            t2_competitor_analysis = Task(
                description='Using the exact-match queries from Task 1, use the Search Tool to scan Upwork competitor profiles. Extract a list of the 5 most commonly used words in their titles and descriptions. Note any guarantees or numbers they use.', 
                agent=agent_scraper, 
                expected_output="A brief report of Top Competitor Titles, most frequent keywords, and common formatting tricks used by top-earning freelancers in this niche."
            )
            
            t3_profile_creation = Task(
                description=f'''
                Using the competitor research from Task 2, craft an ultimate Upwork Profile Blueprint for this specific user.
                Here is the user's personal context: Experience/Achievements: "{experience}". 
                Tone required: "{tone}".

                YOU MUST OUTPUT EXACLTY THIS FORMAT:
                1. 👑 **Optimized Profile Title:** (Give 3 variations max 70 characters. Keyword-heavy).
                2. 🎣 **The "Feed" Hook:** (The first 2 sentences. THIS IS CRUCIAL. It's the only thing clients see before clicking "View Profile". Start with the user's achievement, not "Hi my name is").
                3. 📜 **The Profile Body:** (Bullet points of what they do, solving the pain points identified in Task 1).
                4. 🚀 **Call to Action (CTA):** (How to get the client to invite them to a job).
                5. 🏷️ **10 Exact Skill Tags:** (A comma-separated list of Upwork tags to attach to the profile).
                ''', 
                agent=agent_copywriter, 
                expected_output="A structured, ready-to-copy-paste Upwork Profile."
            )

            crew = Crew(
                agents=[agent_buyer, agent_scraper, agent_copywriter], 
                tasks=[t1_buyer_intent, t2_competitor_analysis, t3_profile_creation], 
                process=Process.sequential,
                max_rpm=4 
            )
            
            result = crew.kickoff()

            # --- DISPLAY RESULTS ---
            st.success("✅ Workflow Complete! Here is your custom-tailored, data-backed Upwork Profile:")
            st.markdown(result.raw if hasattr(result, 'raw') else str(result))
            
        except Exception as e:
            st.error("The system hit a roadblock. Here is the exact problem:")
            st.error(str(e))
