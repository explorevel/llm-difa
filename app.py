import os
import json
import requests
#from dotenv import load_dotenv

from langchain_core.tools import tool
#from langchain.agents import agent_types, initialize_agent, load_tools
from datetime import datetime, timedelta
from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.memory import ConversationBufferWindowMemory

# Chat module
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.agents import create_tool_calling_agent, AgentExecutor, load_tools

if "sectors_api_key" not in st.session_state:
    st.session_state["sectors_api_key"] = ""
if "groq_api_key" not in st.session_state:
    st.session_state["groq_api_key"] = ""

with st.sidebar:
    SECTORS_API_KEY = st.text_input("Sectors API Key", key="sectors_api_key", type="password")
    GROQ_API_KEY = st.text_input("Groq API Key", key="groq_api_key", type="password")
    button = st.button("Set API Keys")

    if button:
        st.write("API Keys set!")

    st.link_button("Get Sectors API Key", "https://sectors.app/api")
    st.link_button("Get Groq API Key", "https://console.groq.com/keys")

#load_dotenv() #load the .env file

# SECTORS_API_KEY = os.getenv("SECTORS_API_KEY")
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# CALENDAR_API_KEY = os.getenv("CALENDAR_API_KEY")
CALENDAR_API_KEY = st.secrets("CALENDAR_API_KEY")

def retrieve_from_endpoint(url: str) -> dict:
    headers = {"Authorization": SECTORS_API_KEY}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)
    return json.dumps(data)

def fetch_holidays_and_mass_leave(year: int) -> set:
    cache_file = f"nontrading.json"
    
    if os.path.exists(cache_file) and year in [2022, 2023, 2024]:
        with open(cache_file, 'r') as f:
            holiday_dates = {date for date in json.load(f)}
    else:
        url = f"https://calendarific.com/api/v2/holidays?&api_key={CALENDAR_API_KEY}&country=ID&year={year}"
        response = requests.get(url)
        holidays = response.json().get('response', {}).get('holidays', [])
        
        new_date = {(holiday['date']['iso']) 
                         for holiday in holidays 
                         if holiday['type'][0] in ['National holiday', 'Public holiday', 'Observance']}
        
        with open(cache_file, 'w') as f:
            json.dump([str(date) for date in new_date], f)
    return holiday_dates

def is_holiday(date:str) -> bool:
    date = datetime.strptime(date, '%Y-%m-%d').date()
    holiday_date = fetch_holidays_and_mass_leave(date.year)
    return str(date) in holiday_date

def check_trading_day(date) -> bool:
    return not is_holiday(date)

def closest_trading_day(date:str, method = 'forward') -> str:
    temp_date = datetime.strptime(date, '%Y-%m-%d').date()
    if method == 'backward':
        while not check_trading_day(str(temp_date)):
            temp_date = temp_date - timedelta(1)
    if method == 'forward':
        while not check_trading_day(str(temp_date)):
            temp_date = temp_date + timedelta(1)
    return str(temp_date)

def date_handler(start_date:str, end_date:str):
    """
    handle the start and end date.
    """
    temp_start_date = datetime.strptime(closest_trading_day(start_date, 'forward'), '%Y-%m-%d').date()
    temp_end_date = datetime.strptime(closest_trading_day(end_date, 'backward'), '%Y-%m-%d').date()
    if (temp_end_date - temp_start_date).days < 0:
        temp_start_date = temp_end_date
    return str(temp_start_date), str(temp_end_date)


@tool
def get_top_company(classifications: str = "market_cap", top_n: int = 5, year: int = 2023, sub_sector: str = "banks"): 
    """
    Only use this tool if the query specificly ask about top company
    get top company classification such as revenue, total_dividend, dividend_yield, market_cap, earnings. If the classification
    is not specified use revenue or total_dividend as default, and give top 5 company if not specified. please only use sub_sector given here "alternative-energy,apparel-luxury-goods,automobiles-components,banks,basic-materials,consumer-services,financing-service,food-beverage,food-staples-retailing,healthcare-equipment-providers,heavy-constructions-civil-engineering,holding-investment-companies,household-goods,industrial-goods,industrial-services,insurance,investment-service,leisure-goods,logistics-deliveries,media-entertainment,multi-sector-holdings,nondurable-household-products,oil-gas-coal,pharmaceuticals-health-care-research,properties-real-estate,retailing,software-it-services,technology-hardware-equipment,telecommunication,tobacco,transportation,transportation-infrastructure,utilities"
    please only use the classification given. importanly always change dividend into total_dividend. 
    Very important point is to always merge the sub sector word using "-" and in lowercase. If the query about dividen please use 
    total_dividen as classification but if the question is about dividen yield please use dividend_yield
    """
    url = f"https://api.sectors.app/v1/companies/top/?classifications={classifications}&n_stock={top_n}&year={year}&sub_sector={sub_sector}"
    return retrieve_from_endpoint(url)

@tool
def get_company_overview(stock: str) -> str:
    """
    Get company overview including but not limited to: company IPO listing date, 
    market capitalization (market cap), market capitalization (market_cap) ranking, 
    address, website, contact infom listing_board, industry, sub_industry, sector, sub_sector
    """
    url = f"https://api.sectors.app/v1/company/report/{stock}/?sections=overview"

    return retrieve_from_endpoint(url)

@tool
def get_performance_since_ipo(stock: str) -> str:
    """
    Get company performance since its IPO listing date including 7 days percentages change,
    30 days percentages change etc.
    """
    def convert_changes(data):
        # Iterate through each key in the dictionary
        data = json.loads(data)
        for key in data:
            # Check if the key starts with "chg_" and is numeric
            cond1 = key.startswith("chg_")
            cond2 = type(data[key])==int or type(data[key])==float
            if cond1 & cond2:
                # Multiply the value by 100
                data[key] = data[key] * 100
        return data
    
    url = f"https://api.sectors.app/v1/listing-performance/{stock.upper()}/"
    return convert_changes(retrieve_from_endpoint(url))

@tool
def get_top_companies_by_tx_volume(start_date: str, end_date: str, top_n: int = 5) -> str:
    """
    get the top company by transaction volume. This tool will show stock that are most traded in certain period.
    return only top n companies by transaction volume. Always give the highest total volume first in the first row
    followed by second, third and so on.
    """
    message = []
    new_start_date, new_end_date = date_handler(start_date,end_date)
    if new_start_date != start_date or new_end_date != end_date:
        message.append(f"You can't answer given date because it is non-trading day, Now the data show for {new_start_date} - {new_end_date} while mention the reason behind this new_date")
    
    def get_top_trx_average(data_dict, top_n):
        data_dict = json.loads(data_dict)
        # Initialize a dictionary to store the results
        company_data = {}

        # Loop through the data and aggregate volume and price for each company
        for companies in data_dict.values():
            for company in companies:
                name = company['company_name']
                symbol = company['symbol']
                volume = company['volume']
                price = company['price']
                
                # If the company doesn't exist in the dictionary, initialize its entry
                if name not in company_data:
                    company_data[name] = {'symbol': symbol, 'total_volume': 0, 'total_price': 0, 'count': 0}
                
                # Update the company's total volume, total price, and symbol
                company_data[name]['total_volume'] += volume #* price
                company_data[name]['total_price'] += price
                company_data[name]['count'] += 1

        # Calculate the average price for each company
        for name, data in company_data.items():
            data['average_price'] = int(data['total_price'] / data['count'])
            # Remove unnecessary 'count' and 'total_price' fields from the final result
            del data['count']
            del data['total_price']
        # Sort the company data by total_volume in descending order
        sorted_data = dict(sorted(company_data.items(), key=lambda item: item[1]['total_volume'], reverse=True))
        sorted_company_data = dict(list(sorted_data.items())[:top_n])
        return sorted_company_data

        # Print the resulting dictionary
    url = f"https://api.sectors.app/v1/most-traded/?start={new_start_date}&end={new_end_date}&n_stock={top_n}"#&adjusted=True"

    return get_top_trx_average(retrieve_from_endpoint(url), top_n)

@tool
def get_stock_price(stock: str, start_date: str, end_date: str) -> str:
    """
    This tool will give you stock price from start to end_date, by default please use the last 7 days.
    this will help to determine stock margin or stock uptrend or downtrend.
    This tools will give you the closing price of any stock on idx.

    If the result from invoke is ({''}) or retrieve_from_endpoint got char 0 as input please follow this:
    1. invoke again the tool with day after start_date (start_date+1), try for at least 5 times. Remember to mention that the original date requested is not available due to non-trading day.
    2. if after 5 invoke still return empty then give value from nearest trading day.
    3. give the correct result
    """
    message = []
    new_start_date, new_end_date = date_handler(start_date,end_date)
    if new_start_date != start_date or new_end_date != end_date:
        message.append(f"You can't answer given date because it is non-trading day, Now the data show for {new_start_date} - {new_end_date} while mention the reason behind this new_date")

    url = f"https://api.sectors.app/v1/daily/{stock}/?start={new_start_date}&end={new_end_date}"
    return retrieve_from_endpoint(url)

@tool
def get_index_price(index_code: str, start_date: str, end_date: str) -> str:
    """
    This tool will give you IDX index price, by default please use the last 7 days.
    this will help to determine stock margin or stock uptrend or downtrend.
    This tools will give you the closing price of any index on idx
    """
    message = []
    new_start_date, new_end_date = date_handler(start_date,end_date)
    if new_start_date != start_date or new_end_date != end_date:
        message.append(f"You can't answer given date because it is non-trading day, Now the data show for {new_start_date} - {new_end_date} while mention the reason behind this new_date")

    url = f"https://api.sectors.app/v1/index-daily/{index_code.lower()}/?start={new_start_date}&end={end_date}"
    return retrieve_from_endpoint(url)

#lc_tool = load_tools(["ddg-search"])
tools = [
    get_top_company,
    get_company_overview,
    get_performance_since_ipo,
    get_top_companies_by_tx_volume,
    get_stock_price,
    get_index_price,
    DuckDuckGoSearchResults(name="search")
]

memory = ConversationBufferWindowMemory(
    memory_key="chat_history", k=5, human_prefix="human", ai_prefix="ai"
)

# # Chat module
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_groq import ChatGroq
# from langchain.agents import create_tool_calling_agent, AgentExecutor
# almost get to prompt: "Al check the start_date if it belong to the weekend than answers using last weekday but still mention that it use last weekday data."
# ex prompt "Whenever you return a list of names, return also the corresponding values for each name and the date."
# ex again: Importantly for get_top_companies_by_tx_volume always give answersordered by total volume desceding.

@st.cache_resource
def LLM_Chat():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""
                you are IDX digital financial assistant called Difa. Dont give buy or any
                financial advise since you are not an advisor, just give answers.
                Answer the following queries, being as factual and analytical 
                as you can. If you need the start and end dates but they are not 
                explicitly provided, infer from the query. If the volume was about 
                a single day, the start and end parameter should be the same. if the 
                start and end date not specified just give last week to {datetime.today()}. 
                current year is {datetime.now().year}. If the answers contain a lot of 
                number or data please give the answers with markdown table else in string.
                Please use commas when showing answers with large number, to make it easier
                for reader to read. only use DuckDuckGoSearchResults as last resource.
                """,
            ),
            ("ai", "{chat_history}"),
            ("human", "{input}"),
            # msg containing previous agent tool invocations
            # and corresponding tool outputs
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    llm = ChatGroq(
        temperature=0.1,
        model_name="llama-3.1-70b-versatile",
        groq_api_key=GROQ_API_KEY,
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory, handle_parsing_errors=True)
    return agent_executor

query_1 = "What are the top 3 companies by transaction volume over the last 7 days?"
query_2 = "Based on the closing prices of BBCA between 1st and 30th of June 2024, are we seeing an uptrend or downtrend? Try to explain why."
query_3 = "What is the company with the largest market cap between BBCA and BREN? For said company, retrieve the email, phone number, listing date and website for further research."
query_4 = "What is the performance of GOTO (symbol: GOTO) since its IPO listing?"
query_5 = "If i had invested into GOTO vs BREN on their respective IPO listing date, which one would have given me a better return over a 90 day horizon?"
query_6 = "What are the top 5 companies by transaction volume on the first of this month?"
query_7 = "What are the most traded stock yesterday?"
query_8 = "What are the top 7 most traded stocks between 6th June to 10th June this year?"


queries = [query_1, query_2, query_3, query_4, query_5, query_6, query_7, query_8]

# for query in queries:
#     print("Question:", query)
#     result = agent_executor.invoke({"input": query})
#     print("Answer:", "\n", result["output"], "\n\n======\n\n")

# print(datetime.today())

st.title("🇲🇨 - DIFA (Digital IDX Financial Assistant)")
st.markdown("Ask me anything about IDX!")
st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input():
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    if (st.session_state["sectors_api_key"] == "" or st.session_state["groq_api_key"] == ""):
        st.error("Please set your API Keys first before using the chatbot.")
        st.stop()

    with st.chat_message("assistant"):
        with st.status("🧠 thinking...") as status:
            st_callback = StreamlitCallbackHandler(st.container())
            model = LLM_Chat()
            response = None
            try:
                response = model.invoke({"input": prompt}, callback_handler=st_callback)
                print(response)
                status.update(label="💡 answers ready!", state="complete", expanded=False)
            except Exception as e:
                st.error(
                    f"Something wrong happened. Please try again later. Reason: {type(e).__name__}",
                    icon="🚨",
                )
                print(e)
        if response:
            st.write(response["output"])
            st.session_state.messages.append(
                {"role": "assistant", "content": response["output"]}
            )
        else:
            st.error(
                f"Something wrong happened. Please try again later.",
                icon="🚨",
            )
