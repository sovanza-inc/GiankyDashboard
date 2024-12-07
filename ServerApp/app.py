# Import Streamlit first
import streamlit as st
import traceback
import json  # Add this near other import statements

# Streamlit configuration (MUST be the first Streamlit command)
st.set_page_config(
    page_title="Gianky NFT Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import other required libraries
import os
import sys
import web3
from web3 import Web3
from eth_utils import to_checksum_address, is_address
from typing import Dict, Any
import pandas as pd
import plotly.graph_objects as go
import logging
import requests
import pandas as pd
from datetime import datetime
import plotly.express as px
import time
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# PolygonScan API Configuration
POLYGONSCAN_API_KEY = os.getenv('POLYGONSCAN_API_KEY', '')

# Inject custom CSS
def inject_custom_css():
    """Inject custom CSS styles into the Streamlit app"""
    try:
        with open('styles.css', 'r') as f:
            css = f.read()
        st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error loading custom CSS: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dashboard_debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    # Inject custom CSS after page config
    inject_custom_css()
    
    # Custom sidebar navigation with icons
    try:
        # Use logo from root directory
        logo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logo.png')
        if os.path.exists(logo_path):
            st.sidebar.image(logo_path, use_column_width=True)
        else:
            logger.warning(f"Logo not found at {logo_path}")
    except Exception as e:
        st.sidebar.markdown("### Gianky NFT")
        logger.error(f"Error loading logo: {e}")
    
    # Enhanced navigation with icons
    nav_options = [
        " Contract Overview", 
        " Address Lookup", 
        " Top Holders", 
        " NFT Collection"
    ]
    
    # Create custom sidebar with improved styling
    page = st.sidebar.radio(
        "Navigation", 
        nav_options, 
        key="main_navigation",
        help="Select a dashboard section"
    )
    
    # Page routing with loading animations
    with st.spinner("Loading Dashboard..."):
        time.sleep(0.5)  # Simulated loading
        
        if "Contract Overview" in page:
            render_contract_page()
        elif "Address Lookup" in page:
            render_address_lookup()
        elif "Top Holders" in page:
            render_holders_page()
        elif "NFT Collection" in page:
            render_nft_collection()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### Gianky NFT Dashboard
    *Powered by Blockchain Alpha*
    

    """)

# Load contract ABI with enhanced error handling
def load_contract_abi(file_path):
    try:
        with open(file_path, 'r') as f:
            abi = json.load(f)
        logger.info(f"Successfully loaded contract ABI from {file_path}")
        return abi
    except FileNotFoundError:
        logger.error(f"Contract ABI file not found: {file_path}")
        st.error(f"Contract ABI file not found: {file_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in contract ABI: {e}")
        st.error(f"Invalid JSON in contract ABI: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error loading contract ABI: {e}")
        st.error(f"Unexpected error loading contract ABI: {e}")
        sys.exit(1)

# Load contract ABI
abi_path = os.path.join(os.path.dirname(__file__), 'contract_abi.json')
CONTRACT_ABI = load_contract_abi(abi_path)

# Initialize Web3 and contract with comprehensive error handling
def initialize_web3_contract(rpc_url, contract_address, contract_abi):
    try:
        # Validate RPC URL
        if not rpc_url or not rpc_url.startswith(('http://', 'https://')):
            raise ValueError(f"Invalid RPC URL: {rpc_url}")
        
        # Initialize Web3
        w3 = Web3(Web3.HTTPProvider(rpc_url))
        
        # Validate contract address
        if not is_address(contract_address):
            raise ValueError(f"Invalid contract address format: {contract_address}")
        
        contract_address = to_checksum_address(contract_address)
        
        # Check connection
        try:
            w3.eth.block_number
        except Exception as e:
            raise ConnectionError(f"Failed to connect to RPC endpoint: {rpc_url}. Error: {e}")
        
        # Initialize contract
        contract = w3.eth.contract(address=contract_address, abi=contract_abi)
        
        logger.info(f"Successfully initialized Web3 and contract. Connected to {rpc_url}")
        return w3, contract
    
    except Exception as e:
        error_msg = f"Web3 Initialization Error: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        st.error(error_msg)
        sys.exit(1)

# RPC and Contract Configuration
RPC_URL = os.getenv('RPC_URL', 'https://polygon-rpc.com')
CONTRACT_ADDRESS = os.getenv('CONTRACT_ADDRESS', '0xdc91E2fD661E88a9a1bcB1c826B5579232fc9898')

# Initialize Web3 and Contract
w3, contract = initialize_web3_contract(RPC_URL, CONTRACT_ADDRESS, CONTRACT_ABI)

# Contract tier constants
TIER_MAPPING = {
    1: "Starter",
    2: "Basic", 
    3: "Diamond",
    4: "VIP"
}

def format_address(address: str) -> str:
    """Format address for display"""
    return f"{address[:6]}...{address[-4:]}" if address and isinstance(address, str) else "N/A"

def safe_contract_call(func, *args, **kwargs):
    """
    Safely call contract functions with comprehensive error handling
    
    :param func: Contract function to call
    :param args: Positional arguments for the function
    :param kwargs: Keyword arguments for the function
    :return: Function result or None if error occurs
    """
    try:
        result = func(*args, **kwargs)
        return result
    except Exception as e:
        error_msg = f"Contract Call Error: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return None

def get_contract_stats():
    """Get all readable contract statistics with robust error handling"""
    basic_info = {
        "name": safe_contract_call(contract.functions.name().call) or "N/A",
        "symbol": safe_contract_call(contract.functions.symbol().call) or "N/A",
        "total_supply": safe_contract_call(contract.functions.totalSupply().call) or 0,
        "owner": safe_contract_call(contract.functions.owner().call) or "N/A"
    }
    
    # Log contract statistics for debugging
    logger.info(f"Contract Statistics: {basic_info}")
    
    # Additional optional methods with error handling
    optional_methods = ["giankyToken", "splitter"]
    for method in optional_methods:
        try:
            basic_info[method] = getattr(contract.functions, method)().call()
        except Exception as e:
            basic_info[method] = f"Error: {str(e)}"
            logger.warning(f"Could not retrieve {method}: {e}")
    
    return basic_info

def get_contract_overview():
    """Retrieve comprehensive overview of the Gianky NFT contract"""
    try:
        # Retrieve various contract constants and metrics
        overview_data = {
            "Contract Address": CONTRACT_ADDRESS,
            "Network": "Polygon",
            
            # Token Pricing Constants
            "Basic Price": safe_contract_call(contract.functions.BASIC_PRICE().call),
            "Diamond Price": safe_contract_call(contract.functions.DIAMOND_PRICE().call),
            "Premium Price": safe_contract_call(contract.functions.PREMIUM_PRICE().call),
            "Standard Price": safe_contract_call(contract.functions.STANDARD_PRICE().call),
            "Starter Price": safe_contract_call(contract.functions.STARTER_PRICE().call),
            "VIP Price": safe_contract_call(contract.functions.VIP_PRICE().call),
            "Gianky Reward Rate": safe_contract_call(contract.functions.GIANKY_REWARD_RATE().call),
            
            # Existing metrics
            "Total Supply": safe_contract_call(contract.functions.totalSupply().call),
            "Minted Tokens": safe_contract_call(contract.functions.totalSupply().call),
        }
        
        # Format the overview for display
        overview_text = " Gianky NFT Contract Overview\n\n"
        overview_text += " Contract Details:\n"
        overview_text += f"   - Address: {format_address(overview_data['Contract Address'])}\n"
        overview_text += f"   - Network: {overview_data['Network']}\n\n"
        
        overview_text += " Pricing Structure:\n"
        price_mapping = {
            "Basic Price": overview_data['Basic Price'],
            "Diamond Price": overview_data['Diamond Price'],
            "Premium Price": overview_data['Premium Price'],
            "Standard Price": overview_data['Standard Price'],
            "Starter Price": overview_data['Starter Price'],
            "VIP Price": overview_data['VIP Price']
        }
        
        for price_type, price_value in price_mapping.items():
            overview_text += f"   - {price_type}: {price_value} MATIC\n"
        
        overview_text += f"\n Gianky Reward Rate: {overview_data['Gianky Reward Rate']}\n"
        overview_text += f"\n Token Metrics:\n"
        overview_text += f"   - Total Supply: {overview_data['Total Supply']}\n"
        overview_text += f"   - Minted Tokens: {overview_data['Minted Tokens']}\n"
        
        return overview_text
    
    except Exception as e:
        logger.error(f"Error in contract overview: {str(e)}")
        logger.error(traceback.format_exc())
        return f"Error retrieving contract overview: {str(e)}"

def get_top_holders():
    """Get top NFT holders using Transfer events with standard Web3.py method"""
    try:
        logger.info("Attempting to fetch Transfer events...")
        
        # Get the latest block
        latest_block = w3.eth.block_number
        logger.info(f"Latest block: {latest_block}")
        
        # Prepare event signature
        transfer_event_signature = w3.keccak(text="Transfer(address,address,uint256)").hex()
        
        # Prepare filter parameters
        from_block = max(0, latest_block - 1000)
        to_block = latest_block
        
        # Fetch logs directly using eth_getLogs
        logs = w3.eth.get_logs({
            'fromBlock': from_block,
            'toBlock': to_block,
            'address': contract.address,  # Changed from CONTRACT_ADDRESS
            'topics': [transfer_event_signature]
        })
        
        logger.info(f"Found {len(logs)} Transfer event logs")
        
        # Track current owners
        holder_balances = {}
        zero_address = "0x0000000000000000000000000000000000000000"
        
        # Process logs
        for log in logs:
            try:
                # Decode event topics
                from_address = '0x' + log['topics'][1].hex()[-40:]
                to_address = '0x' + log['topics'][2].hex()[-40:]
                
                # Handle minting (from zero address)
                if from_address.lower() == zero_address.lower():
                    if to_address not in holder_balances:
                        holder_balances[to_address] = 0
                    holder_balances[to_address] += 1
                
                # Handle transfers
                else:
                    # Decrease previous owner's balance
                    if from_address in holder_balances:
                        holder_balances[from_address] = max(0, holder_balances[from_address] - 1)
                        if holder_balances[from_address] == 0:
                            del holder_balances[from_address]
                    
                    # Increase new owner's balance
                    if to_address.lower() != zero_address.lower():
                        if to_address not in holder_balances:
                            holder_balances[to_address] = 0
                        holder_balances[to_address] += 1
            
            except Exception as e:
                logger.error(f"Error processing event log: {e}")
                continue
        
        if not holder_balances:
            logger.warning("No valid holders found in recent events")
            return "No tokens minted yet or unable to retrieve token information"
        
        processed_tokens = sum(holder_balances.values())
        logger.info(f"Total processed tokens: {processed_tokens}")
        
        # Sort holders by balance and filter top 10
        sorted_holders = sorted(holder_balances.items(), key=lambda x: (x[1], x[0].lower()), reverse=True)[:10]
        
        # Format output with more details
        holders_text = f"Top 10 NFT Holders (Based on last {to_block - from_block} blocks)\n"
        holders_text += f"Total Tracked Tokens: {processed_tokens}\n\n"
        
        for rank, (addr, balance) in enumerate(sorted_holders, 1):
            percentage = (balance / processed_tokens) * 100
            holders_text += f"{rank}. {format_address(addr)}: {balance} NFTs ({percentage:.1f}%)\n"
        
        return holders_text
        
    except Exception as e:
        error_msg = f"Error fetching holders from events: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return error_msg

def lookup_address(address: str):
    """Get all information about an address"""
    try:
        balance = safe_contract_call(contract.functions.balanceOf(address).call)
        owned_tokens = []
        
        total_supply = safe_contract_call(contract.functions.totalSupply().call)
        for token_id in range(1, total_supply + 1):
            try:
                if safe_contract_call(contract.functions.ownerOf(token_id).call).lower() == address.lower():
                    token_uri = safe_contract_call(contract.functions.tokenURI(token_id).call)
                    owned_tokens.append({
                        "id": token_id,
                        "uri": token_uri[:100] + "..." if len(token_uri) > 100 else token_uri
                    })
            except Exception as e:
                logger.error(f"Error looking up token #{token_id}: {str(e)}")
                continue

        return {
            "address": address,
            "balance": balance,
            "owned_tokens": owned_tokens
        }
    except Exception as e:
        logger.error(f"Error looking up address {address}: {str(e)}")
        return None

# Add Polygonscan API configuration
POLYGONSCAN_API_KEY = "1C4WZUR9A92W6Y24SJ7DMW9Q4MM6928JPJ"
POLYGONSCAN_API_URL = "https://api.polygonscan.com/api"

def get_top_holders_polygonscan():
    """Get top 15 NFT holders using Polygonscan API"""
    try:
        logger.info("Fetching top holders from Polygonscan...")
        
        # Prepare API request parameters for NFT tokens
        params = {
            'module': 'account',
            'action': 'tokennfttx',  # NFT transactions
            'contractaddress': CONTRACT_ADDRESS,
            'apikey': POLYGONSCAN_API_KEY,
            'page': 1,
            'offset': 100,  # Get more transactions to analyze
            'sort': 'desc'
        }
        
        # Make API request
        response = requests.get(POLYGONSCAN_API_URL, params=params)
        data = response.json()
        
        if data['status'] == '1' and data['message'] == 'OK':
            transactions = data['result']
            
            # Process transactions to determine current holders
            holders = {}
            processed_tokens = set()
            
            # Process transactions from newest to oldest
            for tx in transactions:
                token_id = int(tx['tokenID'])
                
                # Skip if we've already found the current holder of this token
                if token_id in processed_tokens:
                    continue
                
                to_address = tx['to'].lower()
                
                # Add to holders count
                if to_address not in holders:
                    holders[to_address] = {'balance': 0, 'last_updated': tx['timeStamp']}
                holders[to_address]['balance'] += 1
                
                # Mark token as processed
                processed_tokens.add(token_id)
            
            # Convert to list and sort by balance
            processed_holders = [
                {
                    'address': addr,
                    'balance': data['balance'],
                    'last_updated': datetime.fromtimestamp(int(data['last_updated']))
                }
                for addr, data in holders.items()
                if addr != '0x0000000000000000000000000000000000000000'  # Exclude zero address
            ]
            
            # Sort by balance and get top 15
            processed_holders.sort(key=lambda x: x['balance'], reverse=True)
            processed_holders = processed_holders[:15]
            
            # Calculate total tokens
            total_tokens = sum(holder['balance'] for holder in processed_holders)
            
            # Create DataFrame for visualization
            df = pd.DataFrame(processed_holders)
            if not df.empty:
                df['percentage'] = (df['balance'] / total_tokens * 100)
            
                # Format output
                holders_text = " Top 15 NFT Holders (via Polygonscan)\n\n"
                
                # Display holders with additional metrics
                for idx, row in df.iterrows():
                    holders_text += f"{idx + 1}. {format_address(row['address'])}\n"
                    holders_text += f"   Balance: {row['balance']} NFTs ({row['percentage']:.2f}%)\n"
                    holders_text += f"   Last Updated: {row['last_updated'].strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                
                # Add summary statistics
                holders_text += f"\n Summary Statistics:\n"
                holders_text += f"Total Tracked Tokens: {total_tokens}\n"
                holders_text += f"Average Holdings: {total_tokens / len(processed_holders):.2f} NFTs\n"
                
                return holders_text, df
            else:
                return "No holders found in recent transactions.", None
            
        else:
            error_msg = f"Polygonscan API Error: {data.get('message', 'Unknown error')} (Status: {data.get('status', 'Unknown')})"
            logger.error(error_msg)
            if 'result' in data:
                logger.error(f"API Response: {data['result']}")
            return error_msg, None
            
    except Exception as e:
        error_msg = f"Error fetching holders from Polygonscan: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return error_msg, None

def render_holders_page():
    """Render the Top Holders page with Polygonscan data"""
    st.title(" Top NFT Holders")

    # Add data source selector
    data_source = st.radio(
        "Select Data Source",
        ["Polygonscan API"],
        help="Choose the source for holders data"
    )
    
    if st.button("Fetch Holders Data"):
        with st.spinner("Retrieving holders data..."):
            if data_source == "Polygonscan API":
                holders_text, holders_df = get_top_holders_polygonscan()
                
                if holders_df is not None:
                    # Calculate summary metrics
                    total_tokens = holders_df['balance'].sum()
                    avg_holdings = total_tokens / len(holders_df)
                    max_balance = holders_df['balance'].max()
                    
                    # Display summary metrics in columns
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Tracked Tokens", f"{total_tokens:,}")
                    with col2:
                        st.metric("Average Holdings", f"{avg_holdings:.2f}")
                    with col3:
                        st.metric("Largest Holder", f"{max_balance:,}")
                    
                    # Create a more appealing table
                    st.subheader(" Holder Rankings")
                    
                    # Format the data for display
                    display_df = holders_df.copy()
                    display_df['Rank'] = range(1, len(display_df) + 1)
                    display_df['Address'] = display_df['address'].apply(format_address)
                    display_df['Balance'] = display_df['balance'].apply(lambda x: f"{x:,}")
                    display_df['Percentage'] = display_df['percentage'].apply(lambda x: f"{x:.2f}%")
                    display_df['Last Updated'] = display_df['last_updated'].dt.strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Reorder columns and rename them
                    display_df = display_df[['Rank', 'Address', 'Balance', 'Percentage', 'Last Updated']]
                    
                    # Apply custom styling
                    st.dataframe(
                        display_df,
                        column_config={
                            "Rank": st.column_config.NumberColumn(
                                " Rank",
                                help="Position in holder rankings",
                                format="%d"
                            ),
                            "Address": st.column_config.TextColumn(
                                " Address",
                                help="Wallet address of the holder",
                                width="medium"
                            ),
                            "Balance": st.column_config.TextColumn(
                                " Balance",
                                help="Number of NFTs held"
                            ),
                            "Percentage": st.column_config.TextColumn(
                                " Share",
                                help="Percentage of total supply"
                            ),
                            "Last Updated": st.column_config.TextColumn(
                                " Last Updated",
                                help="Time of last transaction"
                            )
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                    
                    # Add visualizations
                    st.subheader(" Holdings Distribution")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Bar chart of holdings
                        fig = px.bar(
                            holders_df,
                            x=holders_df.index + 1,
                            y='balance',
                            title='NFT Holdings by Rank',
                            labels={'x': 'Holder Rank', 'balance': 'Number of NFTs'},
                            hover_data=['percentage']
                        )
                        fig.update_layout(
                            showlegend=False,
                            hovermode='x unified',
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Pie chart of holdings distribution
                        fig_pie = px.pie(
                            holders_df,
                            values='balance',
                            names=holders_df['address'].apply(format_address),
                            title='Holdings Distribution by Address',
                            hole=0.4
                        )
                        fig_pie.update_layout(
                            showlegend=False,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)'
                        )
                        fig_pie.update_traces(textposition='inside')
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                else:
                    st.error(holders_text)

def render_contract_page():
    """Render the Contract Overview page with automatic stats"""
    st.title(" Gianky NFT Contract Overview")
    
    # Predefined tier prices and mapping
    TIER_PRICES = {
        'Starter': 20,
        'Basic': 50,
        'Standard': 100,
        'VIP': 500,
        'Premium': 1000,
        'Diamond': 5000
    }
    
    # Exact contract constant mapping
    PRICE_CONSTANTS = {
        'Starter': 'STARTER_PRICE',
        'Basic': 'BASIC_PRICE',
        'Standard': 'STANDARD_PRICE',
        'VIP': 'VIP_PRICE',
        'Premium': 'PREMIUM_PRICE',
        'Diamond': 'DIAMOND_PRICE'
    }
    
    # Fetch contract stats on page load
    try:
        # Fetch contract statistics
        total_minted = contract.functions.totalSupply().call()
        
        # Fetch and convert prices
        prices = {}
        for tier, const_name in PRICE_CONSTANTS.items():
            # Hardcode Starter price to 20 POL
            if tier == 'Starter':
                price_pol = 20
                logger.info("Starter price hardcoded to 20 POL")
                prices[tier] = price_pol
                continue
            
            # Default to predefined price
            price_pol = TIER_PRICES[tier]
            
            try:
                # Dynamically call the function
                price_wei = getattr(contract.functions, const_name)().call()
                
                # Convert from Wei to Polygon (POL)
                retrieved_price_pol = w3.from_wei(price_wei, 'ether')
                
                # Use retrieved price if it's not zero
                if retrieved_price_pol > 0:
                    price_pol = retrieved_price_pol
                
                # Log successful price retrieval
                logger.info(f"Successfully retrieved {tier} price: {price_pol} POL")
            
            except Exception as e:
                # Log the retrieval failure
                logger.warning(f"Failed to retrieve {tier} price: {str(e)}")
            
            # Store the final price
            prices[tier] = price_pol
        
        # Detailed tier-specific details and benefits
        tier_details = {
            "Starter": """
            Entry-level NFT with basic ecosystem access
            - Limited cashback in Gianky Coin
            - Basic referral rewards
            - Initial platform engagement
            """,
            "Basic": """
            Expanded access to Gianky ecosystem
            - Increased cashback percentage
            - Enhanced referral rewards
            - More platform features
            """,
            "Standard": """
            Comprehensive platform participation
            - Higher cashback rates
            - Advanced referral reward levels
            - Exclusive website offers
            """,
            "VIP": """
            Premium ecosystem engagement
            - Significant cashback rewards
            - Multi-level referral system access
            - Priority platform features
            """,
            "Premium": """
            Advanced platform integration
            - Maximum cashback potential
            - Extensive referral reward structure
            - Exclusive discounts and offers
            """,
            "Diamond": """
            Ultimate Gianky ecosystem membership
            - Highest cashback percentage
            - Top-tier referral rewards
            - Complete platform access
            - Priority in future developments
            """
        }
        
        # Detailed benefits for each tier
        benefits = {
            "Starter": [
                "Basic NFT Purchase",
                "Initial Gianky Coin Cashback",
                "Entry-level Referral Rewards"
            ],
            "Basic": [
                "Enhanced Cashback Rewards",
                "Expanded Referral Program",
                "Website Discount Eligibility"
            ],
            "Standard": [
                "Increased Cashback Percentage",
                "Advanced Referral Tracking",
                "Exclusive Platform Offers"
            ],
            "VIP": [
                "Significant Cashback Rewards",
                "Multi-level Referral System",
                "Premium Platform Access"
            ],
            "Premium": [
                "Maximum Cashback Potential",
                "Extensive Referral Rewards",
                "Priority Support",
                "Exclusive Discounts"
            ],
            "Diamond": [
                "Highest Cashback Rate",
                "Top-tier Referral Rewards",
                "Complete Ecosystem Access",
                "Future Development Priority"
            ]
        }
        
        # Pricing Tabs
        pricing_tabs = st.tabs([
            "Starter", "Basic", "Standard", 
            "VIP", "Premium", "Diamond"
        ])
        
        # Detailed pricing information for each tier
        ordered_tiers = [
            'Starter', 'Basic', 'Standard', 
            'VIP', 'Premium', 'Diamond'
        ]
        
        for i, tier in enumerate(ordered_tiers):
            with pricing_tabs[i]:
                # Get price, fallback to predefined price
                price = prices.get(tier, TIER_PRICES[tier])
                
                # Create columns for tier details
                col1, col2 = st.columns(2)
                
                with col1:
                    # Tier Price Card
                    st.metric(
                        f"{tier} Tier Price", 
                        f"{price:.0f} POL",
                        help=f"Official price for {tier} NFT tier"
                    )
                    
                    # Tier description
                    st.markdown(tier_details.get(tier, "Tier description not available"))
                
                with col2:
                    # Create a dataframe of benefits
                    benefits_df = pd.DataFrame({
                        'Benefit': benefits.get(tier, [])
                    })
                    
                    # Display benefits
                    st.dataframe(
                        benefits_df,
                        column_config={
                            "Benefit": st.column_config.TextColumn(
                                "Tier Benefits",
                                help=f"Benefits included in {tier} tier"
                            )
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                
                # Additional ecosystem information
                st.info("""
                **Gianky Ecosystem Benefits:**
                - Cashback in Gianky Coin
                - Multi-level Referral Rewards
                - Royalties from Transactions
                - Exclusive Website Offers
                - Discounted Rates with Gianky Coin
                - Future NFT Staking Potential
                """)
        
        # Divider before visualizations
        st.markdown("---")
        
        # Create two columns for visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Enhanced Bar Chart with Plotly
            price_data = pd.DataFrame([
                {'NFT Tier': tier, 'Price': prices.get(tier, TIER_PRICES[tier])} 
                for tier in ordered_tiers
            ])
            
            fig_bar = px.bar(
                price_data, 
                x='NFT Tier', 
                y='Price', 
                title='NFT Tier Pricing',
                labels={'Price': 'Price in POL'},
                color='NFT Tier',
                color_discrete_sequence=px.colors.sequential.Plasma_r
            )
            
            fig_bar.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                title_font_size=16,
                title_x=0.5,
                xaxis_title='',
                yaxis_title='Price (POL)',
                height=400
            )
            
            fig_bar.update_traces(
                marker_line_color='rgb(8,48,107)',
                marker_line_width=1.5,
                opacity=0.8
            )
            
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            # Pie Chart for Price Distribution
            fig_pie = px.pie(
                price_data, 
                values='Price', 
                names='NFT Tier', 
                title='Price Distribution Across Tiers',
                color='NFT Tier',
                color_discrete_sequence=px.colors.sequential.Plasma_r
            )
            
            fig_pie.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                title_font_size=16,
                title_x=0.5
            )
            
            fig_pie.update_traces(
                textposition='inside', 
                textinfo='percent+label',
                pull=[0.1 if price > 500 else 0 for price in price_data['Price']],
                marker=dict(line=dict(color='#000000', width=2))
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Additional Insights Section
        st.subheader(" Pricing Insights")
        
        # Calculate pricing insights
        total_price = price_data['Price'].sum()
        avg_price = price_data['Price'].mean()
        max_price = price_data['Price'].max()
        min_price = price_data['Price'].min()
        price_range = max_price - min_price
        
        # Create columns for insights
        insight_cols = st.columns(3)
        
        # Prepare detailed insights
        insights = [
            (f"Total Ecosystem Pricing", f"{total_price:.0f} POL", 
             "Cumulative value of all NFT tiers"),
            (f"Average Tier Price", f"{avg_price:.0f} POL", 
             "Mean price across all NFT tiers"),
            (f"Price Range", f"{price_range:.0f} POL", 
             "Difference between highest and lowest tier")
        ]
        
        # Display insights with expanded information
        for col, (label, value, description) in zip(insight_cols, insights):
            with col:
                st.metric(label, value)
                with st.expander("More Details"):
                    st.markdown(description)
        
        # Advanced pricing analysis
        st.subheader(" Pricing Strategy Analysis")
        
        # Prepare pricing strategy insights
        strategy_insights = [
            {
                "Aspect": "Tier Progression",
                "Description": "Systematic price scaling to reward platform engagement",
                "Key Points": [
                    f"Lowest Tier: {min_price:.0f} POL",
                    f"Highest Tier: {max_price:.0f} POL",
                    f"Price Multiplier: {max_price/min_price:.1f}x"
                ]
            },
            {
                "Aspect": "Ecosystem Value Proposition",
                "Description": "Structured pricing to incentivize platform participation",
                "Key Points": [
                    "Incremental benefits with each tier",
                    "Designed to encourage user progression",
                    "Aligned with platform growth strategy"
                ]
            },
            {
                "Aspect": "Investment Potential",
                "Description": "NFT tiers as strategic investment vehicles",
                "Key Points": [
                    "Increasing rewards with higher tiers",
                    "Potential for future value appreciation",
                    "Multiple revenue streams"
                ]
            }
        ]
        
        # Display strategy insights
        for insight in strategy_insights:
            with st.expander(f" {insight['Aspect']}"):
                st.markdown(f"**Description:** {insight['Description']}")
                st.markdown("**Key Insights:**")
                for point in insight['Key Points']:
                    st.markdown(f"- {point}")
        
        # Pricing Distribution Visualization
        st.subheader(" Pricing Distribution")
        
        # Create a more detailed distribution plot
        fig_dist = px.histogram(
            price_data, 
            x='NFT Tier', 
            y='Price', 
            title='NFT Tier Price Distribution',
            labels={'Price': 'Price in POL'},
            color='NFT Tier',
            color_discrete_sequence=px.colors.sequential.Plasma_r
        )
        
        fig_dist.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font_size=16,
            title_x=0.5,
            xaxis_title='NFT Tier',
            yaxis_title='Price (POL)',
            height=400
        )
        
        fig_dist.update_traces(
            marker_line_color='rgb(8,48,107)',
            marker_line_width=1.5,
            opacity=0.8
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Final ecosystem context
        st.info("""
        **Gianky NFT Pricing Strategy**
        - Tiered approach to platform engagement
        - Incremental benefits with increasing investment
        - Designed to support long-term ecosystem growth
        """)
        
    except Exception as e:
        st.error(f"Error fetching contract statistics: {str(e)}")
        logger.error(f"Contract stats error: {traceback.format_exc()}")

def render_address_lookup():
    """Enhanced Address Lookup Page with Comprehensive Analytics"""
    st.title(" Address Lookup")
    
    # Address input with validation
    address = st.text_input(
        "Enter Wallet Address", 
        placeholder="0x...",
        help="Enter a valid Ethereum/Polygon wallet address"
    )
    
    # Validation and lookup button
    col1, col2 = st.columns([3, 1])
    with col1:
        lookup_type = st.selectbox(
            "Lookup Type",
            [
                "Comprehensive Overview"
            ]
        )
    
    with col2:
        lookup_button = st.button("Lookup", use_container_width=True)
    
    # Lookup process
    if lookup_button and address:
        try:
            # Validate address
            if not Web3.is_address(address):
                st.error("Invalid wallet address. Please check and try again.")
                return
            
            # Fetch address information
            info = lookup_address(address)
            
            if not info:
                st.warning("No information found for this address.")
                return
            
            # Main information display
            st.subheader(f" Address: {format_address(address)}")
            
            # Create tabs for comprehensive overview
            tabs = st.tabs([
                "NFT Overview", 
                "Rewards", 
                "Transaction History"
            ])
            
            with tabs[0]:  # NFT Overview
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Total NFTs", 
                        f"{info.get('balance', 0):,}",
                        help="Number of NFTs owned"
                    )
                
                with col2:
                    referrer_count = sum(1 for x in info.get('owned_tokens', []) if x.get('referrer', None))
                    st.metric(
                        "NFTs with Referrer", 
                        f"{referrer_count:,}",
                        help="NFTs with parent referrer"
                    )
                
                with col3:
                    st.metric(
                        "Total Value", 
                        f"{info.get('total_value', 0):.2f} POL",
                        help="Estimated total value of NFT holdings"
                    )
                
                # NFT Holdings Visualization
                if info.get('owned_tokens'):
                    # Prepare tier distribution data
                    tier_counts = {}
                    for token in info['owned_tokens']:
                        tier = token.get('tier', 'Unknown')
                        tier_counts[tier] = tier_counts.get(tier, 0) + 1
                    
                    # Create pie chart for tier distribution
                    tier_df = pd.DataFrame.from_dict(
                        tier_counts, 
                        orient='index', 
                        columns=['Count']
                    ).reset_index()
                    tier_df.columns = ['Tier', 'Count']
                    
                    fig_tier_dist = px.pie(
                        tier_df, 
                        values='Count', 
                        names='Tier', 
                        title='NFT Tier Distribution',
                        color='Tier',
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    
                    fig_tier_dist.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        title_x=0.5
                    )
                    
                    st.plotly_chart(fig_tier_dist, use_container_width=True)
            
            with tabs[1]:  # Rewards
                st.subheader(" Rewards Summary")
                
                # Comprehensive rewards metrics
                rewards_cols = st.columns(3)
                
                rewards = {
                    "Total Cashback": f"{info.get('total_cashback', 0):.2f} Gianky Coins",
                    "Referral Rewards": f"{info.get('referral_rewards', 0):.2f} Gianky Coins",
                    "Royalty Earnings": f"{info.get('royalty_earnings', 0):.2f} Gianky Coins"
                }
                
                for col, (label, value) in zip(rewards_cols, rewards.items()):
                    with col:
                        st.metric(label, value)
            
            with tabs[2]:  # Transaction History
                render_transaction_history(address)
        
        except Exception as e:
            st.error(f"Error processing address: {str(e)}")
            logger.error(f"Address lookup error: {traceback.format_exc()}")
    
    # Additional guidance
    st.markdown("""
    ### Gianky Ecosystem Insights
    - Track your NFT holdings and rewards
    - Monitor referral performance
    - Understand your engagement level
    """)

def fetch_transaction_history(address, contract_address=None, max_transactions=50):
    """
    Fetch transaction history for a given address from PolygonScan API
    
    Args:
        address (str): Wallet address to fetch transactions for
        contract_address (str, optional): Specific contract address to filter transactions
        max_transactions (int, optional): Maximum number of transactions to retrieve
    
    Returns:
        list: Detailed transaction history
    """
    if not POLYGONSCAN_API_KEY:
        st.warning("PolygonScan API key not configured. Transaction history unavailable.")
        return []
    
    base_url = "https://api.polygonscan.com/api"
    
    # Prepare query parameters
    params = {
        "module": "account",
        "action": "txlist",
        "address": address,
        "startblock": 0,
        "endblock": 99999999,
        "page": 1,
        "offset": max_transactions,
        "sort": "desc",
        "apikey": POLYGONSCAN_API_KEY
    }
    
    # Optional contract address filtering
    if contract_address:
        params.update({
            "action": "tokentx",
            "contractaddress": contract_address
        })
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if data['status'] == '1' and data['message'] == 'OK':
            transactions = data['result']
            
            # Process and enrich transaction data
            processed_transactions = []
            for tx in transactions:
                processed_tx = {
                    "hash": tx.get('hash', 'N/A'),
                    "from": tx.get('from', 'N/A'),
                    "to": tx.get('to', 'N/A'),
                    "value": float(tx.get('value', 0)) / 10**18,  # Convert from Wei to MATIC
                    "gas": float(tx.get('gas', 0)),
                    "gas_price": float(tx.get('gasPrice', 0)) / 10**9,  # Convert to Gwei
                    "timestamp": datetime.fromtimestamp(int(tx.get('timeStamp', 0))),
                    "block_number": tx.get('blockNumber', 'N/A'),
                    "tx_type": "Native Transfer" if not contract_address else "Token Transfer"
                }
                processed_transactions.append(processed_tx)
            
            return processed_transactions
        else:
            st.warning(f"PolygonScan API Error: {data.get('message', 'Unknown error')}")
            return []
    
    except requests.RequestException as e:
        st.error(f"Error fetching transaction history: {e}")
        logger.error(f"Transaction history fetch error: {traceback.format_exc()}")
        return []

def render_transaction_history(address):
    """
    Render transaction history in a detailed, interactive Streamlit view
    
    Args:
        address (str): Wallet address to fetch transactions for
    """
    st.subheader(" Transaction History")
    
    # Transaction type and contract selection
    col1, col2 = st.columns(2)
    
    with col1:
        tx_type = st.selectbox(
            "Transaction Type", 
            [
                "All Transactions", 
                "Native MATIC Transfers", 
                "Token Transfers", 
                "Contract Interactions"
            ]
        )
    
    with col2:
        contract_filter = st.text_input(
            "Filter by Contract Address", 
            placeholder="0x...",
            help="Optional: Filter transactions for a specific contract"
        )
    
    # Fetch and display transactions
    try:
        # Fetch transactions based on filter
        if contract_filter and Web3.is_address(contract_filter):
            transactions = fetch_transaction_history(address, contract_filter)
        else:
            transactions = fetch_transaction_history(address)
        
        if not transactions:
            st.info("No transactions found for this address.")
            return
        
        # Convert to DataFrame for easier manipulation
        df_transactions = pd.DataFrame(transactions)
        
        # Apply transaction type filter
        if tx_type == "Native MATIC Transfers":
            df_transactions = df_transactions[df_transactions['tx_type'] == "Native Transfer"]
        elif tx_type == "Token Transfers":
            df_transactions = df_transactions[df_transactions['tx_type'] == "Token Transfer"]
        
        # Display transaction statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Transactions", len(df_transactions))
        
        with col2:
            st.metric(
                "Total Value Transferred", 
                f"{df_transactions['value'].sum():.4f} MATIC"
            )
        
        with col3:
            st.metric(
                "Average Transaction Value", 
                f"{df_transactions['value'].mean():.4f} MATIC"
            )
        
        # Interactive transaction table
        st.dataframe(
            df_transactions,
            column_config={
                "hash": st.column_config.TextColumn("Transaction Hash"),
                "from": st.column_config.TextColumn("From"),
                "to": st.column_config.TextColumn("To"),
                "value": st.column_config.NumberColumn(
                    "Value (MATIC)", 
                    format="%.4f"
                ),
                "timestamp": st.column_config.DatetimeColumn("Timestamp"),
                "tx_type": st.column_config.TextColumn("Transaction Type")
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Visualization of transaction value over time
        if not df_transactions.empty:
            fig_tx_value = px.line(
                df_transactions, 
                x='timestamp', 
                y='value', 
                title='Transaction Value Over Time',
                labels={'value': 'MATIC', 'timestamp': 'Date'}
            )
            
            fig_tx_value.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                title_font_size=16,
                title_x=0.5
            )
            
            st.plotly_chart(fig_tx_value, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error processing transaction history: {e}")
        logger.error(f"Transaction history render error: {traceback.format_exc()}")

def fetch_nfts(address):
    """
    Fetch Gianky NFTs for a given address on Polygon network
    
    Args:
        address (str): Wallet address to fetch NFTs for
    
    Returns:
        list: List of Gianky NFTs owned by the address
    """
    try:
        # Backend endpoint for NFT retrieval
        endpoint = "https://alphabackened.vercel.app/getnfts"
        
        # Prepare request parameters
        params = {
            'address': address,
            'chain': 137  # Polygon chain ID
        }
        
        # Make GET request to fetch NFTs
        response = requests.get(endpoint, params=params, timeout=10)
        response.raise_for_status()
        
        # Parse NFT data
        nfts_data = response.json().get('result', [])
        
        # Filter NFTs for the specific Gianky contract
        gianky_nfts = [
            nft for nft in nfts_data 
            if nft.get('token_address', '').lower() == GIANKY_CONTRACT_ADDRESS.lower()
        ]
        
        # Log the number of Gianky NFTs retrieved
        st.write(f"Retrieved {len(gianky_nfts)} Gianky NFTs")
        
        return gianky_nfts
    
    except requests.RequestException as e:
        st.error(f"Error fetching NFTs: {e}")
        logger.error(f"NFT fetch error: {traceback.format_exc()}")
        return []
    except Exception as e:
        st.error(f"Unexpected error in NFT fetching: {e}")
        logger.error(f"Unexpected NFT fetch error: {traceback.format_exc()}")
        return []

def map_token_id_to_tier(token_id):
    """
    Map token ID to its corresponding tier based on predefined ranges
    
    Args:
        token_id (int or str): Token ID to classify
    
    Returns:
        str: Tier name
    """
    try:
        # Convert token_id to integer
        token_id = int(token_id)
        
        # Define tier ranges
        tier_ranges = [
            (1, 1000000, 'STARTER'),
            (1000001, 2000000, 'BASIC'),
            (2000001, 3000000, 'STANDARD'),
            (3000001, 4000000, 'VIP'),
            (4000001, 5000000, 'PREMIUM'),
            (5000001, 6000000, 'DIAMOND')
        ]
        
        # Find the matching tier
        for min_id, max_id, tier_name in tier_ranges:
            if min_id <= token_id <= max_id:
                return tier_name
        
        # If no tier matches
        return 'UNKNOWN'
    
    except (ValueError, TypeError):
        return 'INVALID'

def get_nft_referral_stats(token_id):
    """
    Get both parent referrer ID and number of NFTs referred by this ID
    
    Args:
        token_id (int): The NFT token ID
        
    Returns:
        tuple: (parent_referrer_id, referral_count)
    """
    try:
        # Get parent referrer ID
        parent_referrer = contract.functions.nftReferrals(token_id).call()
        # Get number of NFTs referred by this ID
        referral_count = contract.functions.referralCounts(token_id).call()
        
        return ('N/A' if parent_referrer == 0 else str(parent_referrer), 
                'N/A' if referral_count == 0 else str(referral_count))
    except Exception as e:
        logger.error(f"Error getting referral stats for token {token_id}: {str(e)}")
        return ('N/A', 'N/A')

def process_nft_data(nfts):
    """Process NFT data and return a DataFrame with additional information"""
    try:
        if not nfts:
            return pd.DataFrame()
        
        # Create list to store processed NFT data
        processed_nfts = []
        
        for nft in nfts:
            token_id = int(nft.get('token_id', 0))
            tier = map_token_id_to_tier(token_id)
            
            # Get both parent referrer ID and referral count
            parent_referrer, referral_count = get_nft_referral_stats(token_id)
            
            processed_nfts.append({
                'token_id': token_id,
                'tier': tier,
                'name': nft.get('name', f'Gianky {tier} NFT #{token_id}'),
                'token_address': nft.get('token_address', 'N/A'),
                'parent_referrer': parent_referrer,
                'referral_count': referral_count
            })
        
        return pd.DataFrame(processed_nfts)
    
    except Exception as e:
        logger.error(f"Error processing NFT data: {str(e)}")
        return pd.DataFrame()

def calculate_estimated_time(nft_count):
    """Calculate estimated processing time based on NFT count"""
    # Base ratio: 100 NFTs = 60 seconds
    estimated_seconds = (nft_count * 60) / 100
    
    if estimated_seconds < 60:
        return f"{round(estimated_seconds)} seconds"
    else:
        minutes = estimated_seconds / 60
        return f"{round(minutes, 1)} minutes"

def render_nft_details(address):
    """Render comprehensive NFT details for a given address"""
    try:
        # First get the NFT count and show estimated time
        nft_balance = contract.functions.balanceOf(address).call()
        estimated_time = calculate_estimated_time(nft_balance)
        st.info(f"Found {nft_balance} NFTs. Estimated processing time: {estimated_time}")
        
        # Show loading spinner while fetching NFTs
        with st.spinner('Fetching NFT collection...'):
            # Progress bar
            progress_bar = st.progress(0)
            status_placeholder = st.empty()
            
            # Update progress for initialization
            progress_bar.progress(10)
            status_placeholder.markdown("Initializing...")
            
            # Hardcoded video mapping for NFT tiers
            NFT_TIER_VIDEOS = {
                'STARTER': '/Users/mac/Desktop/GiankyDashboarrd/ServerApp/strarter.mp4',
                'BASIC': '/Users/mac/Desktop/GiankyDashboarrd/ServerApp/basic.mp4',
                'STANDARD': '/Users/mac/Desktop/GiankyDashboarrd/ServerApp/standard.mp4',
                'VIP': '/Users/mac/Desktop/GiankyDashboarrd/ServerApp/vip.mp4',
                'PREMIUM': '/Users/mac/Desktop/GiankyDashboarrd/ServerApp/premium.mp4',
                'DIAMOND': '/Users/mac/Desktop/GiankyDashboarrd/ServerApp/diamond.mp4'
            }

            # Update progress for NFT fetching
            progress_bar.progress(20)
            status_placeholder.markdown("Fetching NFTs from blockchain...")
            
            # Fetch and process NFTs
            nfts = fetch_nfts(address)
            
            # Update progress for data processing
            progress_bar.progress(60)
            status_placeholder.markdown("Processing NFT metadata and referral information...")
            
            nft_df = process_nft_data(nfts)
            
            # Update progress for rendering
            progress_bar.progress(90)
            status_placeholder.markdown("Preparing visualization...")
            
            # Create a visually appealing NFT overview
            st.markdown("## Gianky NFT Collection")
            
            # Summary statistics
            if not nft_df.empty:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total NFTs", len(nft_df), help="Number of Gianky NFTs owned")
                with col2:
                    referrer_count = sum(1 for x in nft_df['parent_referrer'] if x != 'N/A')
                    st.metric("NFTs with Referrer", referrer_count, help="NFTs with parent referrer")
                with col3:
                    referring_count = sum(1 for x in nft_df['referral_count'] if x != 'N/A')
                    st.metric("Referring NFTs", referring_count, help="NFTs that are referring others")
                with col4:
                    percentage = round((referring_count / len(nft_df) * 100), 2) if len(nft_df) > 0 else 0
                    st.metric("Referring %", f"{percentage}%", help="Percentage of NFTs referring others")
            
            # Final progress update
            progress_bar.progress(100)
            status_placeholder.markdown("Loading complete!")
            time.sleep(1)  # Show completion message briefly
            # Remove progress indicators
            progress_bar.empty()
            status_placeholder.empty()
            
            # Detailed NFT Grid
            st.markdown("### NFT Collection Details")
            
            # Create columns for grid layout
            if not nfts:
                st.warning("No Gianky NFTs found for this address.")
                return
            
            # Prepare NFT grid
            nft_grid = st.columns(3)
            
            for i, nft_row in nft_df.iterrows():
                # Cycle through grid columns
                with nft_grid[i % 3]:
                    # Card-like container for each NFT
                    with st.container():
                        # Get NFT details
                        token_id = nft_row['token_id']
                        tier = nft_row['tier']
                        parent_referrer = nft_row['parent_referrer']
                        referral_count = nft_row['referral_count']
                        
                        st.markdown(f"#### Gianky {tier} NFT #{token_id}")
                        
                        # Try to load tier-specific video
                        try:
                            # Get video path for the specific tier
                            video_path = NFT_TIER_VIDEOS.get(tier, NFT_TIER_VIDEOS.get('STARTER'))
                            
                            # Open and read the video file
                            with open(video_path, 'rb') as video_file:
                                video_bytes = video_file.read()
                            
                            # Display video
                            st.video(video_bytes)
                        except Exception as e:
                            st.warning(f"Unable to load video for {tier} tier")
                            # Fallback image if video fails
                            st.image("https://via.placeholder.com/250x250.png?text=NFT+Video+Unavailable", 
                                     caption="Video Not Available", 
                                     use_column_width=True)
                        
                        # Expandable details
                        with st.expander("NFT Details"):
                            # Display key NFT attributes
                            st.markdown(f"""
                            - **Token ID:** {token_id}
                            - **Tier:** {tier}
                            - **Parent Referrer ID:** {parent_referrer}
                            - **NFTs Referred Count:** {referral_count}
                            - **Contract:** {nft_row['token_address']}
                            - **Name:** {nft_row['name']}
                            """)
            
            # Analytics Visualization
            if not nft_df.empty:
                st.markdown("### Collection Analytics")
                
                # Tier Distribution
                tier_counts = nft_df['tier'].value_counts()
                fig_tier = px.pie(
                    values=tier_counts.values,
                    names=tier_counts.index,
                    title='NFT Tier Distribution',
                    hole=0.3
                )
                fig_tier.update_layout(height=500)  # Increased height for better visibility
                st.plotly_chart(fig_tier, use_container_width=True)
                
                # Parent Referrer Distribution
                referrer_data = nft_df[nft_df['parent_referrer'] != 'N/A'].copy()
                if not referrer_data.empty:
                    fig_ref = px.bar(
                        referrer_data,
                        x='token_id',
                        y='parent_referrer',
                        color='tier',
                        title='Parent Referrer Distribution',
                        labels={'token_id': 'Token ID', 'parent_referrer': 'Parent Referrer ID'}
                    )
                    fig_ref.update_layout(
                        showlegend=True,
                        height=500,  # Increased height for better visibility
                        xaxis_title="Token ID",
                        yaxis_title="Parent Referrer ID"
                    )
                    st.plotly_chart(fig_ref, use_container_width=True)
                else:
                    st.info("No NFTs with parent referrers found.")
                
                # Referral Count Distribution
                referring_data = nft_df[nft_df['referral_count'] != 'N/A'].copy()
                if not referring_data.empty:
                    fig_count = px.bar(
                        referring_data,
                        x='token_id',
                        y='referral_count',
                        color='tier',
                        title='NFTs Referred Count Distribution',
                        labels={'token_id': 'Token ID', 'referral_count': 'NFTs Referred Count'}
                    )
                    fig_count.update_layout(
                        showlegend=True,
                        height=500,  # Increased height for better visibility
                        xaxis_title="Token ID",
                        yaxis_title="NFTs Referred Count"
                    )
                    st.plotly_chart(fig_count, use_container_width=True)
                else:
                    st.info("No NFTs referring others found.")
                    
    except Exception as e:
        st.error(f"Error processing address: {str(e)}")
        logger.error(f"Error in render_nft_details: {traceback.format_exc()}")

def render_nft_collection():
    """Render the NFT Collection page"""
    st.title(" NFT Collection")
    
    # Address input with validation
    address = st.text_input(
        "Enter Wallet Address", 
        placeholder="0x...",
        help="Enter a valid Ethereum/Polygon wallet address"
    )
    
    # Validation and lookup button
    col1, col2 = st.columns([3, 1])
    with col1:
        lookup_type = st.selectbox(
            "Lookup Type",
            [
                "Comprehensive Overview"
            ]
        )
    
    with col2:
        lookup_button = st.button("Lookup", use_container_width=True)
    
    # Lookup process
    if lookup_button and address:
        try:
            # Validate address
            if not Web3.is_address(address):
                st.error("Invalid wallet address. Please check and try again.")
                return
            
            # Fetch address information
            render_nft_details(address)
        
        except Exception as e:
            st.error(f"Error processing address: {str(e)}")
            logger.error(f"Address lookup error: {traceback.format_exc()}")

GIANKY_CONTRACT_ADDRESS = '0xdc91E2fD661E88a9a1bcB1c826B5579232fc9898'

if __name__ == "__main__":
    main()