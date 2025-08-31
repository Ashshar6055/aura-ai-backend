import os
from dotenv import load_dotenv

# Define the full path to your .env file using an r-string
dotenv_path = r"C:\Users\91914\OneDrive\Desktop\wbs\pass.env"

# Pass the full path to the load_dotenv function
load_dotenv(dotenv_path=dotenv_path)

# Now, try to get the variable
api_key = os.getenv("GEMINI_API_KEY")

# Check if it worked
if api_key:
    print("✅ Success! The .env file was loaded from the specific path.")
    print(f"   API Key loaded: {api_key[:5]}...{api_key[-5:]}")
else:
    print("❌ Error: Could not find the GEMINI_API_KEY from the specified path.")
    print("   Please check the following:")
    print("   1. Is the path you provided correct?")
    print("   2. Is the variable name inside the file 'GEMINI_API_KEY'?")