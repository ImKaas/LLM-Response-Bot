import logging
import requests
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
import os
from dotenv import load_dotenv

API_URL = "http://localhost:8000/ask"  # Update if hosted remotely

# Your Telegram bot token from BotFather
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

logging.basicConfig(level=logging.INFO)

# Function to call your FastAPI endpoint
def ask_question_to_api(question: str):
    try:
        response = requests.post(API_URL, json={"question": question})
        if response.status_code == 200:
            return response.json().get("answer", "No answer found.")
        else:
            return f"API Error: {response.status_code}\n{response.text}"
    except Exception as e:
        return f"Exception: {str(e)}"

# Handler for user messages
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    await update.message.reply_text("Thinking...")
    answer = ask_question_to_api(user_message)
    await update.message.reply_text(answer)
    

# Start command handler
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hi! Send me your question, and I’ll try to help you!")

# Main app setup
def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()