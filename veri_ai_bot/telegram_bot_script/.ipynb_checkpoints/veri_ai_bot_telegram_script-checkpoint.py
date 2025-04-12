import asyncio
import numpy as np
import cv2
from io import BytesIO
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext, CallbackQueryHandler
import tensorflow as tf
from tensorflow.keras.models import load_model
import nest_asyncio
import time
from telegram.error import NetworkError

# Fix event loop issues in Jupyter Notebook
nest_asyncio.apply()

# Model path - only AI detection now
AI_DETECTION_MODEL_PATH = r"C:\Users\User\Desktop\DATA SCIENCE\Github\Veri-Ai-Bot\veri_ai_bot\ai_human_distinguish\models\ResNet50V2-AIvsHumanGenImages.keras"

# Load model
ai_detection_model = load_model(AI_DETECTION_MODEL_PATH)

# Class labels
AI_DETECTION_LABELS = ["Human-Created", "AI-Generated"]

# Store the last processed image for each user
user_images = {}

def load_image_from_bytes(byte_image: BytesIO) -> np.ndarray:
    """Load and preprocess image from byte stream."""
    byte_image.seek(0)
    file_bytes = np.asarray(bytearray(byte_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (512, 512))  # Resize to model's expected input size
    image = image.astype(np.float32) / 255.0  # Normalize to [0,1]
    return image

def predict_ai_image(image: np.ndarray) -> str:
    """Predict whether image is AI-generated or human-created."""
    processed_image = np.expand_dims(image, axis=0)
    
    prediction = ai_detection_model.predict(processed_image)[0]
    prob_ai = float(prediction[1]) if len(prediction) > 1 else float(prediction)
    prob_human = 1.0 - prob_ai
    predicted_class = 1 if prob_ai >= 0.5 else 0
    confidence = round(prob_ai * 100, 2) if predicted_class == 1 else round(prob_human * 100, 2)
    label = AI_DETECTION_LABELS[predicted_class]
    
    # Add emoji based on the prediction
    emoji = "🤖" if predicted_class == 1 else "👨‍🎨"
    
    return f"{emoji} *AI Detection Result:* {label}\n🎯 *Confidence:* {confidence}%\n\n⚠️ This is an automated assessment and may not be 100% accurate."

async def start(update: Update, context: CallbackContext) -> None:
    user_name = update.effective_user.first_name
    
    welcome_message = (
        f"👋 *Welcome, {user_name}!* 🌟\n\n"
        "I'm *VeriAI Bot* - your image analysis assistant!\n\n"
        "✨ *What can I do?* ✨\n"
        "🔍 Detect if images are AI-generated or human-created\n\n"
        "📸 Send me any image to get started or use /help to learn more."
    )
    
    await update.message.reply_text(welcome_message, parse_mode="Markdown")

async def help_command(update: Update, context: CallbackContext) -> None:
    help_message = (
        "🌟 *VeriAI Bot Help* 🌟\n\n"
        "📋 *Available Commands:*\n"
        "/start - Start the bot\n"
        "/help - Show this help message\n\n"
        
        "🔍 *AI Image Detection:*\n"
        "• Determines if images are AI-generated or human-created\n"
        "• Best with clear, uncompressed images\n\n"
        
        "To use the feature, simply send me an image! 📸"
    )
    
    await update.message.reply_text(help_message, parse_mode="Markdown")

async def handle_image(update: Update, context: CallbackContext) -> None:
    user_id = update.effective_user.id
    await update.message.reply_text("🔄 *Processing your image...*", parse_mode="Markdown")
    
    photo = update.message.photo[-1]  # Get the largest photo
    retries = 3
    while retries > 0:
        try:
            photo_file = await context.bot.get_file(photo.file_id)
            byte_image = BytesIO()
            await photo_file.download_to_memory(out=byte_image)
            break
        except NetworkError:
            retries -= 1
            await update.message.reply_text("⚠️ Network error. Retrying...")
            time.sleep(2)
        except Exception as e:
            await update.message.reply_text(f"❌ *Error:* {str(e)}\nPlease try again.", parse_mode="Markdown")
            return
    
    if retries == 0:
        await update.message.reply_text("❌ Failed to download image after multiple attempts. Please try again.")
        return
    
    try:
        # Process the image and store it for this user
        image = load_image_from_bytes(byte_image)
        user_images[user_id] = image
        
        # Since we only have AI detection now, we can directly analyze or offer just one option
        await update.message.reply_text("🔄 *Analyzing if image is AI-generated or human-created...*", parse_mode="Markdown")
        result = predict_ai_image(image)
        await update.message.reply_text(result, parse_mode="Markdown")
        
    except Exception as e:
        await update.message.reply_text(f"❌ *Error:* {str(e)}\nPlease try with a different image.", parse_mode="Markdown")

async def text_handler(update: Update, context: CallbackContext) -> None:
    """Handle text messages by prompting user to send an image."""
    await update.message.reply_text(
        "📸 *Please send me an image to analyze!*\n"
        "Use /help to learn what I can do.",
        parse_mode="Markdown"
    )

async def main():
    TOKEN = "8198710095:AAElcwkTbIQibPRSvR4piOK2EvaegpXTXcI"
    application = Application.builder().token(TOKEN).build()
    
    # Add command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    
    # Add image handler
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))
    
    # Add text handler
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))
    
    # Start the bot
    await application.run_polling()

if __name__ == "__main__":
    asyncio.run(main())