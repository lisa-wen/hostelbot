from telegram import Update, ReplyKeyboardRemove
from telegram.ext import CommandHandler, ConversationHandler, Application, ContextTypes, MessageHandler, filters
from dotenv import load_dotenv
import pickle
import os
import json


WEDDING = range(0)
# Die Daten aus den Dateien laden
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('tfidf.pkl', 'rb') as tfidf_file:
    tfidf_vectorizer = pickle.load(tfidf_file)
with open("faq.json", "r", encoding="UTF-8") as file:
    faq = json.load(file)


async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Gibt Informationen zum Start der Unterhaltung"""
    await update.message.reply_text(
        'Hallo! Ich gebe dir Informationen zum Hochzeitsangebot und zu Veranstaltungen. Falls du etwas zum '
        'Hochzeitsangebot wissen möchtest, stelle deine Frage jetzt!'
    )
    return WEDDING


async def get_wedding(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Sagt die Kategorie für einen neuen Text bzw. eine neue Frage vorher"""
    new_text = [update.message.text]
    new_text_tfidf = tfidf_vectorizer.transform(new_text)
    predicted_label = model.predict(new_text_tfidf)[0]
    await update.message.reply_text(faq[predicted_label])
    await update.message.reply_text("Sende eine neue Frage oder /cancel, um das Gespräch zu beenden")
    return WEDDING


async def help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Gibt weiterführende Informationen, wenn Nutzer*innen um Hilfe bitten bzw.
    nicht klar ist, wie die Unterhaltung weitergehen kann."""
    await update.message.reply_text(
        "Sorry! Ich kann deine Anfrage nicht verarbeiten. Sende /start, um den Chat zu beginnen und /cancel, "
        "um ihn zu beenden"
    )


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Beendet die Unterhaltung."""
    user_data = context.user_data
    user_data.clear()
    await update.message.reply_text(
        "Tschüß! Es war schön mit dir zu sprechen. Hoffentlich klappt es bald mal wieder!", reply_markup=ReplyKeyboardRemove()
    )
    return ConversationHandler.END


def main() :
    load_dotenv()
    bot_token = os.getenv("BOT_TOKEN")
    application = Application.builder().token(bot_token).build()
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', handle_start)],
        states={
            WEDDING: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_wedding)],
        },
        fallbacks = [
            CommandHandler('start', handle_start),
            MessageHandler(~filters.TEXT & ~filters.COMMAND, help),
            CommandHandler('cancel', cancel)
        ]
    )
    application.add_handler(conv_handler)
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()
