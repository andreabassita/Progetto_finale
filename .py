import os
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")

import discord
from discord.ext import commands
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import io
import requests

# 1. Attivazione Intent per leggere i messaggi e allegati
intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

# 2. Caricamento del modello Teachable Machine e delle etichette
model = tf.keras.models.load_model("keras_model.h5", compile=False)
class_names = open("labels.txt", "r", encoding="utf-8").readlines()

# Funzione per elaborare e classificare l'immagine
def classify_image(image_bytes):
    # Ridimensiona l'immagine a 224x224 (dimensione standard di Teachable Machine)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # Converti l'immagine in un array numpy
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Carica l'immagine nell'array
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Predizione
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    return class_name, confidence_score

@bot.event
async def on_ready():
    print(f"Bot connesso come {bot.user}")

# Comando per analizzare l'immagine allegata al messaggio
@bot.command()
async def analizza(ctx):
    # Verifica se è stata allegata un'immagine
    if not ctx.message.attachments:
        await ctx.send("Per favore, allega un'immagine al messaggio con il comando `!analizza`.")
        return

    attachment = ctx.message.attachments[0]
    
    # Controlla che l'allegato sia un'immagine
    if not any(attachment.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.webp']):
        await ctx.send("Invia un file immagine valido (JPG, PNG, WEBP).")
        return

    await ctx.send("Analisi in corso...")

    # Scarica l'immagine inviata dall'utente
    response = requests.get(attachment.url)
    
    # Esegui la classificazione
    class_name, confidence = classify_image(response.content)

    # Invia il risultato nel canale Discord
    percentuale = round(confidence * 100, 2)
    await ctx.send(f"**Risultato:** {class_name}\n**Accuratezza:** {percentuale}%")

# Inserisci qui il TOKEN del tuo bot Discord
bot.run(TOKEN)
