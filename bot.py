import os
import json
from datetime import datetime, timedelta, timezone

import discord
from discord import app_commands
from dotenv import load_dotenv

from google import genai
from keep_alive import keep_alive


# -----------------------
# Config / Env
# -----------------------
load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# Use a modern supported model name. Put this in .env if you want to change later:
# GEMINI_MODEL=gemini-2.5-flash-lite
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite").strip()

if not DISCORD_TOKEN:
    raise RuntimeError("Missing DISCORD_TOKEN in .env")
if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY in .env")

client_ai = genai.Client(api_key=GEMINI_API_KEY)


# -----------------------
# Helpers
# -----------------------
SUPPORTED_FLAGS = {
    "🇬🇧": "English",
    "🇺🇸": "English",
    "🇮🇳": "Hindi (Devanagari script)",
    "🇩🇪": "German",
}

def load_slang_glossary(path: str = "slang.json") -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception:
        return {}

SLANG = load_slang_glossary()

def safe_name(name: str) -> str:
    """
    Prevent accidental pings/mentions in output:
    - Break @ with zero-width space.
    - Neutralize mention-like tokens if they appear.
    """
    if not name:
        return "Unknown"

    name = name.replace("@", "@\u200b")
    name = name.replace("<@", "<@\u200b")
    return name

def clean_message_content(content: str) -> str:
    """
    Normalize whitespace safely WITHOUT regex to avoid edge-case backslash/escape issues.
    """
    content = (content or "").strip()
    content = " ".join(content.split())
    return content

def build_translation_prompt(text: str, target_language: str) -> str:
    glossary = json.dumps(SLANG, ensure_ascii=False, indent=2) if SLANG else "{}"
    return f"""
You are a highly accurate translator for informal chat. The input may be Hinglish (Hindi written in Latin letters mixed with English),
with slang, abbreviations, and funny spellings. Translate into: {target_language}.

Rules:
- Preserve meaning, tone, and intent.
- If the text is already in the target language, still rewrite it cleanly in the target language.
- Expand/interpret Hinglish and phonetic Hindi correctly.
- Use this slang glossary when helpful:
{glossary}

Return ONLY the translated text. No extra commentary.

Text:
{text}
""".strip()

def build_summary_prompt(formatted_chat: str) -> str:
    glossary = json.dumps(SLANG, ensure_ascii=False, indent=2) if SLANG else "{}"
    return f"""
You are summarizing the last 24 hours of a Discord channel. The content may include Hinglish + slang + funny spellings.

Write a SHORT bullet-point summary.
Requirements:
- Use short bullets only (no long paragraphs).
- When referring to members, use the plain text names shown in the chat lines.
- Do NOT use @mentions or ping formats.
- Capture key topics, decisions, and outcomes.
- If there was a heated discussion/fight, mention which members were involved and what it was about, and what the outcome was.
- Use this slang glossary when helpful:
{glossary}

Chat (chronological):
{formatted_chat}

Return ONLY bullets, each starting with "- ".
""".strip()

async def gemini_generate(text_prompt: str) -> str:
    """
    Gemini call wrapper using the NEW google-genai SDK.
    Note: This function is async for convenience with Discord handlers, but the SDK call itself is synchronous.
    """
    resp = client_ai.models.generate_content(
        model=GEMINI_MODEL,
        contents=text_prompt,
    )
    return (getattr(resp, "text", "") or "").strip()

def chunk_text_blocks(lines: list[str], max_chars: int = 12000) -> list[str]:
    """
    Chunk chat lines to avoid model context overflow.
    Uses character limits as a simple proxy.
    """
    chunks = []
    current = []
    current_len = 0

    for line in lines:
        if current_len + len(line) + 1 > max_chars and current:
            chunks.append("\n".join(current))
            current = [line]
            current_len = len(line) + 1
        else:
            current.append(line)
            current_len += len(line) + 1

    if current:
        chunks.append("\n".join(current))

    return chunks


# -----------------------
# Discord Client
# -----------------------
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.messages = True
intents.reactions = True

NO_MENTIONS = discord.AllowedMentions.none()

class SummarizerTranslatorBot(discord.Client):
    def __init__(self):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self):
        # Sync slash commands globally
        await self.tree.sync()

client = SummarizerTranslatorBot()


# -----------------------
# Slash Command: /summary
# -----------------------
@client.tree.command(name="summary", description="Summarize messages from the past 24 hours in this channel.")
async def summary(interaction: discord.Interaction):
    await interaction.response.defer(thinking=True, ephemeral=False)

    channel = interaction.channel
    if not isinstance(channel, discord.TextChannel):
        await interaction.followup.send(
            "This command can only be used in a text channel.",
            allowed_mentions=NO_MENTIONS
        )
        return

    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=24)

    messages = []
    try:
        async for msg in channel.history(limit=None, after=since, oldest_first=True):
            if msg.author.bot:
                continue

            content = clean_message_content(msg.content)
            if not content:
                continue

            author_name = safe_name(getattr(msg.author, "display_name", msg.author.name))
            ts = msg.created_at.strftime("%Y-%m-%d %H:%M UTC")
            messages.append(f"[{ts}] {author_name}: {content}")

    except discord.Forbidden:
        await interaction.followup.send(
            "I don't have permission to read message history in this channel.",
            allowed_mentions=NO_MENTIONS
        )
        return
    except Exception as e:
        await interaction.followup.send(
            f"Error while fetching messages: {e}",
            allowed_mentions=NO_MENTIONS
        )
        return

    if not messages:
        await interaction.followup.send(
            "No messages found in the past 24 hours.",
            allowed_mentions=NO_MENTIONS
        )
        return

    chunks = chunk_text_blocks(messages, max_chars=12000)

    try:
        if len(chunks) == 1:
            prompt = build_summary_prompt(chunks[0])
            final = await gemini_generate(prompt)
            await interaction.followup.send(
                final if final else "Could not generate summary.",
                allowed_mentions=NO_MENTIONS
            )
            return

        partial_summaries = []
        for i, ch in enumerate(chunks, start=1):
            map_prompt = f"""
Summarize this subset of Discord chat into SHORT bullet points.
Return ONLY bullets starting with "- ".
Chunk {i}/{len(chunks)}:

{ch}
""".strip()
            partial = await gemini_generate(map_prompt)
            partial_summaries.append(partial)

        reduce_input = "\n".join(partial_summaries)
        reduce_prompt = f"""
Combine these partial summaries into ONE short bullet-point summary.
Requirements:
- Short bullets only
- Mention member names as plain text (no pings)
- Capture fights/disagreements and outcomes if present
Return ONLY bullets starting with "- ".

Partial summaries:
{reduce_input}
""".strip()

        final = await gemini_generate(reduce_prompt)
        await interaction.followup.send(
            final if final else "Could not generate summary.",
            allowed_mentions=NO_MENTIONS
        )

    except Exception as e:
        await interaction.followup.send(
            f"Error generating summary: {e}",
            allowed_mentions=NO_MENTIONS
        )


# -----------------------
# Reaction-based Translation
# -----------------------
@client.event
async def on_raw_reaction_add(payload: discord.RawReactionActionEvent):
    if not client.user:
        return
    if payload.user_id == client.user.id:
        return

    emoji_str = str(payload.emoji)
    if emoji_str not in SUPPORTED_FLAGS:
        return

    target_language = SUPPORTED_FLAGS[emoji_str]

    channel = client.get_channel(payload.channel_id)
    if channel is None:
        try:
            channel = await client.fetch_channel(payload.channel_id)
        except Exception:
            return

    if not isinstance(channel, (discord.TextChannel, discord.Thread)):
        return

    try:
        msg = await channel.fetch_message(payload.message_id)
    except Exception:
        return

    if msg.author.bot:
        return

    text = clean_message_content(msg.content)
    if not text:
        return

    try:
        prompt = build_translation_prompt(text, target_language)
        translated = await gemini_generate(prompt)
        if not translated:
            return

        await msg.reply(
            translated,
            mention_author=False,
            allowed_mentions=NO_MENTIONS
        )

    except discord.Forbidden:
        return
    except Exception:
        return

keep_alive()

# -----------------------
# Run
# -----------------------
client.run(DISCORD_TOKEN)
