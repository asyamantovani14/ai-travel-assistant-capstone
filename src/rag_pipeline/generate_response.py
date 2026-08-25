import os
import sys
import pathlib
import json
import re
import requests
from dotenv import load_dotenv
from openai import OpenAI

# Ensure src/ is in sys.path
ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.logger import log_interaction
from nlp.ner_utils import extract_entities
from agents.tool_wrappers import generate_smart_enrichment
from utils.csv_logger import save_response_to_csv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Path to enriched knowledge base
KB_FILE = os.path.join(ROOT_DIR, "data", "knowledge_base", "family_travel_knowledge_enriched.json")


# =========================
# Markdown / URL utilities
# =========================

URL_BARE_RE = re.compile(r'(?<!\]\()https?://[^\s)<>"]+')
MD_LINK_RE = re.compile(r'\[([^\]]+)\]\((https?://[^\s)]+)\)')
ANGLE_LINK_RE = re.compile(r'<(https?://[^>\s]+)>')

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; AsyaThesisBot/1.0)"}

def autolink_bare_urls(md: str) -> str:
    """
    Wraps bare URLs in <...> so Markdown makes them clickable.
    Example: https://example.com -> <https://example.com>
    """
    if not md:
        return md
    return URL_BARE_RE.sub(lambda m: f"<{m.group(0)}>", md)

def format_response_markdown(text: str) -> str:
    """
    Inserts '---' between paragraphs for clean Markdown rendering.
    """
    blocks = [block.strip() for block in text.strip().split("\n\n") if block.strip()]
    return "\n\n---\n\n".join(blocks)

def load_enriched_kb():
    """Load enriched knowledge base JSON."""
    if not os.path.exists(KB_FILE):
        return []
    with open(KB_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def build_kb_snippet(kb_entries):
    """
    Build a short snippet from enriched KB entries
    (title + up to 2 tips + up to 5 tags, up to 3 entries total).
    """
    snippet = []
    for e in kb_entries:
        title = e.get("title")
        tips = e.get("tips", [])[:2]
        tags = e.get("tags", [])[:5]
        entry_lines = [f"**{title}**"] if title else []
        for t in tips:
            entry_lines.append(f"- {t}")
        if tags:
            entry_lines.append(f"Tags: {', '.join(tags)}")
        if entry_lines:
            snippet.append("\n".join(entry_lines))
        if len(snippet) >= 3:
            break
    return "\n\n".join(snippet)


def enrich_prompt_with_tools(prompt, entities):
    """Build a provider-neutral enrichment instruction for legacy callers."""
    destination = entities.get("destination")
    origin = entities.get("origin")
    cuisine = entities.get("cuisine")
    details = [prompt]
    if origin and destination:
        details.append(f"Verify live transport options from {origin} to {destination}.")
    if destination:
        details.append(
            f"Hotel options in {destination}, attractions, and restaurants must be "
            "checked against live providers before being presented as current."
        )
    if cuisine:
        details.append(f"Prefer verified {cuisine} restaurants when available.")
    return "\n".join(details)

def check_url(url: str, timeout: int = 6) -> tuple[bool, int, str]:
    """
    Check if a URL responds (2xx/3xx). Follows redirects.
    Returns: (ok, status_code, final_url)
    """
    try:
        r = requests.head(url, allow_redirects=True, headers=HEADERS, timeout=timeout)
        # Some sites block HEAD or require GET
        if r.status_code >= 400 or r.status_code in (405, 403):
            r = requests.get(url, allow_redirects=True, headers=HEADERS, timeout=timeout, stream=True)
        ok = 200 <= r.status_code < 400
        return ok, r.status_code, r.url
    except requests.RequestException:
        return False, 0, url

def extract_links_from_markdown(md: str):
    """
    Extract links from the model's Markdown output in three forms:
    - [label](url)
    - <url>
    - bare URLs (which we already convert to <url> via autolink_bare_urls)
    Returns a list of tuples (label, url).
    """
    links = []

    # 1) [label](url)
    for label, url in MD_LINK_RE.findall(md):
        links.append((label.strip(), url.strip()))

    # 2) <url>
    for url in ANGLE_LINK_RE.findall(md):
        links.append((url, url))

    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for label, url in links:
        key = (label, url)
        if key not in seen:
            seen.add(key)
            uniq.append((label, url))
    return uniq

def render_valid_blog_links(links, max_items=12):
    """
    Takes a list of (label, url), verifies the URLs, and shows
    only valid ones with clickable buttons.
    """
    import streamlit as st

    valid_count = 0
    for label, url in links:
        if valid_count >= max_items:
            break
        ok, status, final_url = check_url(url)
        if not ok:
            st.caption(f"⛔ Unreachable link ({status}) – {label}")
            continue
        with st.container(border=True):
            st.markdown(f"**{label}**")
            st.markdown(f"[Open in new tab]({final_url})")
            st.link_button("🔗 Open blog", final_url)
        valid_count += 1

    if valid_count == 0:
        st.warning("No reachable blogs were found in the results.")


# =========================
# Core generation functions
# =========================

def _conversation_text(conversation_history, max_messages=8):
    """Render recent conversation turns as compact prompt context."""
    if not conversation_history:
        return ""
    turns = []
    for message in conversation_history[-max_messages:]:
        role = str(message.get("role", "user")).strip().title()
        content = str(message.get("content", "")).strip()
        if role in {"User", "Assistant"} and content:
            turns.append(f"{role}: {content}")
    return "\n".join(turns)


def generate_response(
    query,
    context_docs,
    model=None,
    client=None,
    log_dir="logs",
    conversation_history=None,
    render=False,
):
    model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    provided_client = client is not None
    if client is None:
        client = OpenAI(api_key=api_key)

    # Extract entities and tool context
    conversation = _conversation_text(conversation_history)
    entity_query = f"{conversation}\nUser: {query}" if conversation else query
    extracted_entities = extract_entities(entity_query)
    tool_context = (
        enrich_prompt_with_tools("Travel provider context", extracted_entities)
        if provided_client
        else generate_smart_enrichment(extracted_entities)
    )

    # Load enriched KB and build snippet
    kb_entries = load_enriched_kb()
    kb_snippet = build_kb_snippet(kb_entries)

    # Prepare context docs
    truncated_docs = [doc[:1000] for doc in context_docs if isinstance(doc, str) and doc.strip()]
    has_human_opinion = bool(kb_entries)

    # Assemble prompt
    prompt_parts = [
        "You are a professional and friendly travel assistant.",
        "",
        "Your task is to generate a customized, engaging travel itinerary using the user query and blog-based documents.",
        "",
        "Instructions:",
        "- Create a daily itinerary if possible.",
        "- Use **direct quotes** from the blogs with **citations** (e.g. [source](https://blog.com/post)).",
        "- Include practical tips and tags extracted from the blogs.",
        "- If no relevant blog data exists, explain clearly there are no blog-based human opinions.",
        "- Keep the format clean and Markdown-friendly.",
        "- Add a final note suggesting the user ask again for other destinations or options.",
        "- If you include bare URLs, wrap them in angle brackets like <https://example.com> for Markdown autolink.",
        ""
    ]
    if conversation:
        prompt_parts.append("Recent conversation (preserve established trip details):")
        prompt_parts.append(conversation)
        prompt_parts.append("")
    if kb_snippet:
        prompt_parts.append("Blog Enriched Data:")
        prompt_parts.append(kb_snippet)
        prompt_parts.append("")
    prompt_parts.append("Context from tools and blogs:")
    prompt_parts.append(tool_context)
    prompt_parts.extend(truncated_docs)
    prompt_parts.append("")
    prompt_parts.append(f"User Query:\n{query}")
    prompt_parts.append("")
    prompt_parts.append("Answer (markdown format):")
    final_prompt = "\n".join(prompt_parts)

    # Tone
    if any(k in query.lower() for k in ["family", "adventure", "backpack"]):
        tone = "You are an energetic and curious travel expert specializing in family trips, adventures and relaxing getaways."
    else:
        tone = "You are a calm and thoughtful travel planner with a focus on high-quality suggestions."

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": tone},
                {"role": "user", "content": final_prompt}
            ],
            temperature=0.8,
            max_tokens=700
        )

        raw_response = response.choices[0].message.content.strip()
        final_response = format_response_markdown(raw_response)
        final_response = autolink_bare_urls(final_response)  # make bare URLs clickable

        if not has_human_opinion:
            final_response += (
                "\n\n---\n\n"
                "⚠️ We couldn't find any relevant blog-based human opinions for your query. "
                "Try refining your question or exploring a different destination."
            )

        if render:
            import streamlit as st

            st.download_button("Download (Markdown)", data=final_response, file_name="response.md")
            st.markdown(final_response, unsafe_allow_html=False)
            links = extract_links_from_markdown(final_response)
            if links:
                st.subheader("Blogs found (verified)")
                render_valid_blog_links(links, max_items=12)

        # --- logging
        log_interaction(
            query=query,
            matched_docs=context_docs,
            response=final_response,
            extracted_entities=extracted_entities,
            model=model,
            final_prompt=final_prompt,
            log_dir=log_dir
        )

        save_response_to_csv(
            query=query,
            response=final_response,
            model=model,
            entities=extracted_entities,
            prompt=final_prompt
        )

        return final_response

    except Exception as e:
        error_msg = f"Error generating response: {e}"
        log_interaction(
            query=query,
            matched_docs=context_docs,
            response=error_msg,
            model=model,
            log_dir=log_dir
        )
        if render:
            import streamlit as st

            st.error(error_msg)
        return error_msg


def generate_response_without_rag(query, model=None, client=None, render=False):
    model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    if client is None:
        client = OpenAI(api_key=api_key)
    try:
        messages = [
            {"role": "system", "content": "You are a helpful travel assistant."},
            {"role": "user", "content": query}
        ]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        raw_response = response.choices[0].message.content.strip()
        final_response = format_response_markdown(raw_response)
        final_response = autolink_bare_urls(final_response)

        if render:
            import streamlit as st

            st.download_button("Download (Markdown)", data=final_response, file_name="gpt_response.md")
            st.markdown(final_response, unsafe_allow_html=False)
            links = extract_links_from_markdown(final_response)
            if links:
                st.subheader("Blogs found (verified)")
                render_valid_blog_links(links, max_items=12)

        return final_response
    except Exception as e:
        error_msg = f"Error (no RAG): {e}"
        if render:
            import streamlit as st

            st.error(error_msg)
        return error_msg
