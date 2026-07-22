import streamlit as st
from bs4 import BeautifulSoup
import json
import google.generativeai as genai
import datetime
import os
import requests
from dateutil import parser
from urllib.parse import urljoin, urlparse

# --- Configuration & Setup ---
st.set_page_config(page_title="JSON-LD Schema Generator", layout="wide")

# --- Helper Functions ---

def get_safe_text(soup, selector):
    """Safely extracts text from a soup selector."""
    element = soup.select_one(selector)
    return element.get_text(strip=True) if element else ""

def get_safe_attr(soup, selector, attr):
    """Safely extracts an attribute from a soup selector."""
    element = soup.select_one(selector)
    return element.get(attr) if element else ""

def clean_html_for_ai(soup):
    """
    Cleans HTML for AI processing. 
    Removes clutter but preserves links and structural tags that might help context.
    """
    # Work on a copy to not affect the original soup if needed elsewhere (though we parse fresh usually)
    content = soup.find('article') or soup.find('div', class_='entry-content') or soup.body
    
    if not content:
        return soup.get_text(strip=True)[:10000] # Fallback

    # Remove unwanted tags
    for tag in content(['script', 'style', 'nav', 'footer', 'iframe', 'noscript']):
        tag.decompose()
        
    # Simplify links: Replace <a href="X">Y</a> with "Y (Link: X)" to help AI find social profiles
    # But for now, let's just pass the cleaned HTML text, Gemini handles HTML well.
    # To be safe and save tokens, let's get text but keep some structure if possible.
    # Actually, passing the raw (cleaned) HTML of the article body is often best for extraction.
    return str(content)[:30000] # Limit to 30k chars to be safe with limits

def validate_and_fix_url(url, base_url):
    """
    Validates and fixes URLs to ensure they are absolute.
    """
    if not url:
        return ""
    
    # Check if it's a local file path (common in saved pages)
    if "_files/" in url or url.startswith("file://"):
        return "" # Discard local paths as we can't reliably reverse them without more info
        
    # Fix relative URLs
    if not url.startswith(('http://', 'https://')):
        if base_url:
            return urljoin(base_url, url)
        return "" # Cannot fix without base_url
        
    return url

def fetch_html(url):
    """
    Fetch a live page's HTML. Returns the HTML string, or None after showing an error.
    Works for server-rendered pages (e.g. WordPress); JS-rendered SPAs won't expose content.
    """
    if not url.startswith(('http://', 'https://')):
        st.error("Inserisci un URL valido che inizi con http:// o https://")
        return None
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; SchemaGenerator/1.0)"}
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        st.error(f"Errore nel recupero della pagina: {e}")
        return None

    ctype = resp.headers.get("Content-Type", "").lower()
    if "html" not in ctype and "<html" not in resp.text[:2000].lower():
        st.error(f"L'URL non restituisce una pagina HTML (Content-Type: {ctype or 'sconosciuto'}).")
        return None

    # requests defaults to ISO-8859-1 without a charset header, mangling accented chars.
    if not resp.encoding or resp.encoding.lower() == "iso-8859-1":
        resp.encoding = resp.apparent_encoding or resp.encoding
    return resp.text

def extract_heuristics(soup, base_url_hint=""):
    """
    Phase 1: Surgical Extraction of metadata available in DOM.
    base_url_hint: the known page URL (when fetched by URL) used as canonical/base fallback.
    """
    data = {}

    # 1. Basic Meta Tags
    data['url'] = (get_safe_attr(soup, 'link[rel="canonical"]', 'href')
                   or get_safe_attr(soup, 'meta[property="og:url"]', 'content')
                   or base_url_hint)
    base_url = data['url'] or base_url_hint  # Use this as base for resolving relative links
    
    data['headline'] = get_safe_attr(soup, 'meta[property="og:title"]', 'content') or soup.title.string
    data['description'] = get_safe_attr(soup, 'meta[name="description"]', 'content') or get_safe_attr(soup, 'meta[property="og:description"]', 'content')
    
    # 2. Dates
    # Try og:updated_time first, then article:published_time
    data['datePublished'] = get_safe_attr(soup, 'meta[property="article:published_time"]', 'content')
    data['dateModified'] = get_safe_attr(soup, 'meta[property="og:updated_time"]', 'content')
    
    if not data['datePublished'] and data['dateModified']:
        data['datePublished'] = data['dateModified'] # Fallback
        
    # 3. Image
    raw_image = get_safe_attr(soup, 'meta[property="og:image"]', 'content')
    data['image_url'] = validate_and_fix_url(raw_image, base_url)
    
    data['image_width'] = get_safe_attr(soup, 'meta[property="og:image:width"]', 'content')
    data['image_height'] = get_safe_attr(soup, 'meta[property="og:image:height"]', 'content')
    
    # 4. Audio (The Critical Part)
    # Selector: figure.wp-block-audio
    audio_figure = soup.select_one('figure.wp-block-audio')
    if audio_figure:
        audio_tag = audio_figure.find('audio')
        raw_audio = ""
        
        # Try audio tag src
        if audio_tag and audio_tag.get('src'):
            raw_audio = audio_tag['src']
            
        # If invalid or missing, try anchor tag inside figure (often a download link)
        if not raw_audio or "_files/" in raw_audio:
             link_tag = audio_figure.find('a')
             if link_tag and link_tag.get('href'):
                 raw_audio = link_tag['href']
        
        data['audio_url'] = validate_and_fix_url(raw_audio, base_url)
            
        caption = audio_figure.find('figcaption')
        if caption:
            data['audio_description'] = caption.get_text(strip=True)
    
    # Fallback for audio_description if not found
    if not data.get('audio_description') and data.get('description'):
        data['audio_description'] = data['description']

    # 5. Article body (plain text) for the articleBody property.
    # Truncated to keep the schema lean; the user can refine it in the form.
    body_root = soup.find('article') or soup.find('div', class_='entry-content') or soup.body
    if body_root:
        data['articleBody'] = body_root.get_text(separator=' ', strip=True)[:5000]

    return data

def extract_with_ai(html_content, api_key, model_name, heuristic_data):
    """
    Phase 3: AI Enrichment.
    """
    if not api_key:
        return {}

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        
        prompt = f"""
        You are an expert SEO Data Extractor. 
        Analyze the following HTML content from a blog interview and extract the requested JSON data.
        
        CONTEXT:
        - The blog post title is: "{heuristic_data.get('headline', '')}"
        - We have already detected this audio URL: "{heuristic_data.get('audio_url', 'None')}"
        
        TASK:
        Extract/Generate the following fields in valid JSON format:
        
        1. "alternativeHeadline": A subtitle or extended title found in the text. If none, generate a compelling one based on the content.
        2. "summary": A 2-sentence summary of the interview.
        3. "keywords": An array of 5-8 relevant SEO keywords (strings).
        4. "interviewee": An object containing:
            - "name": Name of the person interviewed.
            - "jobTitle": Their job title.
            - "company": Their company/organization.
            - "bio": A short, authoritative bio (30-50 words) based on the text.
            - "socialLinks": An array of URLs to their social profiles (LinkedIn, Twitter/X, etc.) found in the text.
            - "image_url": URL of a photo of the interviewee if found in the body (look for img tags near their name).
        5. "mentions": An array of strings listing tools, companies, or people mentioned in the interview (for 'about' schema).
        6. "seriesName": The name of the column/series (e.g., "SEO Confidential").
        7. "seriesDescription": A one or two sentence description of the column/series (in Italian).

        HTML CONTENT:
        {html_content}
        """
        
        response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        parsed = json.loads(response.text)
        
        # Handle list return type just in case
        if isinstance(parsed, list):
            return parsed[0] if parsed else {}
        return parsed

    except Exception as e:
        st.error(f"AI Extraction Error: {e}")
        return {}

# --- Schema constants (single source of truth, mirror schema_instructions.txt) ---
SERIES_URL = "https://www.roberto-serra.com/news-category/interviste/"
DEFAULT_SERIES_NAME = "SEO Confidential"

AUTHOR_PERSON = {
    "@type": "Person",
    "url": "https://www.roberto-serra.com/chi-sono-roberto-serra/",
    "name": "Roberto Serra",
    "@id": "https://www.roberto-serra.com/chi-sono-roberto-serra/#person",
}

PUBLISHER_ORG = {
    "@type": "Organization",
    "name": "Roberto Serra",
    "url": "https://www.roberto-serra.com/",
    "logo": {
        "@type": "ImageObject",
        "url": "https://www.roberto-serra.com/wp-content/uploads/2025/11/logo-roberto-serra.png",
        "width": 500,
        "height": 132,
    },
}

# --- Schema helpers ---

def _to_int(value):
    """Best-effort int conversion; returns None if not convertible."""
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None

def normalize_iso_date(value):
    """Return an ISO 8601 string if parseable, else the original (validation flags it)."""
    if not value:
        return ""
    try:
        return parser.parse(value).isoformat()
    except (ValueError, OverflowError, TypeError):
        return value

def is_iso_date(value):
    """True if value is a valid ISO 8601 date/datetime."""
    if not value:
        return False
    try:
        parser.isoparse(value)
        return True
    except (ValueError, TypeError):
        return False

def prune_empty(obj):
    """Recursively drop empty values (None, '', [], {}) so the schema never
    carries blank/placeholder properties (schema_instructions.txt, rule 9)."""
    empties = (None, "", [], {})
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            pruned = prune_empty(value)
            if pruned not in empties:
                out[key] = pruned
        return out
    if isinstance(obj, list):
        return [v for v in (prune_empty(i) for i in obj) if v not in empties]
    if isinstance(obj, str):
        return obj.strip()
    return obj

def validate_schema_data(data):
    """Return (errors, warnings): human messages for the review panel.
    Errors block generation; warnings just flag omitted/weak fields."""
    errors, warnings = [], []

    if not (data.get('headline') or '').strip():
        errors.append("Manca lo **Headline** (titolo articolo).")

    url = (data.get('url') or '').strip()
    if not url:
        errors.append("Manca la **Canonical URL**: senza, @id e mainEntityOfPage restano vuoti.")
    elif not url.startswith(('http://', 'https://')):
        errors.append("La **Canonical URL** deve iniziare con http:// o https://.")

    published = (data.get('datePublished') or '').strip()
    if not published:
        warnings.append("Manca la **Data di pubblicazione**.")
    elif not is_iso_date(published):
        warnings.append("La **Data di pubblicazione** non è in formato ISO 8601 valido.")

    modified = (data.get('dateModified') or '').strip()
    if modified and not is_iso_date(modified):
        warnings.append("La **Data di modifica** non è in formato ISO 8601 valido.")

    if not (data.get('image_url') or '').strip():
        warnings.append("Manca l'**immagine di copertina**: il blocco image verrà omesso.")

    interviewee = data.get('interviewee') or {}
    if not (interviewee.get('name') or '').strip():
        warnings.append("Manca il **nome dell'intervistato**: il blocco interviewee verrà omesso.")

    audio_url = (data.get('audio_url') or '').strip()
    if audio_url and not audio_url.startswith(('http://', 'https://')):
        warnings.append("L'**Audio URL** non è un URL assoluto valido: il blocco audio verrà omesso.")

    return errors, warnings

def _person_id(url, name):
    """Stable @id for the interviewee entity."""
    if not url:
        return ""
    slug = name.strip().lower().replace(' ', '-')
    return f"{url}#person-{slug}"

def generate_json_ld(data):
    """Build the JSON-LD script mirroring schema_instructions.txt, omitting empty props."""
    url = (data.get('url') or '').strip()

    keywords = [k.strip() for k in data.get('keywords', []) if (k or '').strip()]
    about_list = [{"@type": "Thing", "name": k} for k in keywords]
    about_list += [{"@type": "Thing", "name": (m or '').strip()}
                   for m in data.get('mentions', []) if (m or '').strip()]

    image = {}
    if (data.get('image_url') or '').strip():
        image = {
            "@type": "ImageObject",
            "url": data['image_url'].strip(),
            # Rule 5: default to standard dimensions when unknown.
            "width": _to_int(data.get('image_width')) or 1200,
            "height": _to_int(data.get('image_height')) or 628,
        }

    series_name = (data.get('seriesName') or '').strip() or DEFAULT_SERIES_NAME

    schema = {
        "@context": "https://schema.org",
        "@type": ["BlogPosting", "Article", "Interview"],
        "@id": f"{url}#article" if url else "",
        "mainEntityOfPage": {"@type": "WebPage", "@id": url},
        "headline": data.get('headline', ''),
        "alternativeHeadline": data.get('alternativeHeadline', ''),
        "description": data.get('description', ''),
        "inLanguage": "it-IT",
        "datePublished": normalize_iso_date(data.get('datePublished')),
        "dateModified": normalize_iso_date(data.get('dateModified')),
        "articleSection": series_name,
        "keywords": keywords,
        "about": about_list,
        "isPartOf": {
            "@type": "CreativeWorkSeries",
            "@id": f"{SERIES_URL}#series",
            "name": series_name,
            "url": SERIES_URL,
            "description": data.get('seriesDescription', ''),
        },
        "image": image,
        "audio": {},
        "author": AUTHOR_PERSON,
        "interviewer": AUTHOR_PERSON,
        "interviewee": {},
        "publisher": PUBLISHER_ORG,
        "articleBody": data.get('articleBody', ''),
    }

    # Rule 6: keep audio only when a valid absolute URL is present.
    audio_url = (data.get('audio_url') or '').strip()
    if audio_url.startswith(('http://', 'https://')):
        schema["audio"] = {
            "@type": "AudioObject",
            "name": (data.get('audio_name') or data.get('headline') or '').strip(),
            "description": data.get('audio_description', ''),
            "contentUrl": audio_url,
            "encodingFormat": "audio/mpeg",
        }

    # Rule 7: main entity of the interview.
    interviewee = data.get('interviewee') or {}
    if (interviewee.get('name') or '').strip():
        schema["interviewee"] = {
            "@type": "Person",
            "@id": (interviewee.get('entity_url') or '').strip() or _person_id(url, interviewee['name']),
            "name": interviewee.get('name'),
            "description": interviewee.get('bio'),
            "jobTitle": interviewee.get('jobTitle'),
            "affiliation": {"@type": "Organization", "name": interviewee.get('company')},
            "image": interviewee.get('image_url'),
            "sameAs": [s.strip() for s in interviewee.get('socialLinks', []) if (s or '').strip()],
        }

    schema = prune_empty(schema)
    json_content = json.dumps(schema, indent=2, ensure_ascii=False)
    return f'<script type="application/ld+json">\n{json_content}\n</script>'


# --- Main App UI ---

st.title("🛠️ JSON-LD Schema Generator")
st.markdown("Genera lo schema JSON-LD da un **URL** o da un **file HTML**.")

# Sidebar
# Sidebar
with st.sidebar:
    st.header("Settings")
    
    # 1. API Key Persistence (Secrets or Input)
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
        st.success("API Key loaded from secrets!")
    else:
        api_key = st.text_input("Google Gemini API Key", type="password")
        st.caption("To save permanently, create `.streamlit/secrets.toml` with `GEMINI_API_KEY = 'YOUR_KEY'`")
    
    # 2. Model persistence: session_state (in-session) + best-effort config.json (cross-restart)
    CONFIG_FILE = "config.json"

    def load_last_model():
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    return json.load(f).get("last_model")
            except Exception:
                return None
        return None

    def save_last_model(model):
        # Best effort: on read-only/ephemeral filesystems (cloud deploy) just skip.
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump({"last_model": model}, f)
        except Exception:
            pass

    if "last_model" not in st.session_state:
        st.session_state["last_model"] = load_last_model()
    last_model = st.session_state["last_model"]

    # Non-text variants we never want for JSON extraction.
    EXCLUDE_MODEL = ("image", "tts", "embedding", "vision", "computer-use", "audio")
    # Future-proof aliases preferred as defaults (won't get deprecated like pinned versions).
    PREFERRED_MODELS = ["models/gemini-flash-latest", "models/gemini-pro-latest"]
    FALLBACK_MODEL = "models/gemini-flash-latest"

    def get_available_models(api_key):
        if not api_key:
            return []
        try:
            genai.configure(api_key=api_key)
            models = []
            for m in genai.list_models():
                if 'generateContent' not in m.supported_generation_methods:
                    continue
                name = m.name
                if any(x in name for x in EXCLUDE_MODEL):
                    continue
                if "gemini-2." in name or "gemini-3." in name or "latest" in name:
                    models.append(name)

            # Fallback: any gemini text model
            if not models:
                models = [m.name for m in genai.list_models()
                          if 'generateContent' in m.supported_generation_methods
                          and "gemini" in m.name
                          and not any(x in m.name for x in EXCLUDE_MODEL)]

            models = sorted(set(models), reverse=True)

            # Surface future-proof aliases first (flash-latest as top default).
            for pref in reversed(PREFERRED_MODELS):
                if pref in models:
                    models.remove(pref)
                    models.insert(0, pref)
            return models
        except Exception:
            # Suppress error on startup if the key is invalid.
            return [FALLBACK_MODEL]

    model_options = get_available_models(api_key) if api_key else [FALLBACK_MODEL]
    
    # Determine index for selectbox
    default_index = 0
    if last_model and last_model in model_options:
        default_index = model_options.index(last_model)
        
    model_name = st.selectbox("Model", model_options, index=default_index)

    # Persist selection (session + best-effort disk)
    if model_name != last_model:
        st.session_state["last_model"] = model_name
        save_last_model(model_name)

    st.info("Get your key from Google AI Studio.")

# --- Input: da URL o da file ---
st.subheader("1. Sorgente")
source = st.radio(
    "Da dove prendo l'HTML?",
    ["🔗 Da URL", "📄 Carica file HTML"],
    horizontal=True,
    label_visibility="collapsed",
)

source_url = ""
uploaded_file = None

if source == "🔗 Da URL":
    source_url = st.text_input("URL della pagina", placeholder="https://www.roberto-serra.com/...")
    ready = bool(source_url.strip())
else:
    uploaded_file = st.file_uploader("Upload HTML File", type=['html'])
    ready = uploaded_file is not None

if st.button("Analyze & Extract", type="primary", disabled=not ready):
    with st.spinner("Recupero e analisi in corso..."):
        # 1. Obtain HTML from the chosen source
        if source == "🔗 Da URL":
            html_content = fetch_html(source_url.strip())
        else:
            html_content = uploaded_file.read().decode("utf-8", errors="replace")

        if html_content:
            soup = BeautifulSoup(html_content, 'html.parser')

            # 2. Heuristics (known URL used as canonical/base fallback)
            heuristics = extract_heuristics(soup, base_url_hint=source_url.strip())

            # 3. AI Extraction
            ai_data = {}
            if api_key:
                clean_text = clean_html_for_ai(soup)
                ai_data = extract_with_ai(clean_text, api_key, model_name, heuristics)
            else:
                st.warning("No API Key provided. Skipping AI enrichment.")

            # 4. Merge Data (start with heuristics, AI fills gaps)
            final_data = heuristics.copy()

            if not final_data.get('alternativeHeadline'):
                final_data['alternativeHeadline'] = ai_data.get('alternativeHeadline', '')

            final_data['interviewee'] = ai_data.get('interviewee', {})
            final_data['keywords'] = ai_data.get('keywords', [])
            final_data['mentions'] = ai_data.get('mentions', [])
            final_data['seriesName'] = ai_data.get('seriesName', 'SEO Confidential')
            final_data['seriesDescription'] = ai_data.get('seriesDescription', '')

            # Store in session state; drop any stale generated output
            st.session_state['extracted_data'] = final_data
            st.session_state.pop('json_output', None)
            st.success("Extraction Complete!")

# Display & Edit Form
if 'extracted_data' in st.session_state:
    data = st.session_state['extracted_data']
    
    st.divider()
    st.subheader("Review & Edit Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📄 Article Info")
        data['headline'] = st.text_input("Headline", data.get('headline', ''))
        data['alternativeHeadline'] = st.text_input("Alternative Headline", data.get('alternativeHeadline', ''))
        data['description'] = st.text_area("Description", data.get('description', ''))
        data['url'] = st.text_input("Canonical URL", data.get('url', ''))
        data['datePublished'] = st.text_input("Date Published", data.get('datePublished', ''),
                                               help="Formato ISO 8601, es. 2025-01-05T08:00:00+02:00")
        data['dateModified'] = st.text_input("Date Modified", data.get('dateModified', ''),
                                              help="Formato ISO 8601")
        data['seriesName'] = st.text_input("Series Name", data.get('seriesName', ''))
        data['seriesDescription'] = st.text_area("Series Description", data.get('seriesDescription', ''),
                                                 help="Descrizione della rubrica (isPartOf.description)")

    with col2:
        st.markdown("### 🎙️ Media & Keywords")
        data['audio_url'] = st.text_input("Audio URL", data.get('audio_url', ''))
        data['audio_description'] = st.text_area("Audio Caption", data.get('audio_description', ''))
        data['image_url'] = st.text_input("Image URL", data.get('image_url', ''))
        
        keywords_str = st.text_area("Keywords (comma separated)", ", ".join(data.get('keywords', [])))
        data['keywords'] = [k.strip() for k in keywords_str.split(',') if k.strip()]
        
    st.markdown("### 👤 Interviewee Details")
    int_col1, int_col2 = st.columns(2)
    
    interviewee = data.get('interviewee', {})
    with int_col1:
        interviewee['name'] = st.text_input("Name", interviewee.get('name', ''))
        interviewee['jobTitle'] = st.text_input("Job Title", interviewee.get('jobTitle', ''))
        interviewee['company'] = st.text_input("Company", interviewee.get('company', ''))
    
    with int_col2:
        interviewee['bio'] = st.text_area("Bio", interviewee.get('bio', ''))
        socials_str = st.text_area("Social Links (comma separated)", ", ".join(interviewee.get('socialLinks', [])))
        interviewee['socialLinks'] = [s.strip() for s in socials_str.split(',') if s.strip()]
        interviewee['image_url'] = st.text_input("Interviewee Image URL", interviewee.get('image_url', ''))

    data['interviewee'] = interviewee

    with st.expander("📝 Article Body (opzionale — incluso in articleBody)"):
        data['articleBody'] = st.text_area(
            "Article Body (plain text)", data.get('articleBody', ''),
            height=200, label_visibility="collapsed",
        )

    # Persist edits so they survive reruns / generation
    st.session_state['extracted_data'] = data

    # --- Validation panel ---
    st.divider()
    errors, warnings = validate_schema_data(data)
    if errors:
        st.error("**Da correggere prima di generare:**\n" + "\n".join(f"- {e}" for e in errors))
    if warnings:
        st.warning("**Avvisi (proprietà che verranno omesse):**\n" + "\n".join(f"- {w}" for w in warnings))
    if not errors and not warnings:
        st.success("Tutti i campi principali sono validi ✅")

    # --- Generate JSON ---
    if st.button("Generate JSON-LD", disabled=bool(errors), type="primary"):
        st.session_state['json_output'] = generate_json_ld(data)

    if st.session_state.get('json_output'):
        st.subheader("🎉 Final JSON-LD")
        st.code(st.session_state['json_output'], language='html')
        st.download_button(
            "⬇️ Scarica snippet",
            data=st.session_state['json_output'],
            file_name="schema-jsonld.html",
            mime="text/html",
        )
