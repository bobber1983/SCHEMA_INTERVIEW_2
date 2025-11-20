import streamlit as st
from bs4 import BeautifulSoup
import json
import google.generativeai as genai
import datetime
import os
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

def extract_heuristics(soup):
    """
    Phase 1: Surgical Extraction of metadata available in DOM.
    """
    data = {}
    
    # 1. Basic Meta Tags
    data['url'] = get_safe_attr(soup, 'link[rel="canonical"]', 'href') or get_safe_attr(soup, 'meta[property="og:url"]', 'content')
    base_url = data['url'] # Use this as base for resolving relative links
    
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

def generate_json_ld(data):
    """
    Generates the final JSON-LD script based on the specific template.
    """
    
    # Prepare keywords array string
    keywords_json = json.dumps(data.get('keywords', []))
    
    # Prepare About array
    about_list = [{"@type": "Thing", "name": k} for k in data.get('keywords', [])]
    # Add mentions if available
    if data.get('mentions'):
         about_list.extend([{"@type": "Thing", "name": m} for m in data['mentions']])
    about_json = json.dumps(about_list)
    
    # Prepare SameAs array for interviewee
    same_as_json = json.dumps(data.get('interviewee', {}).get('socialLinks', []))

    # Construct the JSON-LD dictionary
    schema = {
        "@context": "https://schema.org",
        "@type": ["BlogPosting", "Interview"],
        "mainEntityOfPage": {
            "@type": "WebPage",
            "@id": data.get('url', '')
        },
        "inLanguage": "it-IT",
        "headline": data.get('headline', ''),
        "alternativeHeadline": data.get('alternativeHeadline', ''),
        "description": data.get('description', ''),
        "image": {
            "@type": "ImageObject",
            "url": data.get('image_url', ''),
            "width": data.get('image_width', ''),
            "height": data.get('image_height', '')
        },
        "author": {
            "@type": "Person",
            "name": "Roberto Serra",
            "url": "https://www.roberto-serra.com/chi-sono-roberto-serra/",
            "@id": "https://www.roberto-serra.com/chi-sono-roberto-serra/#person"
        },
        "interviewer": {
            "@type": "Person",
            "name": "Roberto Serra",
            "url": "https://www.roberto-serra.com/chi-sono-roberto-serra/",
            "@id": "https://www.roberto-serra.com/chi-sono-roberto-serra/#person"
        },
        "publisher": {
            "@type": "Organization",
            "name": "Roberto Serra SEO Agency",
            "url": "https://www.roberto-serra.com/",
            "logo": {
                "@type": "ImageObject",
                "url": "https://www.roberto-serra.com/wp-content/uploads/2022/07/logo-roberto-serra.png",
                "width": 500,
                "height": 132
            }
        },
        "datePublished": data.get('datePublished', ''),
        "dateModified": data.get('dateModified', ''),
        "keywords": data.get('keywords', []),
        "about": about_list,
        "isPartOf": {
            "@type": "CreativeWorkSeries",
            "name": data.get('seriesName', 'SEO Confidential'),
            "url": "https://www.roberto-serra.com/news-category/interviste/",
            "@id": "https://www.roberto-serra.com/news-category/interviste/#series"
        }
    }

    # Add Audio if present
    if data.get('audio_url'):
        schema["audio"] = {
            "@type": "AudioObject",
            "contentUrl": data['audio_url'],
            "description": data.get('audio_description', ''),
            "name": data.get('headline', ''), # Or specific audio title if extracted
            "encodingFormat": "audio/mpeg"
        }

    # Add Interviewee (The main entity)
    interviewee = data.get('interviewee', {})
    if interviewee.get('name'):
        # Construct a unique ID for the interviewee
        person_id = f"{data.get('url', '')}#person-{interviewee['name'].lower().replace(' ', '-')}"
        
        schema["interviewee"] = {
             "@type": "Person",
             "@id": person_id,
             "name": interviewee.get('name'),
             "description": interviewee.get('bio'),
             "jobTitle": interviewee.get('jobTitle'),
             "affiliation": {
                 "@type": "Organization",
                 "name": interviewee.get('company')
             },
             "image": interviewee.get('image_url'),
             "sameAs": interviewee.get('socialLinks', [])
        }

    json_content = json.dumps(schema, indent=4)
    return f'<script type="application/ld+json">\n{json_content}\n</script>'


# --- Main App UI ---

st.title("🛠️ JSON-LD Schema Generator")
st.markdown("Upload an HTML file to generate the schema.")

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
    
    # 2. Model Persistence (config.json)
    CONFIG_FILE = "config.json"
    
    def load_config():
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def save_config(config):
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f)

    config = load_config()
    last_model = config.get("last_model")

    def get_available_models(api_key):
        if not api_key:
            return []
        try:
            genai.configure(api_key=api_key)
            models = []
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    # Filter: User requested only >= 2.0
                    if "gemini-2." in m.name or "gemini-3." in m.name: 
                        models.append(m.name)
            
            # Fallback
            if not models:
                 for m in genai.list_models():
                    if 'generateContent' in m.supported_generation_methods and "gemini" in m.name:
                        models.append(m.name)
            
            # Sort models
            models = sorted(models, reverse=True)
            
            # Prioritize gemini-2.5-pro if available
            target_model = "models/gemini-2.5-pro"
            if target_model in models:
                models.remove(target_model)
                models.insert(0, target_model)
            
            return models
        except Exception as e:
            # st.error(f"Error fetching models: {e}") # Suppress error on startup if key is invalid
            return ["models/gemini-1.5-flash"] 

    model_options = get_available_models(api_key) if api_key else ["models/gemini-1.5-flash"]
    
    # Determine index for selectbox
    default_index = 0
    if last_model and last_model in model_options:
        default_index = model_options.index(last_model)
        
    model_name = st.selectbox("Model", model_options, index=default_index)
    
    # Save if changed
    if model_name != last_model:
        config["last_model"] = model_name
        save_config(config)
        
    st.info("Get your key from Google AI Studio.")

# File Upload
uploaded_file = st.file_uploader("Upload HTML File", type=['html'])

if uploaded_file:
    # Read file
    html_content = uploaded_file.read().decode("utf-8")
    soup = BeautifulSoup(html_content, 'html.parser')
    
    if st.button("Analyze & Extract"):
        with st.spinner("Extracting Data..."):
            # 1. Heuristics
            heuristics = extract_heuristics(soup)
            
            # 2. AI Extraction
            ai_data = {}
            if api_key:
                clean_text = clean_html_for_ai(soup)
                ai_data = extract_with_ai(clean_text, api_key, model_name, heuristics)
            else:
                st.warning("No API Key provided. Skipping AI enrichment.")
            
            # 3. Merge Data
            # Start with heuristics, update with AI (AI fills gaps or overrides if better)
            final_data = heuristics.copy()
            
            # Merge simple fields if missing in heuristics
            if not final_data.get('alternativeHeadline'):
                final_data['alternativeHeadline'] = ai_data.get('alternativeHeadline', '')
            
            # Merge Interviewee
            final_data['interviewee'] = ai_data.get('interviewee', {})
            
            # Merge Keywords & Mentions
            final_data['keywords'] = ai_data.get('keywords', [])
            final_data['mentions'] = ai_data.get('mentions', [])
            final_data['seriesName'] = ai_data.get('seriesName', 'SEO Confidential')
            
            # Store in session state
            st.session_state['extracted_data'] = final_data
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
        data['datePublished'] = st.text_input("Date Published", data.get('datePublished', ''))
        data['dateModified'] = st.text_input("Date Modified", data.get('dateModified', ''))
        data['seriesName'] = st.text_input("Series Name", data.get('seriesName', ''))
        
    with col2:
        st.markdown("### 🎙️ Media & Keywords")
        data['audio_url'] = st.text_input("Audio URL", data.get('audio_url', ''))
        data['audio_description'] = st.text_area("Audio Caption", data.get('audio_description', ''))
        data['image_url'] = st.text_input("Image URL", data.get('image_url', ''))
        
        keywords_str = st.text_area("Keywords (comma separated)", ", ".join(data.get('keywords', [])))
        data['keywords'] = [k.strip() for k in keywords_str.split(',')]
        
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
        interviewee['socialLinks'] = [s.strip() for s in socials_str.split(',')]
        interviewee['image_url'] = st.text_input("Interviewee Image URL", interviewee.get('image_url', ''))
    
    data['interviewee'] = interviewee
    
    # Generate JSON
    st.divider()
    if st.button("Generate JSON-LD"):
        json_output = generate_json_ld(data)
        st.subheader("🎉 Final JSON-LD")
        st.code(json_output, language='json')
