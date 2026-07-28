import os
import gc
import unicodedata

# ═══════════════════════════════════════════════════════════════════════════════
# 1. STRICT RESOURCE LIMITS (1 vCPU, 1 GB RAM)
# ═══════════════════════════════════════════════════════════════════════════════
os.environ["MKL_DISABLE_FAST_MM"] = "1"
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import streamlit as st

# ═══════════════════════════════════════════════════════════════════════════════
# 2. STREAMLIT PAGE CONFIG (MUST BE THE FIRST STREAMLIT CALL)
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Bangla Dialect Translator", 
    page_icon="🇧🇩", 
    layout="centered"
)

import ctranslate2
from transformers import AutoTokenizer

# ═══════════════════════════════════════════════════════════════════════════════
# 3. APP CONFIGURATION & DIALECT MAPPING
# ═══════════════════════════════════════════════════════════════════════════════
MODEL_PATH = "./banglat5_lora_ct2"
TOKENIZER_NAME = "csebuetnlp/banglat5_small"

LANG_MAPPING = {
    "Standard Bangla": "bangla_speech",
    "Sylheti": "sylhet_bangla_speech",
    "Barishal": "barishal_bangla_speech",
    "Chittagong": "chittagong_bangla_speech",
    "Mymensingh": "mymensingh_bangla_speech",
    "Noakhali": "noakhali_bangla_speech",
    "Rangpur": "rangpur_bangla_speech",
    "Rajshahi": "rajshahi_bangla_speech",
    "Kishoreganj": "kishorgonj_bangla_speech",
    "Narail": "narail_bangla_speech",
    "Narsingdi": "narsingdi_bangla_speech",
    "Tangail": "tangail_bangla_speech"
}

# ═══════════════════════════════════════════════════════════════════════════════
# 4. CACHED MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading translation engine into memory...")
def load_translator():
    gc.collect()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    translator = ctranslate2.Translator(
        MODEL_PATH,
        device="cpu",
        compute_type="int8",
        inter_threads=1,
        intra_threads=1
    )
    return tokenizer, translator

tokenizer, translator = load_translator()

# ═══════════════════════════════════════════════════════════════════════════════
# 5. TRANSLATION LOGIC
# ═══════════════════════════════════════════════════════════════════════════════
def preprocess_bangla_text(text: str) -> str:
    """Normalizes Bangla Unicode characters and strips redundant whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split())

def translate(src_lang: str, tgt_lang: str, text: str) -> str:
    cleaned_text = preprocess_bangla_text(text)
    prompt = f"translate {src_lang} to {tgt_lang}: {cleaned_text}"
    
    tokens = tokenizer.tokenize(prompt) + ["</s>"]
    
    results = translator.translate_batch(
        [tokens],
        max_decoding_length=128,
        min_decoding_length=2,
        beam_size=5,
        length_penalty=0.8,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
    )
    
    output_tokens = results[0].hypotheses[0]
    output_text = tokenizer.decode(
        tokenizer.convert_tokens_to_ids(output_tokens), 
        skip_special_tokens=True
    )
    
    return preprocess_bangla_text(output_text)

# ═══════════════════════════════════════════════════════════════════════════════
# 6. STREAMLIT FRONTEND UI
# ═══════════════════════════════════════════════════════════════════════════════
st.title("🇧🇩 Bangla Poly-Dialect Translator")
st.markdown("Translate between Standard Bangla and 11 Regional Dialects.")

# Layout: Two columns for source and target selection
col1, col2 = st.columns(2)

with col1:
    source_display = st.selectbox("Source Dialect", list(LANG_MAPPING.keys()), index=0)

with col2:
    target_display = st.selectbox("Target Dialect", list(LANG_MAPPING.keys()), index=1)

# Text Input Area
input_text = st.text_area(
    f"Enter text in {source_display}:", 
    placeholder="তখন চারপাশের প্রকৃতি এক অপরূপ সৌন্দর্যে সেজে ওঠে...",
    height=150
)

# Translate Button
if st.button("Translate", type="primary", use_container_width=True):
    if not input_text.strip():
        st.warning("Please enter some text to translate.")
    elif source_display == target_display:
        st.info("Source and Target dialects are the same. Please choose different dialects.")
    else:
        with st.spinner("Translating..."):
            src_code = LANG_MAPPING[source_display]
            tgt_code = LANG_MAPPING[target_display]
            
            translation = translate(src_code, tgt_code, input_text)
            
            st.success("Translation Complete!")
            st.text_area(f"Translation in {target_display}:", value=translation, height=150)