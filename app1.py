import os
import gc
import re
import unicodedata

# ═══════════════════════════════════════════════════════════════════════════════
# 1. STRICT RESOURCE & MEMORY GUARDS (Must run before imports)
# ═══════════════════════════════════════════════════════════════════════════════


import streamlit as st

# ═══════════════════════════════════════════════════════════════════════════════
# 2. STREAMLIT PAGE CONFIGURATION (Must be the first Streamlit command)
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Bangla Poly-Dialect Translator", 
    page_icon="🇧🇩", 
    layout="centered"
)

import ctranslate2
# Direct import to avoid PyTorch dependency checks inside transformers
from transformers.models.t5 import T5TokenizerFast as AutoTokenizer

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
# 5. TEXT PROCESSING & SENTENCE CHUNKING LOGIC
# ═══════════════════════════════════════════════════════════════════════════════
def preprocess_bangla_text(text: str) -> str:
    """Normalizes Bangla Unicode characters (NFC) and strips redundant spaces."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split())

def split_into_sentences(text: str) -> list[str]:
    """Splits long paragraphs into individual sentences based on Dari (।), ?, !, or newlines."""
    raw_chunks = re.split(r'(?<=[।?!])\s+|\n+', text)
    cleaned_chunks = [s.strip() for s in raw_chunks if s.strip()]
    return cleaned_chunks

def translate_sentence(src_lang: str, tgt_lang: str, sentence: str) -> str:
    """Translates a single sentence using low-memory CTranslate2 parameters."""
    cleaned_sentence = preprocess_bangla_text(sentence)
    prompt = f"translate {src_lang} to {tgt_lang}: {cleaned_sentence}"
    
    tokens = tokenizer.tokenize(prompt) + ["</s>"]
    
    results = translator.translate_batch(
        [tokens],
        max_decoding_length=128,  # Plenty of length for one single sentence
        min_decoding_length=2,
        beam_size=4,
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

def translate_long_text(src_lang: str, tgt_lang: str, full_text: str, progress_bar=None) -> str:
    """Translates entire articles/paragraphs without truncation by processing sentence by sentence."""
    sentences = split_into_sentences(full_text)
    
    if not sentences:
        return ""
    
    translated_sentences = []
    total = len(sentences)
    
    for idx, sentence in enumerate(sentences):
        translated = translate_sentence(src_lang, tgt_lang, sentence)
        translated_sentences.append(translated)
        
        # Update progress bar if translating long text
        if progress_bar is not None and total > 1:
            progress_bar.progress((idx + 1) / total)
            
    # Free memory after processing document
    gc.collect()
    
    return " ".join(translated_sentences)

# ═══════════════════════════════════════════════════════════════════════════════
# 6. STREAMLIT FRONTEND INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════
st.title("🇧🇩 Bangla Poly-Dialect Translator")
st.markdown("Translate sentences, long paragraphs, or articles between Standard Bangla and 11 Regional Dialects.")

# Layout: Two columns for source and target selection
col1, col2 = st.columns(2)

with col1:
    source_display = st.selectbox("Source Dialect", list(LANG_MAPPING.keys()), index=0)

with col2:
    target_display = st.selectbox("Target Dialect", list(LANG_MAPPING.keys()), index=1)

# Text Input Area
input_text = st.text_area(
    f"Enter text in {source_display}:", 
    placeholder="এখানে আপনার বাক্য বা সম্পূর্ণ অনুচ্ছেদ লিখুন...",
    height=200
)

# Translate Button Action
if st.button("Translate Text", type="primary", use_container_width=True):
    if not input_text.strip():
        st.warning("Please enter some text to translate.")
    elif source_display == target_display:
        st.info("Source and Target dialects are the same. Please choose different dialects.")
    else:
        src_code = LANG_MAPPING[source_display]
        tgt_code = LANG_MAPPING[target_display]
        
        progress_bar = st.progress(0.0)
        
        with st.spinner("Translating sentence by sentence..."):
            translation = translate_long_text(src_code, tgt_code, input_text, progress_bar=progress_bar)
            
        progress_bar.empty()
        
        st.success("Translation Complete!")
        st.text_area(
            f"Translation in {target_display}:", 
            value=translation, 
            height=200
        )