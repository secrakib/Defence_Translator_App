import os
import gc
import re
import unicodedata
import streamlit as st

# ═══════════════════════════════════════════════════════════════════════════════
# 1. STREAMLIT PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Bangla Poly-Dialect Translator", 
    page_icon="🇧🇩", 
    layout="centered"
)

import ctranslate2
from transformers.models.t5 import T5TokenizerFast as AutoTokenizer

# ═══════════════════════════════════════════════════════════════════════════════
# 2. APP CONFIGURATION & DIALECT MAPPING
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
# 3. CACHED MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading translation engine into memory...")
def load_translator():
    gc.collect()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    translator = ctranslate2.Translator(
        MODEL_PATH,
        device="cpu",
        compute_type="int8", # int8 is already correctly applied here!
        inter_threads=1,
        intra_threads=4      # Increased slightly to allow parallel batch processing
    )
    return tokenizer, translator

tokenizer, translator = load_translator()

# ═══════════════════════════════════════════════════════════════════════════════
# 4. TEXT PROCESSING & "VMAP"-STYLE BATCHING LOGIC
# ═══════════════════════════════════════════════════════════════════════════════
def preprocess_bangla_text(text: str) -> str:
    """Normalizes Bangla Unicode characters (NFC) and strips redundant spaces."""
    if not text:
        return ""
    return " ".join(unicodedata.normalize("NFC", text).split())

def split_into_sentences(text: str) -> list[str]:
    """Splits long paragraphs into individual sentences."""
    raw_chunks = re.split(r'(?<=[।?!])\s+|\n+', text)
    return [s.strip() for s in raw_chunks if s.strip()]

def translate_long_text_batched(src_lang: str, tgt_lang: str, full_text: str, progress_bar=None, batch_size=8) -> str:
    """
    Translates text by batching multiple sentences simultaneously (Vectorized/vmap approach).
    This drastically speeds up inference by fully utilizing CTranslate2's C++ matrix math.
    """
    sentences = split_into_sentences(full_text)
    if not sentences:
        return ""
    
    translated_sentences = []
    total_batches = (len(sentences) + batch_size - 1) // batch_size
    
    for i in range(total_batches):
        # 1. Grab a chunk of sentences
        batch_sentences = sentences[i*batch_size : (i+1)*batch_size]
        
        # 2. Tokenize the entire batch at once
        prompts = [f"translate {src_lang} to {tgt_lang}: {preprocess_bangla_text(s)}" for s in batch_sentences]
        tokens_batch = [tokenizer.tokenize(p) + ["</s>"] for p in prompts]
        
        # 3. Translate the ENTIRE batch simultaneously (The 'vmap' equivalent)
        results = translator.translate_batch(
            tokens_batch,
            max_decoding_length=128,
            min_decoding_length=2,
            beam_size=4,
            length_penalty=0.8,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )
        
        # 4. Decode the outputs
        for res in results:
            output_tokens = res.hypotheses[0]
            output_text = tokenizer.decode(
                tokenizer.convert_tokens_to_ids(output_tokens), 
                skip_special_tokens=True
            )
            translated_sentences.append(preprocess_bangla_text(output_text))
            
        # Update progress bar per batch
        if progress_bar is not None and total_batches > 1:
            progress_bar.progress(min((i + 1) / total_batches, 1.0))
            
    gc.collect()
    return " ".join(translated_sentences)

# ═══════════════════════════════════════════════════════════════════════════════
# 5. STREAMLIT FRONTEND INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════
st.title("🇧🇩 Bangla Poly-Dialect Translator")
st.markdown("Translate sentences, long paragraphs, or articles between Standard Bangla and 11 Regional Dialects.")

col1, col2 = st.columns(2)

with col1:
    source_display = st.selectbox("Source Dialect", list(LANG_MAPPING.keys()), index=0)

with col2:
    target_display = st.selectbox("Target Dialect", list(LANG_MAPPING.keys()), index=1)

input_text = st.text_area(
    f"Enter text in {source_display}:", 
    placeholder="এখানে আপনার বাক্য বা সম্পূর্ণ অনুচ্ছেদ লিখুন...",
    height=200
)

if st.button("Translate Text", type="primary", use_container_width=True):
    if not input_text.strip():
        st.warning("Please enter some text to translate.")
    elif source_display == target_display:
        st.info("Source and Target dialects are the same. Please choose different dialects.")
    else:
        src_code = LANG_MAPPING[source_display]
        tgt_code = LANG_MAPPING[target_display]
        
        progress_bar = st.progress(0.0)
        
        with st.spinner("Translating text using batched processing..."):
            # Added a batch size of 8. Adjust this based on available RAM (lower = less RAM, higher = faster)
            translation = translate_long_text_batched(src_code, tgt_code, input_text, progress_bar=progress_bar, batch_size=8)
            
        progress_bar.empty()
        
        st.success("Translation Complete!")
        st.text_area(
            f"Translation in {target_display}:", 
            value=translation, 
            height=200
        )