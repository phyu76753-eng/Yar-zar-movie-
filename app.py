import asyncio
import os
import tempfile
import google.generativeai as genai
import streamlit as st
import edge_tts

st.set_page_config(
    page_title="AI Video Dubbing Editor", layout="wide", page_icon="🎬"
)
st.title("🎬 AI Video Dubbing & Recap Editor")

# Sidebar - API Key Config
st.sidebar.header("⚙️ Settings")
api_key = st.sidebar.text_input("Gemini API Key ထည့်ပါ", type="password")

# Step 1: Upload Video
st.header("Step 1: Video တင်ပါ")
uploaded_video = st.file_uploader(
    "Recap ပြုလုပ်လိုသည့် Video ဖိုင် ရွေးပါ (.mp4)", type=["mp4"]
)

if uploaded_video:
    st.video(uploaded_video)

# Step 2: Source Text & Translation
st.header("Step 2: စာသားစစ်ဆေးခြင်းနှင့် ဘာသာပြန်ခြင်း")
source_text = st.text_area(
    "မူရင်း စာသား ( Chinese / English Subtitle )", height=150
)

if st.button("Gemini ဖြင့် မြန်မာသို့ ဘာသာပြန်မည်"):
    if not api_key:
        st.error("Gemini API Key အရင်ထည့်သွင်းပေးပါ။")
    elif not source_text:
        st.warning("မူရင်း စာသား ရိုက်ထည့်ပါ။")
    else:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.5-flash")
            prompt = f"Translate the following subtitles/transcript into natural, engaging spoken Burmese for a video recap:\n\n{source_text}"

            with st.spinner("Gemini AI ဘာသာပြန်နေပါသည်..."):
                response = model.generate_content(prompt)
                st.session_state["translated_text"] = response.text
                st.success("ဘာသာပြန်ဆိုမှု အောင်မြင်ပါသည်။")
        except Exception as e:
            st.error(f"Error: {e}")

translated_text = st.text_area(
    "မြန်မာဘာသာပြန် စာသား",
    value=st.session_state.get("translated_text", ""),
    height=150,
)

# Step 3: Text-to-Speech (AI Voice)
st.header("Step 3: မြန်မာ AI အသံ ပြုလုပ်ခြင်း")
voice_option = st.selectbox(
    "AI အသံ ရွေးချယ်ပါ",
    [
        "my-MM-ThihaNeural (Male)",
        "my-MM-NilarNeural (Female)",
    ],
)


async def generate_audio_file(text, voice_code, output_path):
    communicate = edge_tts.Communicate(text, voice_code)
    await communicate.save(output_path)


if st.button("AI အသံ ဖန်တီးမည်"):
    if not translated_text:
        st.warning("မြန်မာဘာသာပြန် စာသား ထည့်သွင်းပေးပါ။")
    else:
        voice_code = (
            "my-MM-ThihaNeural"
            if "Thiha" in voice_option
            else "my-MM-NilarNeural"
        )
        with st.spinner("AI အသံ ဖန်တီးနေပါသည်..."):
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            asyncio.run(
                generate_audio_file(
                    translated_text, voice_code, temp_audio.name
                )
            )
            st.session_state["audio_path"] = temp_audio.name
            st.audio(temp_audio.name, format="audio/mp3")
            st.success("အသံဖိုင် ဖန်တီးပြီးပါပြီ။")

# Step 4: Result Export
st.header("Step 4: ရလဒ် ထုတ်ယူခြင်း")
if "audio_path" in st.session_state and uploaded_video:
    st.info(
        "ဗီဒီယိုနှင့် အသံကို ပေါင်းစပ်ရန် အသင့်ဖြစ်နေပါပြီ။ Download ရယူနိုင်ပါသည်။"
    )
