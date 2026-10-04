import asyncio
import os
import tempfile
import google.generativeai as genai
import numpy as np
import streamlit as st
import edge_tts
from moviepy.editor import AudioFileClip, VideoFileClip, vfx

st.set_page_config(
    page_title="Pro AI Movie Recapper", layout="wide", page_icon="🎬"
)
st.title("🎬 Pro AI Movie Recap & Dubbing Tool")

# Sidebar Settings
st.sidebar.header("⚙️ App Settings")
# Streamlit Secrets သို့မဟုတ် Sidebar မှ API Key ကို ယူခြင်း
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    api_key = st.sidebar.text_input("Gemini API Key ထည့်ပါ")

voice_option = st.sidebar.selectbox(
    "AI မြန်မာအသံ ရွေးပါ",
    ["my-MM-ThihaNeural (Male)", "my-MM-NilarNeural (Female)"],
)

recap_style = st.sidebar.selectbox(
    "Movie Recap စတိုင် ရွေးချယ်ပါ",
    [
        "1. Standard Narrative (အစဉ်လိုက် ပေါ့ပေါ့ပါးပါး ပြောပြမည်)",
        "2. Short-Form Fast (TikTok/Reals စက္ကန့်ပိုင်း ရင်ခုန်စရာ)",
        "3. Ending Explained & Analysis (ဇာတ်သိမ်းအသေးစိတ် သုံးသပ်မည်)",
        "4. Funny Commentary (ဟာသနှင့် စနောက်သရော် ပြောပြမည်)",
        "5. Survival & Horror Rules (သဲထိတ်ရင်ဖို လွတ်မြောက်ရေးစနစ်)",
        "6. Character-Focused (ဇာတ်ကောင် သီးသန့် အဓိကထားမည်)",
    ],
)

st.sidebar.subheader("🛡️ Anti-Copyright (မူပိုင်ခွင့် ကာကွယ်ရေး)")
enable_flip = st.sidebar.checkbox("Video ဘယ်/ညာ မှန်တုံ့ပြန် ပြောင်းမည် (Flip)", value=True)
enable_speed = st.sidebar.checkbox("Speed 1.05x အနည်းငယ် မြှင့်မည်", value=True)

# Step 1: Upload Video
st.header("Step 1: Video တင်ပါ")
uploaded_video = st.file_uploader(
    "Recap ပြုလုပ်လိုသည့် Video ဖိုင် ရွေးပါ (.mp4)", type=["mp4", "mov"]
)

if uploaded_video:
    st.video(uploaded_video)

    # Step 2: Generate Script with Gemini
    st.header("Step 2: AI Script & Myanmar Voiceover ဖန်တီးခြင်း")

    if st.button("🚀 AI Script နှင့် မြန်မာအသံ စတင်ဖန်တီးမည်"):
        if not api_key:
            st.error("Sidebar တွင် Gemini API Key အရင်ထည့်သွင်းပေးပါ။")
        else:
            try:
                genai.configure(api_key=api_key)

                # Save video to temp file
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".mp4"
                ) as tmp_v:
                    tmp_v.write(uploaded_video.read())
                    video_path = tmp_v.name

                with st.spinner("AI က ဗီဒီယိုကို လေ့လာပြီး Script ရေးသားနေပါသည်..."):
                    video_file = genai.upload_file(path=video_path)
                    model = genai.GenerativeModel("gemini-2.5-flash")

                    prompt = f"""
                    You are a professional movie recap creator. Watch and listen to this video.
                    Generate a Burmese spoken script for a video recap using this style: {recap_style}.
                    Keep the script clear, natural, and engaging in modern spoken Burmese.
                    """

                    response = model.generate_content([video_file, prompt])
                    script_text = response.text
                    st.session_state["script_text"] = script_text

                    # Delete uploaded file from Gemini server
                    try:
                        genai.delete_file(video_file.name)
                    except:
                        pass

                # TTS Generation
                with st.spinner("AI မြန်မာအသံ ပြုလုပ်နေပါသည်..."):
                    voice_code = (
                        "my-MM-ThihaNeural"
                        if "Thiha" in voice_option
                        else "my-MM-NilarNeural"
                    )

                    async def gen_audio(text, v_code, out_p):
                        communicate = edge_tts.Communicate(text, v_code)
                        await communicate.save(out_p)

                    audio_temp = tempfile.NamedTemporaryFile(
                        delete=False, suffix=".mp3"
                    )
                    asyncio.run(
                        gen_audio(script_text, voice_code, audio_temp.name)
                    )
                    st.session_state["audio_path"] = audio_temp.name
                    st.session_state["video_path"] = video_path

                st.success("Script နှင့် အသံဖိုင် ဖန်တီးပြီးပါပြီ။")

            except Exception as e:
                st.error(f"Error: {e}")

if "script_text" in st.session_state:
    st.subheader("📝 ထွက်ရှိလာသော Recap Script")
    st.text_area("Burmese Script", value=st.session_state["script_text"], height=200)

    # Step 3: Process Video & Lip Sync / Copyright Adjust
    st.header("Step 3: အသံ/ရုပ် ကိုက်ညီအောင် ညှိခြင်းနှင့် Copyright ပြင်ဆင်ခြင်း")

    if st.button("🎬 ဗီဒီယို အပြီးသတ် Render ပြုလုပ်မည်"):
        with st.spinner(
            "ဗီဒီယိုနှင့် အသံကို ချိန်ညှိ၍ Anti-Copyright Filter များ ထည့်သွင်းနေပါသည်..."
        ):
            try:
                video_clip = VideoFileClip(st.session_state["video_path"])
                audio_clip = AudioFileClip(st.session_state["audio_path"])

                # Anti-Copyright 1: Mirror Flip
                if enable_flip:
                    video_clip = video_clip.fx(vfx.mirror_x)

                # Anti-Copyright 2: Speed up slightly
                if enable_speed:
                    video_clip = video_clip.fx(vfx.speedx, 1.05)

                # Audio & Video Sync: Adjust video duration to match audio
                video_duration = video_clip.duration
                audio_duration = audio_clip.duration

                speed_factor = video_duration / audio_duration
                final_video = video_clip.fx(vfx.speedx, speed_factor)

                # Merge Audio and Video
                final_video = final_video.set_audio(audio_clip)

                output_path = tempfile.NamedTemporaryFile(
                    delete=False, suffix=".mp4"
                ).name
                final_video.write_videofile(
                    output_path,
                    codec="libx264",
                    audio_codec="aac",
                    fps=24,
                    verbose=False,
                    logger=None,
                )

                st.success("အပြီးသတ် Video Recap ဖိုင် ထွက်ရှိပါပြီ!")
                st.video(output_path)

                with open(output_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Final Video Download ရယူပါ",
                        data=f,
                        file_name="movie_recap_final.mp4",
                        mime="video/mp4",
                    )

            except Exception as e:
                st.error(f"Video Processing Error: {e}")
