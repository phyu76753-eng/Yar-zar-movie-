import asyncio
import json
import os
import random
import re
import subprocess
import tempfile
import time
from pathlib import Path

import edge_tts
import imageio_ffmpeg
import pandas as pd
import streamlit as st
from google import genai
from moviepy.editor import (
    AudioFileClip,
    CompositeAudioClip,
    VideoFileClip,
    vfx,
)


MODEL_NAME = "gemini-3.8-flash"
FALLBACK_MODELS = [
    MODEL_NAME,
    "gemini-3.6-flash",
    "gemini-3.5-flash",
]

st.set_page_config(
    page_title="AI Myanmar Video Dubbing",
    layout="wide",
    page_icon="🎬",
)
st.title("🎬 AI Movie Dubbing — Any Language to Myanmar")
st.write(
    "ဗီဒီယိုထဲက မည်သည့်ဘာသာစကားဖြင့် ပြောထားသော စကားကိုမဆို အလိုအလျောက် "
    "စာသားနှင့်အချိန်မှတ်တမ်း ထုတ်ယူ၊ မြန်မာဘာသာပြန်ပြီး စာကြောင်းတစ်ကြောင်းချင်း အသံတင်ပေးပါသည်။"
)


# ---------- Utility functions ----------
def parse_json_response(text):
    """Parse JSON even if the model accidentally wraps it in a code fence."""
    if not text:
        raise ValueError("Gemini မှ JSON အဖြေမရရှိပါ။")

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Gemini အဖြေထဲမှာ JSON object မတွေ့ပါ။")

    return json.loads(cleaned[start : end + 1])


def safe_error_message(exc, api_key):
    """Display useful error details without exposing the user's API key."""
    message = str(exc)
    if api_key:
        message = message.replace(api_key, "[API KEY HIDDEN]")
    return f"{type(exc).__name__}: {message}"


def generate_with_fallback(client, contents):
    """Try the main model, then other stable Flash models on transient failures."""
    last_error = None
    retry_markers = (
        "503",
        "UNAVAILABLE",
        "500",
        "INTERNAL",
        "502",
        "504",
        "429",
        "RESOURCE_EXHAUSTED",
        "404",
        "NOT_FOUND",
    )

    for index, model_name in enumerate(FALLBACK_MODELS):
        if index:
            st.warning(
                f"{FALLBACK_MODELS[index - 1]} ачаалалтай байна. "
                f"{model_name} model-оор үргэлжлүүлэн оролдож байна..."
            )
            time.sleep(random.uniform(1.0, 2.5))

        try:
            return client.models.generate_content(
                model=model_name,
                contents=contents,
            )
        except Exception as exc:
            last_error = exc
            error_text = str(exc).upper()
            if not any(marker in error_text for marker in retry_markers):
                raise

    raise last_error


def extract_segments(client, video_path, video_duration):
    """Ask Gemini to detect and transcribe speech in any language with timestamps."""
    uploaded_file = client.files.upload(file=video_path)
    try:
        started_at = time.monotonic()
        while uploaded_file.state and uploaded_file.state.name == "PROCESSING":
            if time.monotonic() - started_at > 600:
                raise TimeoutError("ဗီဒီယို processing အချိန်ကြာလွန်းနေပါသည်။")
            time.sleep(5)
            uploaded_file = client.files.get(name=uploaded_file.name)

        if not uploaded_file.state or uploaded_file.state.name != "ACTIVE":
            raise RuntimeError("Google က video ဖိုင်ကို ပြင်ဆင်မရပါ။")

        prompt = f"""
You are an accurate multilingual speech transcriber for a dubbing workflow.
Listen to the uploaded video and identify the spoken language automatically.
Transcribe all audible dialogue and narration in its original language; it may be any language, not only English.
If speech is not intelligible but clearly readable subtitles are visible, transcribe those subtitles in their original language.
Split the transcript into short, natural subtitle cues, usually about 2 to 7 seconds each.
For every cue provide start and end timestamps in seconds from the beginning of the video.
Keep cues chronological, do not invent dialogue, and preserve the original language and script.
The video duration is approximately {video_duration:.2f} seconds.
Return ONLY valid JSON in exactly this shape:
{{"segments":[{{"start":0.0,"end":2.5,"text":"Original spoken words here"}}]}}
No markdown fences or extra commentary.
"""

        result = generate_with_fallback(
            client,
            [uploaded_file, prompt],
        )
        payload = parse_json_response(result.text)
        raw_segments = payload.get("segments")
        if not isinstance(raw_segments, list):
            raise ValueError("Gemini အဖြေမှာ segments စာရင်း မပါပါ။")

        segments = []
        for item in raw_segments:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            start = float(item.get("start", 0))
            end = float(item.get("end", 0))
            start = max(0.0, min(start, video_duration))
            end = min(video_duration, max(start + 0.15, end))
            if start >= video_duration:
                continue
            segments.append({"start": start, "end": end, "original_text": text})

        segments.sort(key=lambda row: row["start"])
        if not segments:
            raise ValueError(
                "ဗီဒီယိုထဲမှာ ခွဲထုတ်လို့ရတဲ့ စကားသံ သို့မဟုတ် စာတန်း မတွေ့ပါ။ "
                "အသံကြားရပြီး စကားပြောပါဝင်သော ဗီဒီယိုကို စမ်းကြည့်ပါ။"
            )
        return segments
    finally:
        try:
            client.files.delete(name=uploaded_file.name)
        except Exception:
            pass


def translate_segments(client, segments):
    """Translate in batches while requiring exactly one Myanmar line per cue."""
    translations = []
    batch_size = 30

    for batch_start in range(0, len(segments), batch_size):
        batch = segments[batch_start : batch_start + batch_size]
        source_lines = [row["original_text"] for row in batch]
        prompt = f"""
You are a professional multilingual-to-Myanmar video dubbing translator.
Translate every input line, regardless of its original language, into natural, concise spoken Burmese written in Myanmar script.
Do not output English, phonetic guides, numbering, explanations, or extra text.
Preserve meaning and names naturally. Keep the same number and order of lines.
Return ONLY valid JSON with one key called translations and an array of strings.
Input lines:
{json.dumps(source_lines, ensure_ascii=False)}
"""
        response = generate_with_fallback(client, prompt)
        parsed = parse_json_response(response.text)
        batch_translations = parsed.get("translations")
        if not isinstance(batch_translations, list):
            raise ValueError("Gemini ဘာသာပြန်အဖြေမှာ translations စာရင်း မပါပါ။")
        if len(batch_translations) != len(batch):
            raise ValueError(
                "ဘာသာပြန်စာကြောင်းအရေအတွက် မူရင်းစာကြောင်းနဲ့ မကိုက်ပါ။ "
                "ထပ်မံစမ်းကြည့်ပါ။"
            )
        translations.extend(str(line).strip() for line in batch_translations)

    return translations


async def save_tts_audio(text, voice, rate, pitch, output_path):
    communicator = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=f"{rate:+d}%",
        pitch=f"{pitch:+d}Hz",
    )
    await communicator.save(output_path)


# ---------- Sidebar controls ----------
st.sidebar.header("⚙️ အသံနှင့် Sync ချိန်ညှိမှု")
api_key = st.sidebar.text_input("Gemini API Key", type="password")
voice_option = st.sidebar.selectbox(
    "မြန်မာအသံ",
    ["my-MM-ThihaNeural (Male)", "my-MM-NilarNeural (Female)"],
)
voice_rate = st.sidebar.slider(
    "အသံပြောနှုန်း (%)",
    min_value=-30,
    max_value=50,
    value=0,
    step=5,
    help="အပေါင်းတန်ဖိုးက ပိုမြန်၊ အနုတ်တန်ဖိုးက ပိုနှေးစေပါတယ်။",
)
voice_pitch = st.sidebar.slider(
    "အသံလေသံ အနိမ့်/အမြင့် (Hz)",
    min_value=-20,
    max_value=20,
    value=0,
    step=1,
    help="အသံလေသံကို အနိမ့် သို့မဟုတ် အမြင့် ပြောင်းပေးပါတယ်။",
)
audio_volume = st.sidebar.slider(
    "အသံအတိုးအကျယ်",
    min_value=0.0,
    max_value=2.0,
    value=1.0,
    step=0.1,
)
clarity_boost = st.sidebar.slider(
    "အသံကြည်လင်မှု အား (EQ dB)",
    min_value=0,
    max_value=6,
    value=0,
    step=1,
    help="2.5–3 kHz အသံပိုင်းကို နည်းနည်းမြှင့်ပေးပါမယ်။ 0 ဆို EQ မထည့်ပါ။",
)
sync_offset = st.sidebar.slider(
    "အသံစတင်ချိန်ညှိ (စက္ကန့်)",
    min_value=-3.0,
    max_value=3.0,
    value=0.0,
    step=0.1,
    help="အပေါင်းဆို အသံကို နောက်ကျစေပြီး၊ အနုတ်ဆို အသံကို စောစေပါတယ်။",
)
fit_to_cue = st.sidebar.checkbox(
    "စာကြောင်းအသံကို subtitle အချိန်အတွင်း အံဝင်အောင် အရှိန်မြှင့်မည်",
    value=True,
    help="ဘာသာပြန်စာကြောင်းရှည်ပါက စကားပြောသံကို အနည်းငယ်မြန်စေနိုင်ပါတယ်။",
)
flip_video = st.sidebar.checkbox("ဗီဒီယိုကို ဘယ်/ညာလှန်မည်", value=False)


# ---------- Upload and analyze ----------
st.header("အဆင့် ၁ — ဗီဒီယိုတင်ပြီး မူရင်းစကားသံထုတ်ယူပါ")
uploaded_video = st.file_uploader(
    "ဗီဒီယိုဖိုင်ရွေးပါ",
    type=["mp4", "mov", "m4v"],
)

if uploaded_video:
    st.video(uploaded_video)

if uploaded_video and st.button("မူရင်းစကားသံထုတ်ယူပြီး မြန်မာလိုဘာသာပြန်မည်", type="primary"):
    if not api_key:
        st.error("Sidebar မှာ Gemini API Key ထည့်ပါ။")
    else:
        suffix = Path(uploaded_video.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_video:
            temp_video.write(uploaded_video.getbuffer())
            video_path = temp_video.name

        try:
            with st.spinner("ဗီဒီယိုကိုစစ်ဆေးပြီး စာသားနဲ့အချိန်တွေ ထုတ်ယူနေပါသည်..."):
                video_probe = VideoFileClip(video_path)
                try:
                    video_duration = float(video_probe.duration)
                finally:
                    video_probe.close()

                client = genai.Client(api_key=api_key)
                english_segments = extract_segments(
                    client,
                    video_path,
                    video_duration,
                )

            with st.spinner("မူရင်းစကားတွေကို မြန်မာလို ဘာသာပြန်နေပါသည်..."):
                translations = translate_segments(client, english_segments)

            rows = []
            for item, translation in zip(english_segments, translations):
                rows.append(
                    {
                        "Start (sec)": round(item["start"], 2),
                        "End (sec)": round(item["end"], 2),
                        "မူရင်းစာသား": item["original_text"],
                        "မြန်မာဘာသာပြန်": translation,
                    }
                )

            st.session_state["dub_video_path"] = video_path
            st.session_state["dub_video_duration"] = video_duration
            st.session_state["dub_rows"] = rows
            st.session_state.pop("dub_output_path", None)
            st.success(f"စာကြောင်း {len(rows)} ကြောင်း ထုတ်ယူပြီး ဘာသာပြန်ပြီးပါပြီ။")
        except Exception as exc:
            try:
                os.unlink(video_path)
            except OSError:
                pass
            st.error(safe_error_message(exc, api_key))


# ---------- Review editable timestamps and translations ----------
if "dub_rows" in st.session_state:
    st.header("အဆင့် ၂ — စာသားနှင့်အချိန်ကို စစ်ဆေးပြင်ဆင်ပါ")
    st.caption(
        "Gemini က အချိန်မှတ်တမ်းကို အလိုအလျောက် ခန့်မှန်းပေးပါမယ်။ "
        "ပိုတိကျစေရန် Start/End အချိန်နှင့် ဘာသာပြန်စာသားကို ဒီနေရာမှာ ပြင်နိုင်ပါတယ်။"
    )

    edited_rows = st.data_editor(
        pd.DataFrame(st.session_state["dub_rows"]),
        num_rows="dynamic",
        use_container_width=True,
        key="dub_subtitle_editor",
        column_config={
            "Start (sec)": st.column_config.NumberColumn(min_value=0.0, step=0.1),
            "End (sec)": st.column_config.NumberColumn(min_value=0.0, step=0.1),
            "မူရင်းစာသား": st.column_config.TextColumn(),
            "မြန်မာဘာသာပြန်": st.column_config.TextColumn(),
        },
    )

    st.header("အဆင့် ၃ — အချိန်ကိုက် မြန်မာအသံဖန်တီးပါ")
    if st.button("မြန်မာအသံတင်ပြီး ဗီဒီယိုထုတ်မည်", type="primary"):
        if not api_key:
            st.error("Sidebar မှာ Gemini API Key ထည့်ပါ။")
        else:
            voice_code = (
                "my-MM-ThihaNeural"
                if "Thiha" in voice_option
                else "my-MM-NilarNeural"
            )
            generated_audio_paths = []
            base_audio_clips = []
            timeline_audio_clips = []
            source_video = None
            final_video = None
            composite_audio = None

            try:
                usable_rows = []
                for row in edited_rows.to_dict(orient="records"):
                    text = str(row.get("မြန်မာဘာသာပြန်", "")).strip()
                    if not text:
                        continue
                    start = float(row.get("Start (sec)", 0))
                    end = float(row.get("End (sec)", 0))
                    if end <= start:
                        raise ValueError("စာကြောင်းတိုင်းရဲ့ End အချိန်က Start ထက် နောက်ကျရပါမယ်။")
                    usable_rows.append((start, end, text))

                usable_rows.sort(key=lambda item: item[0])
                if not usable_rows:
                    raise ValueError("မြန်မာဘာသာပြန်စာကြောင်း မရှိပါ။")

                progress = st.progress(0.0, text="မြန်မာအသံဖိုင်များ ဖန်တီးနေပါသည်...")
                for index, (start, end, text) in enumerate(usable_rows):
                    audio_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                    audio_temp.close()
                    generated_audio_paths.append(audio_temp.name)

                    asyncio.run(
                        save_tts_audio(
                            text=text,
                            voice=voice_code,
                            rate=voice_rate,
                            pitch=voice_pitch,
                            output_path=audio_temp.name,
                        )
                    )

                    audio_clip = AudioFileClip(audio_temp.name)
                    base_audio_clips.append(audio_clip)
                    cue_duration = max(0.2, end - start)

                    if fit_to_cue and audio_clip.duration > cue_duration:
                        speed_factor = audio_clip.duration / cue_duration
                        audio_clip = audio_clip.fx(vfx.speedx, speed_factor)

                    audio_clip = audio_clip.volumex(audio_volume)
                    clip_start = max(0.0, start + sync_offset)
                    audio_clip = audio_clip.set_start(clip_start)
                    timeline_audio_clips.append(audio_clip)

                    progress.progress(
                        (index + 1) / len(usable_rows),
                        text=f"အသံဖန်တီးနေသည် — {index + 1}/{len(usable_rows)}",
                    )

                source_video = VideoFileClip(st.session_state["dub_video_path"])
                video_for_render = (
                    source_video.fx(vfx.mirror_x)
                    if flip_video
                    else source_video
                )
                video_for_render = video_for_render.without_audio()

                composite_audio = CompositeAudioClip(timeline_audio_clips)
                composite_audio = composite_audio.set_duration(video_for_render.duration)

                audio_for_video = composite_audio
                if clarity_boost > 0:
                    raw_mix = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                    raw_mix.close()
                    eq_mix = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                    eq_mix.close()
                    generated_audio_paths.extend([raw_mix.name, eq_mix.name])

                    composite_audio.write_audiofile(
                        raw_mix.name,
                        fps=44100,
                        nbytes=2,
                        codec="pcm_s16le",
                        logger=None,
                    )
                    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
                    eq_filter = (
                        f"equalizer=f=3000:t=q:w=1:g={clarity_boost}"
                    )
                    subprocess.run(
                        [
                            ffmpeg_path,
                            "-y",
                            "-i",
                            raw_mix.name,
                            "-af",
                            eq_filter,
                            "-c:a",
                            "pcm_s16le",
                            eq_mix.name,
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    audio_for_video = AudioFileClip(eq_mix.name)
                    base_audio_clips.append(audio_for_video)
                    audio_for_video = audio_for_video.set_duration(
                        video_for_render.duration
                    )

                final_video = video_for_render.set_audio(audio_for_video)

                output_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                output_temp.close()
                final_video.write_videofile(
                    output_temp.name,
                    codec="libx264",
                    audio_codec="aac",
                    fps=video_for_render.fps or 24,
                    preset="medium",
                    threads=2,
                    logger=None,
                )

                st.session_state["dub_output_path"] = output_temp.name
                st.success("အချိန်ကိုက် မြန်မာအသံပါသော ဗီဒီယို ပြီးပါပြီ။")
            except Exception as exc:
                st.error(safe_error_message(exc, api_key))
            finally:
                if final_video is not None:
                    try:
                        final_video.close()
                    except Exception:
                        pass
                if source_video is not None:
                    try:
                        source_video.close()
                    except Exception:
                        pass
                if composite_audio is not None:
                    try:
                        composite_audio.close()
                    except Exception:
                        pass
                for clip in base_audio_clips:
                    try:
                        clip.close()
                    except Exception:
                        pass
                for path in generated_audio_paths:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass

if "dub_output_path" in st.session_state:
    output_path = st.session_state["dub_output_path"]
    if os.path.exists(output_path):
        st.video(output_path)
        with open(output_path, "rb") as output_file:
            st.download_button(
                label="⬇️ Dubbing ဗီဒီယိုကို ဒေါင်းလုဒ်လုပ်ပါ",
                data=output_file,
                file_name="myanmar_dubbed_video.mp4",
                mime="video/mp4",
        )
