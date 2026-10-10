import asyncio
import json
import math
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
                f"{FALLBACK_MODELS[index - 1]} အလုပ်များနေပါသည်။ "
                f"{model_name} model ဖြင့် ဆက်လက်ကြိုးစားနေပါသည်..."
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
    """Transcribe the entire video's audio in short chunks with global timestamps."""
    chunk_seconds = 45
    source_clip = VideoFileClip(video_path)
    if source_clip.audio is None:
        source_clip.close()
        raise ValueError("ဒီဗီဒီယိုဖိုင်မှာ အသံလမ်းကြောင်း မပါပါ။")

    chunk_count = max(1, math.ceil(video_duration / chunk_seconds))
    segments = []
    progress = st.progress(0.0, text="ဗီဒီယိုအသံကို အပိုင်းလိုက်စစ်ဆေးနေပါသည်...")

    try:
        for chunk_index in range(chunk_count):
            chunk_start = chunk_index * chunk_seconds
            chunk_end = min(video_duration, chunk_start + chunk_seconds)
            chunk_duration = chunk_end - chunk_start

            audio_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            audio_temp.close()
            uploaded_file = None
            try:
                audio_part = source_clip.audio.subclip(chunk_start, chunk_end)
                audio_part.write_audiofile(
                    audio_temp.name,
                    fps=16000,
                    nbytes=2,
                    codec="pcm_s16le",
                    logger=None,
                )

                uploaded_file = client.files.upload(file=audio_temp.name)
                started_at = time.monotonic()
                while uploaded_file.state and uploaded_file.state.name == "PROCESSING":
                    if time.monotonic() - started_at > 300:
                        raise TimeoutError("အသံအပိုင်းကို ပြင်ဆင်ချိန် များလွန်းနေပါသည်။")
                    time.sleep(3)
                    uploaded_file = client.files.get(name=uploaded_file.name)

                if not uploaded_file.state or uploaded_file.state.name != "ACTIVE":
                    raise RuntimeError("Google က အသံအပိုင်းကို ပြင်ဆင်မရပါ။")

                prompt = f"""
You are an accurate multilingual speech transcriber for video dubbing.
Listen to this audio and identify the language automatically.
Transcribe all audible speech in its original language and script; it may be any language.
Split it into short subtitle cues, usually 2 to 7 seconds each.
Return start and end timestamps in seconds relative to the beginning of this audio clip.
The clip is {chunk_duration:.2f} seconds long. Keep cues chronological and do not invent speech.
Return ONLY valid JSON in this exact shape:
{{"segments":[{{"start":0.0,"end":2.5,"text":"Original spoken words here"}}]}}
If this clip contains no speech, return {{"segments":[]}}.
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

                for item in raw_segments:
                    if not isinstance(item, dict):
                        continue
                    text = str(item.get("text", "")).strip()
                    if not text:
                        continue
                    local_start = float(item.get("start", 0))
                    local_end = float(item.get("end", 0))
                    local_start = max(0.0, min(local_start, chunk_duration))
                    local_end = min(
                        chunk_duration,
                        max(local_start + 0.15, local_end),
                    )
                    if local_start >= chunk_duration:
                        continue
                    segments.append(
                        {
                            "start": chunk_start + local_start,
                            "end": chunk_start + local_end,
                            "original_text": text,
                        }
                    )
            finally:
                if uploaded_file is not None:
                    try:
                        client.files.delete(name=uploaded_file.name)
                    except Exception:
                        pass
                try:
                    os.unlink(audio_temp.name)
                except OSError:
                    pass

            progress.progress(
                (chunk_index + 1) / chunk_count,
                text=f"အသံအပိုင်း {chunk_index + 1}/{chunk_count} ကို စစ်ဆေးပြီးပါပြီ",
            )

        segments.sort(key=lambda row: row["start"])
        if not segments:
            raise ValueError(
                "ဗီဒီယိုရဲ့ အပိုင်းအားလုံးကို စစ်ပြီးပါပြီ၊ ဒါပေမယ့် စကားပြောသံ မတွေ့ပါ။ "
                "အသံပါပြီး စကားပြောသံကြားရတဲ့ ဗီဒီယိုနဲ့ ထပ်စမ်းပါ။"
            )
        return segments
    finally:
        source_clip.close()


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


def format_srt_timestamp(seconds):
    total_ms = max(0, int(round(float(seconds) * 1000)))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{whole_seconds:02},{milliseconds:03}"


def build_srt(rows, text_column, offset=0.0):
    entries = []
    for row in rows:
        text = str(row.get(text_column, "")).strip()
        if not text or text.lower() == "nan":
            continue
        start = max(0.0, float(row["Start (sec)"]) + offset)
        end = max(start + 0.1, float(row["End (sec)"]) + offset)
        entries.append((start, end, text))

    entries.sort(key=lambda entry: entry[0])
    blocks = []
    for index, (start, end, text) in enumerate(entries, start=1):
        blocks.append(
            f"{index}\n"
            f"{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}\n"
            f"{text}"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


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
cloud_light_render = st.sidebar.checkbox(
    "Cloud CPU လျှော့ရန် 720p / 24fps ашигမည်",
    value=True,
    help="Streamlit Cloud တွင် render ကို ပိုမြန်စေရန် resolution နှင့် frame rate ကို လျှော့ပေးပါတယ်။",
)


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

    subtitle_rows = edited_rows.to_dict(orient="records")
    original_srt = build_srt(subtitle_rows, "မူရင်းစာသား")
    myanmar_srt = build_srt(
        subtitle_rows,
        "မြန်မာဘာသာပြန်",
        offset=sync_offset,
    )
    srt_left, srt_right = st.columns(2)
    with srt_left:
        st.download_button(
            "မူရင်းဘာသာ SRT ဒေါင်းလုဒ်",
            data=original_srt.encode("utf-8-sig"),
            file_name="original_subtitles.srt",
            mime="text/plain",
        )
    with srt_right:
        st.download_button(
            "မြန်မာဘာသာ SRT ဒေါင်းလုဒ်",
            data=myanmar_srt.encode("utf-8-sig"),
            file_name="myanmar_subtitles.srt",
            mime="text/plain",
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

                    # Edge TTS нь mono MP3 гаргадаг. MoviePy 1.x-д mono clip-үүдийг
                    # шууд нийлүүлэхэд audio timeline буруу урттай болохоос сэргийлж stereo болгоно.
                    stereo_temp = tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".wav",
                    )
                    stereo_temp.close()
                    generated_audio_paths.append(stereo_temp.name)
                    subprocess.run(
                        [
                            imageio_ffmpeg.get_ffmpeg_exe(),
                            "-y",
                            "-i",
                            audio_temp.name,
                            "-ac",
                            "2",
                            "-ar",
                            "44100",
                            "-c:a",
                            "pcm_s16le",
                            stereo_temp.name,
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    )

                    audio_clip = AudioFileClip(stereo_temp.name)
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
