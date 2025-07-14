#!/usr/bin/env python3
"""
Minimal Wyoming ⇄ ElevenLabs Conversational-Agent bridge.
Tested on Windows 10/11 with Python 3.11, wyoming ≥0.8.0, websockets ≥12.
"""

import asyncio, os, json, base64, argparse, logging
import httpx, time, os, math
import websockets
from typing import Optional
from wyoming.audio import AudioFormat, AudioStart, AudioChunk, AudioStop
from wyoming.event import Event
from wyoming.info import (
    Info, Attribution,
    SndProgram, AsrModel, AsrProgram, TtsProgram, TtsVoice
)
from wyoming.server import AsyncServer, AsyncEventHandler
from wyoming.snd import Played

_LOGGER = logging.getLogger("wyoming-elevenlabs")

ELEVEN_WSS = "wss://api.elevenlabs.io/v1/convai/conversation"

CHUNK_PCM_BYTES   = 640            # 20 ms  (16 000 Hz × 0.02 s × 2 bytes)
CHUNK_PCM_B64_LEN = math.ceil(CHUNK_PCM_BYTES / 3) * 4   #  ≈ 852 chars

async def fetch_signed_url(agent_id: str, api_key: str) -> str:
    url = "https://api.elevenlabs.io/v1/convai/conversation/get-signed-url"
    headers = {}
    if api_key:
        headers["xi-api-key"] = api_key
    params  = {"agent_id": agent_id}
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url, headers=headers, params=params)
    r.raise_for_status()
    return r.json()["signed_url"]

_SIGNED_CACHE = {"url": None, "expires": 0}          # NEW

async def get_signed_url(agent_id, api_key):
    now = time.time()
    if _SIGNED_CACHE["url"] and now < _SIGNED_CACHE["expires"]:
        return _SIGNED_CACHE["url"]
    signed = await fetch_signed_url(agent_id, api_key)
    _SIGNED_CACHE.update({"url": signed, "expires": now + 14*60})
    return signed

# -----------------------------------------------------------------------------
class ElevenSession:
    """Maintains one agent WebSocket per Wyoming client."""
    def __init__(self, agent_id: str, api_key: Optional[str]):
        self.agent_id = agent_id
        self.api_key = api_key
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.ready = asyncio.Event()
        self._sent_start = False

    async def connect(self):
        print(f"Connecting to ElevenLabs agent {self.agent_id}…")
        signed_ws = await get_signed_url(self.agent_id, self.api_key)
        print(f"Using signed WebSocket URL: {signed_ws[:50]}...")
        self.websocket = await websockets.connect(signed_ws)   # no headers!
        
        # First wait for the conversation_initiation_metadata_event
        init_metadata = await self.websocket.recv()
        metadata_frame = json.loads(init_metadata)
        print(f"Received metadata event: {json.dumps(metadata_frame, indent=2)}")
        
        # Extract audio format information
        self.agent_output_format = metadata_frame.get("conversation_initiation_metadata_event", {}).get("agent_output_audio_format", "pcm_22050")
        self.user_input_format = metadata_frame.get("conversation_initiation_metadata_event", {}).get("user_input_audio_format", "pcm_16000")
        print(f"Agent expects output format: {self.agent_output_format}")
        print(f"Agent expects input format: {self.user_input_format}")
        
        # Now send our initiation data
        init_message = {"type": "conversation_initiation_client_data"}
        print(f"Sending initialization message: {json.dumps(init_message)}")
        await self.websocket.send(json.dumps(init_message))
        
        self.ready.set()
        _LOGGER.debug("Opened ElevenLabs session")

    async def send_pcm(self, chunk: AudioChunk):
        """Forward raw PCM from Wyoming to Eleven."""
        print(f"Sending audio chunk to ElevenLabs agent {self.agent_id}… (rate={chunk.rate}, width={chunk.width}, channels={chunk.channels})")
        await self.ready.wait()

        pcm = chunk.audio
        print(f"Audio chunk details: {len(pcm)} bytes, sample rate: {chunk.rate}")
        
        # ANALYZE OUTGOING AUDIO TO ELEVENLABS
        if len(pcm) >= 2:
            import struct
            samples = [struct.unpack('<h', pcm[i:i+2])[0] for i in range(0, min(len(pcm), 20), 2)]
            avg_amp = sum(abs(s) for s in samples) / len(samples) if samples else 0
            print(f"Outgoing audio analysis: avg_amplitude={avg_amp:.1f}, first_samples={samples}")
            if avg_amp < 50:
                print("WARNING: Very quiet audio being sent to ElevenLabs!")
        
        # Send conversation trigger on first audio chunk
        if not self._sent_start:
            # Send conversation initiation response
            print("Sending conversation_initiation_response")
            init_response = {"type": "conversation_initiation_response", "conversation_initiation_response": {"success": True}}
            await self.websocket.send(json.dumps(init_response))
            
            # Also try sending a conversation configuration
            print("Sending conversation configuration")
            config_message = {
                "type": "conversation_configuration",
                "conversation_configuration": {
                    "audio_output_enabled": True,
                    "audio_input_enabled": True
                }
            }
            await self.websocket.send(json.dumps(config_message))
            
            self._sent_start = True
        
        for i in range(0, len(pcm), CHUNK_PCM_BYTES):
            piece = pcm[i:i + CHUNK_PCM_BYTES]
            if not piece:
                continue

            message = {
                "user_audio_chunk": base64.b64encode(piece).decode()
            }
            print(f"Sending message to ElevenLabs...")
            await self.websocket.send(json.dumps(message))
            
        print(f"Sent {len(pcm)} bytes of audio to ElevenLabs agent {self.agent_id}")

    async def receive_events(self):
        """
        Async generator producing Wyoming `AudioStart/Chunk/Stop`
        (and optional `transcript` events) from ElevenLabs frames.
        Reconnects transparently if the socket drops.
        """
        await self.ready.wait()
        self._sent_start = False
        audio_buffer = []  # Buffer audio chunks until complete response

        while True:
            async for msg in self.websocket:
                frame = json.loads(msg)
                ftype = frame.get("type")
                print(f"*** ELEVENLABS FRAME: {ftype} ***")
                print(f"Full frame: {json.dumps(frame, indent=2)}")

                # keep-alive
                if ftype == "ping":
                    await self.websocket.send(json.dumps({
                        "type": "pong",
                        "event_id": frame["ping_event"]["event_id"]
                    }))
                    continue

                # TTS audio back from ElevenLabs - buffer it
                if ftype == "audio":
                    pcm = base64.b64decode(
                        frame["audio_event"]["audio_base_64"])
                    print(f"Buffering {len(pcm)} bytes of audio")
                    
                    # COMPREHENSIVE AUDIO FORMAT ANALYSIS WITH STRUCT PARSING
                    if len(pcm) > 0:
                        import struct
                        print(f"=== COMBINED AUDIO ANALYSIS ===")
                        print(f"Raw bytes length: {len(pcm)}")
                        print(f"First 32 bytes (hex): {pcm[:32].hex()}")
                        
                        # Try to parse as WAV format first
                        is_wav = False
                        if len(pcm) >= 44 and pcm.startswith(b'RIFF'):
                            try:
                                # Parse WAV header using struct
                                wav_header = struct.unpack('<4sI4s4sIHHIIHH4sI', pcm[:44])
                                riff_id, file_size, wave_id, fmt_id, fmt_size, audio_format, channels, sample_rate, byte_rate, block_align, bits_per_sample, data_id, data_size = wav_header
                                
                                print(f"*** WAV FORMAT DETECTED ***")
                                print(f"  Audio format: {audio_format} (1=PCM)")
                                print(f"  Channels: {channels}")
                                print(f"  Sample rate: {sample_rate}Hz")
                                print(f"  Bit depth: {bits_per_sample}-bit")
                                print(f"  Byte rate: {byte_rate}")
                                print(f"  Block align: {block_align}")
                                print(f"  Data size: {data_size} bytes")
                                print(f"  Expected duration: {data_size / byte_rate:.2f}s")
                                is_wav = True
                                
                                # Parse audio data (skip 44-byte header)
                                audio_data = pcm[44:44+data_size]
                                if len(audio_data) >= 2:
                                    if bits_per_sample == 16:
                                        samples = [struct.unpack('<h', audio_data[i:i+2])[0] for i in range(0, len(audio_data), 2)]
                                    elif bits_per_sample == 8:
                                        samples = [struct.unpack('<B', audio_data[i:i+1])[0] - 128 for i in range(len(audio_data))]
                                    else:
                                        samples = []
                                    
                                    if samples:
                                        print(f"  Sample count: {len(samples)}")
                                        print(f"  Sample range: {min(samples)} to {max(samples)}")
                                        print(f"  Average amplitude: {sum(abs(s) for s in samples) / len(samples):.1f}")
                                        
                                        # Check for silence
                                        silent_samples = sum(1 for s in samples if abs(s) < 100)
                                        silence_pct = 100 * silent_samples / len(samples)
                                        print(f"  Silence: {silent_samples}/{len(samples)} ({silence_pct:.1f}%)")
                                        
                                        # Show first few samples
                                        print(f"  First 10 samples: {samples[:10]}")
                                        
                            except Exception as e:
                                print(f"  WAV parsing error: {e}")
                                is_wav = False
                        
                        if not is_wav:
                            print(f"*** RAW PCM DATA (assumed) ***")
                            # Analyze as different PCM formats
                            for rate, desc in [(16000, "16kHz"), (22050, "22kHz"), (24000, "24kHz"), (44100, "44.1kHz"), (48000, "48kHz")]:
                                duration = len(pcm) / (rate * 2)  # Assuming 16-bit mono
                                print(f"  If {desc} 16-bit mono: {duration:.2f}s duration")
                            
                            # Parse as 16-bit signed PCM
                            if len(pcm) >= 2:
                                samples = [struct.unpack('<h', pcm[i:i+2])[0] for i in range(0, len(pcm), 2)]
                                if samples:
                                    print(f"  As 16-bit PCM:")
                                    print(f"    Sample count: {len(samples)}")
                                    print(f"    Range: {min(samples)} to {max(samples)}")
                                    print(f"    Avg amplitude: {sum(abs(s) for s in samples) / len(samples):.1f}")
                                    
                                    silent_samples = sum(1 for s in samples if abs(s) < 100)
                                    silence_pct = 100 * silent_samples / len(samples)
                                    print(f"    Silence: {silent_samples}/{len(samples)} ({silence_pct:.1f}%)")
                                    print(f"    First 10 samples: {samples[:10]}")
                        
                        print(f"=== END COMBINED ANALYSIS ===")
                    
                    audio_buffer.append(pcm)
                    continue

                # If we receive user transcript or agent response and have buffered audio, send it directly
                if audio_buffer and ftype in ["user_transcript", "agent_response"]:
                    print(f"Received {ftype} with buffered audio - sending directly to Home Assistant")
                    
                    # Combine all buffered audio and send it immediately
                    combined_pcm = b''.join(audio_buffer)
                    print(f"Sending {len(combined_pcm)} bytes of audio directly to Home Assistant")
                    
                    # Validate audio data isn't empty
                    if len(combined_pcm) == 0:
                        print("WARNING: No audio data to send!")
                        audio_buffer.clear()
                        continue
                    
                    # COMPREHENSIVE AUDIO FORMAT ANALYSIS
                    print(f"=== COMBINED AUDIO ANALYSIS ===")
                    print(f"Total audio length: {len(combined_pcm)} bytes")
                    
                    # Check if it's WAV format
                    if combined_pcm.startswith(b'RIFF'):
                        print("DETECTED: WAV format audio!")
                        # Parse WAV header
                        if len(combined_pcm) >= 44:
                            import struct
                            try:
                                # WAV header parsing
                                riff, size, wave = struct.unpack('<4sI4s', combined_pcm[:12])
                                fmt_chunk, fmt_size = struct.unpack('<4sI', combined_pcm[12:20])
                                fmt_data = struct.unpack('<HHIIHH', combined_pcm[20:36])
                                audio_format, channels, sample_rate, byte_rate, block_align, bits_per_sample = fmt_data
                                
                                print(f"WAV Format: {audio_format} (1=PCM)")
                                print(f"Channels: {channels}")
                                print(f"Sample Rate: {sample_rate} Hz")
                                print(f"Bits per sample: {bits_per_sample}")
                                print(f"Byte rate: {byte_rate}")
                                
                                # Use WAV header info for format
                                sample_rate = sample_rate
                                sample_width = bits_per_sample // 8
                                channels = channels
                                
                                # Find data chunk
                                data_start = 36
                                while data_start < len(combined_pcm) - 8:
                                    chunk_id = combined_pcm[data_start:data_start+4]
                                    chunk_size = struct.unpack('<I', combined_pcm[data_start+4:data_start+8])[0]
                                    if chunk_id == b'data':
                                        audio_data = combined_pcm[data_start+8:data_start+8+chunk_size]
                                        print(f"Found data chunk: {len(audio_data)} bytes of audio data")
                                        combined_pcm = audio_data  # Use only the audio data
                                        break
                                    data_start += 8 + chunk_size
                                    
                            except Exception as e:
                                print(f"Error parsing WAV header: {e}")
                                # Fall back to assumptions
                                sample_rate = 24000
                                sample_width = 2
                                channels = 1
                    else:
                        print("DETECTED: Raw PCM audio")
                        # Test different sample rates by analyzing the data
                        if len(combined_pcm) >= 4:
                            samples_16bit = [int.from_bytes(combined_pcm[i:i+2], 'little', signed=True) for i in range(0, min(100, len(combined_pcm)), 2)]
                            max_amplitude = max(abs(s) for s in samples_16bit) if samples_16bit else 0
                            print(f"Max amplitude (16-bit interpretation): {max_amplitude}")
                            
                            # ElevenLabs typically uses 24kHz, but let's verify
                            duration_at_24k = len(combined_pcm) / (24000 * 2 * 1)  # 24kHz, 16-bit, mono
                            duration_at_16k = len(combined_pcm) / (16000 * 2 * 1)  # 16kHz, 16-bit, mono
                            duration_at_22k = len(combined_pcm) / (22050 * 2 * 1)  # 22kHz, 16-bit, mono
                            
                            print(f"Duration if 24kHz: {duration_at_24k:.2f}s")
                            print(f"Duration if 16kHz: {duration_at_16k:.2f}s") 
                            print(f"Duration if 22kHz: {duration_at_22k:.2f}s")
                        
                        # Default assumptions for ElevenLabs
                        sample_rate = 24000  # Most likely
                        sample_width = 2     # 16-bit = 2 bytes
                        channels = 1         # mono
                    
                    print(f"Using audio format: {sample_rate}Hz, {sample_width}-byte, {channels}-channel")
                    print(f"Final audio data: {len(combined_pcm)} bytes")
                    print(f"=== END COMBINED ANALYSIS ===")
                    
                    # Send audio start with detected/determined format
                    yield AudioStart(rate=sample_rate, width=sample_width, channels=channels)
                    
                    # Send audio in smaller chunks for better streaming
                    chunk_size = 512  # Smaller chunks for better real-time performance  
                    for i in range(0, len(combined_pcm), chunk_size):
                        chunk_data = combined_pcm[i:i + chunk_size]
                        yield AudioChunk(rate=sample_rate, width=sample_width, channels=channels, audio=chunk_data)
                    
                    # Send audio stop
                    yield AudioStop()
                    
                    # CRITICAL: Send Played event to tell Home Assistant audio finished playing
                    # This is required for Home Assistant to know the satellite finished playback
                    print("=== ABOUT TO YIELD PLAYED EVENT ===")
                    played_event = Event("played", data={})
                    yield played_event
                    print("=== YIELDED PLAYED EVENT ===")
                    
                    print("=== SENT ELEVENLABS AUDIO DIRECTLY TO HOME ASSISTANT ===")
                    print("=== SENT PLAYED EVENT TO COMPLETE AUDIO PLAYBACK ===")
                    
                    # Clear the audio buffer
                    audio_buffer.clear()
                    continue

                # (Optional) user transcript event
                if ftype == "user_transcript":
                    text = frame["user_transcription_event"]["user_transcript"]
                    print(f"User transcript: {text}")
                    yield Event("transcript", data={
                        "text": text,
                        "is_final": True,
                        "confidence": 0.9
                    })
                    
                # Log any other frame types we might be missing
                if ftype not in ["ping", "audio", "user_transcript", "agent_response", "vad_score", "mcp_connection_status"]:
                    print(f"Unhandled frame type: {ftype}")
                    print(f"Frame content: {json.dumps(frame, indent=2)}")
                    
                # Special debug for all frames
                if ftype == "agent_response":
                    response_text = frame.get('agent_response_event', {}).get('agent_response', 'No response text')
                    print(f"Agent response received: {response_text}")
                    if not audio_buffer:
                        print("WARNING: Agent responded but no audio was buffered!")
                    
                # Debug VAD scores with more detail
                if ftype == "vad_score":
                    vad_score = frame.get("vad_score_event", {}).get("vad_score", 0)
                    if vad_score > 0.3:
                        print(f"VAD Score: {vad_score:.4f} - STRONG VOICE DETECTED")
                    elif vad_score > 0.1:
                        print(f"VAD Score: {vad_score:.4f} - Weak voice detected")
                    else:
                        print(f"VAD Score: {vad_score:.4f} - Voice not detected")
                        
                # Check for conversation end or timeout
                if ftype == "conversation_ended":
                    print("ElevenLabs conversation ended")
                    
                # Debug agent thinking/processing
                if ftype == "internal_tentative_agent_response":
                    print("ElevenLabs agent is thinking/processing...")

    async def close(self):
        if self.websocket:
            await self.websocket.close()

# -----------------------------------------------------------------------------
class BridgeHandler(AsyncEventHandler):
    """One instance per Wyoming TCP client."""
    def __init__(self, reader, writer, agent_id, api_key):
        super().__init__(reader, writer)
        self.agent_id = agent_id
        self.api_key = api_key
        self._session: Optional[ElevenSession] = None
        self._eleven_task: Optional[asyncio.Task] = None

    # Wyoming → ElevenLabs -----------------------------------------------------
    async def handle_event(self, event: Event) -> bool:
        print(f"Received event: {event.type}")
        etype = event.type
        if etype == "describe":                 # handshake step 1
            attribution = Attribution(name="ElevenLabs", url="https://elevenlabs.io")

            stt_model = AsrModel(name="eleven_stt",
                installed=True,
                languages=["en"],
                attribution=attribution,
                description="ElevenLabs streaming STT model",
                version="1.0",)

            asr_service = AsrProgram(name="eleven_stt",
                installed=True,
                attribution=attribution,
                description="ElevenLabs streaming STT",
                models=[stt_model],
                version="1.0",)

            # Add TTS capabilities - this is required for Home Assistant Voice Preview Edition
            tts_voice = TtsVoice(
                name="default",
                description="ElevenLabs default voice",
                attribution=attribution,
                installed=True,
                version="1.0",
                languages=["en"]
            )

            tts_service = TtsProgram(
                name="eleven_tts",
                attribution=attribution,
                installed=True,
                description="ElevenLabs TTS",
                voices=[tts_voice],
                version="1.0",
            )

            snd_service = SndProgram(
                name="elevenlabs_snd",
                attribution=attribution,
                installed=True,
                description="ElevenLabs TTS Audio",
                # Audio format options:
                # rate=16000  - 16kHz (lower quality, smaller bandwidth)
                # rate=22050  - 22kHz (medium quality)
                # rate=44100  - 44.1kHz (CD quality)
                # rate=48000  - 48kHz (high quality)
                # width=2     - 16-bit audio (2 bytes per sample)
                # channels=1  - mono audio
                snd_format=AudioFormat(rate=44100, width=2, channels=1),
                version="1.0",
            )
            info = Info(
                asr=[asr_service],
                tts=[tts_service],  # Add TTS capability back
                snd=[snd_service]).event()
            print(f"Sending info event: {info}")
            await self.write_event(info)        # handshake step 2
            return True

        # Home Assistant will send 'transcribe' once before audio-start
        if etype == "transcribe":
            # Set up for transcription mode (wake word detection)
            print("Received transcribe event - ready for wake word detection")
            return True

        if etype == "synthesize":
            # Handle TTS requests - but we don't actually need to do anything
            # since our ElevenLabs agent handles TTS internally
            print("Received synthesize event - ElevenLabs agent will handle TTS internally")
            return True

        if etype == "audio-start":
            self._session = ElevenSession(self.agent_id, self.api_key)
            await self._session.connect()
            # Start background coroutine that pumps Eleven → Wyoming
            self._eleven_task = asyncio.create_task(self._pump_from_eleven())
            return True

        if etype == "audio-chunk" and self._session:
            chunk = AudioChunk.from_event(event)            #:contentReference[oaicite:6]{index=6}
            await self._session.send_pcm(chunk)
            return True

        if etype == "audio-stop" and self._session:
            # Nothing special to do; keep session open for next user turn
            return True

        return True  # keep connection alive

    # ElevenLabs → Wyoming -----------------------------------------------------
    async def _pump_from_eleven(self):
        assert self._session is not None
        try:
            print("=== STARTING AUDIO PUMP FROM ELEVENLABS TO HOME ASSISTANT ===")
            async for wy_event in self._session.receive_events():
                try:
                    print(f"About to write event: {type(wy_event).__name__}")
                    if hasattr(wy_event, 'event'):
                        event_to_write = wy_event.event()
                        print(f"Writing event with .event(): {type(event_to_write).__name__}")
                        await self.write_event(event_to_write)
                        print(f"Successfully wrote event: {type(event_to_write).__name__}")
                    else:
                        print(f"Writing event directly: {type(wy_event).__name__}")
                        await self.write_event(wy_event)
                        print(f"Successfully wrote event: {type(wy_event).__name__}")
                except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError) as write_err:
                    _LOGGER.warning("Connection to Home Assistant lost during audio transmission: %s", write_err)
                    print(f"HOME ASSISTANT CONNECTION ERROR: {write_err}")
                    # Don't break the loop, just log and continue
                    continue
                except Exception as write_err:
                    _LOGGER.error("Error writing to Home Assistant: %s", write_err)
                    print(f"GENERAL ERROR WRITING TO HOME ASSISTANT: {write_err}")
                    continue
        except Exception as err:
            _LOGGER.exception("Error reading from ElevenLabs: %s", err)
            print(f"ERROR READING FROM ELEVENLABS: {err}")

    async def disconnect(self):
        if self._session:
            await self._session.close()
        if self._eleven_task:
            self._eleven_task.cancel()

# -----------------------------------------------------------------------------
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", default="tcp://0.0.0.0:10700",
                        help="tcp://host:port or unix://…")
    parser.add_argument("--agent-id", default=os.getenv("ELEVEN_AGENT_ID"),
                        help="ElevenLabs agent ID")
    parser.add_argument("--api-key", default=os.getenv("ELEVEN_API_KEY"),
                        help="API key (omit for public agents)")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if not args.agent_id:
        parser.error("--agent-id is required. Set ELEVEN_AGENT_ID environment variable or use --agent-id")

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    server = AsyncServer.from_uri(args.listen)               #:contentReference[oaicite:7]{index=7}
    handler_factory = lambda r, w: BridgeHandler(
        r, w, agent_id=args.agent_id, api_key=args.api_key
    )
    _LOGGER.info("Listening on %s", args.listen)
    await server.run(handler_factory)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass