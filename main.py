#!/usr/bin/env python3
"""
Minimal Wyoming ⇄ ElevenLabs Conversational-Agent bridge.
Tested on Windows 10/11 with Python 3.11, wyoming ≥0.8.0, websockets ≥12.
"""

import asyncio, os, json, base64, argparse, logging
import httpx, time, os, math
import websockets
from wyoming.audio import AudioFormat, AudioStart, AudioChunk, AudioStop
from wyoming.event import Event
from wyoming.info import (
    Info, Attribution, TtsProgram, TtsVoice,
    SndProgram, AsrModel, AsrProgram
)
from wyoming.server import AsyncServer, AsyncEventHandler

_LOGGER = logging.getLogger("wyoming-elevenlabs")

ELEVEN_WSS = "wss://api.elevenlabs.io/v1/convai/conversation"

CHUNK_PCM_BYTES   = 640            # 20 ms  (16 000 Hz × 0.02 s × 2 bytes)
CHUNK_PCM_B64_LEN = math.ceil(CHUNK_PCM_BYTES / 3) * 4   #  ≈ 852 chars

async def fetch_signed_url(agent_id: str, api_key: str) -> str:
    url = "https://api.elevenlabs.io/v1/convai/conversation/get-signed-url"
    headers = {"xi-api-key": api_key}
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
    def __init__(self, agent_id: str, api_key: str | None):
        self.agent_id = agent_id
        self.api_key = api_key
        self.websocket: websockets.WebSocketClientProtocol | None = None
        self.ready = asyncio.Event()
        self._sent_start = False

    async def connect(self):
        print(f"Connecting to ElevenLabs agent {self.agent_id}…")
        signed_ws = await get_signed_url(self.agent_id, self.api_key)
        self.websocket = await websockets.connect(signed_ws)   # no headers!
        await self.websocket.send(json.dumps({
            "type": "conversation_initiation_client_data"
        }))
        self.ready.set()
        _LOGGER.debug("Opened ElevenLabs session")

    async def send_pcm(self, chunk: AudioChunk):
        """Forward raw PCM from Wyoming to Eleven."""
        print(f"Sending audio chunk to ElevenLabs agent {self.agent_id}…")
        await self.ready.wait()

        pcm = chunk.audio
        for i in range(0, len(pcm), CHUNK_PCM_BYTES):
            piece = pcm[i:i + CHUNK_PCM_BYTES]
            if not piece:
                continue

            message = {
                "user_audio_chunk": base64.b64encode(piece).decode()
            }
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

        while True:
            async for msg in self.websocket:
                frame = json.loads(msg)
                ftype = frame.get("type")
                print(f"Received {ftype} frame from ElevenLabs")

                # keep-alive
                if ftype == "ping":
                    await self.websocket.send(json.dumps({
                        "type": "pong",
                        "event_id": frame["ping_event"]["event_id"]
                    }))
                    continue

                # TTS audio back from ElevenLabs
                if ftype == "audio":
                    pcm = base64.b64decode(
                        frame["audio_event"]["audio_base_64"])
                    print(f"Received {len(pcm)} bytes of audio")
                    if not self._sent_start:
                        self._sent_start = True
                        print("AudioStart")
                        yield AudioStart(rate=16000,
                                            width=2, channels=1).event()
                        
                    print("AudioChunk")
                    yield AudioChunk(rate=16000, width=2,
                                        channels=1, audio=pcm).event()
                    continue

                # Conversation turn finished
                if ftype == "agent_response" and self._sent_start:
                    self._sent_start = False
                    yield AudioStop().event()
                    continue

                # (Optional) user transcript event
                if ftype == "user_transcript":
                    text = frame["user_transcription_event"] \
                                ["user_transcript"]
                    yield Event("transcript", data={
                        "text": text,
                        "is_final": True,
                        "confidence": 0.9
                    })

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
        self._session: ElevenSession | None = None
        self._eleven_task: asyncio.Task | None = None

    # Wyoming → ElevenLabs -----------------------------------------------------
    async def handle_event(self, event: Event) -> bool:
        print(f"Received event: {event.type}")
        etype = event.type
        if etype == "describe":                 # handshake step 1
            attribution = Attribution(name="ElevenLabs", url="https://elevenlabs.io")
            voice = TtsVoice(
                name="default",
                attribution=attribution,
                installed=True,
                languages=["en"],
                description="ElevenLabs streaming voice",
                version="1.0",
            )

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

            tts_service = TtsProgram(
                name="eleven_tts",
                attribution=attribution,
                installed=True,
                description="ElevenLabs Conversational-AI bridge",
                voices=[voice],
                version="1.0",
            )

            snd_service = SndProgram(
                name="pcm16k",
                attribution=attribution,
                installed=True,
                description="Raw PCM 16 kHz",
                snd_format=AudioFormat(rate=16000, width=2, channels=1),
                version="1.0",
            )
            info = Info(
                asr=[asr_service],
                tts=[tts_service], 
                snd=[snd_service]).event()
            print(f"Sending info event: {info}")
            await self.write_event(info)        # handshake step 2
            return True

        # Home Assistant will send 'transcribe' once before audio-start
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
            async for wy_event in self._session.receive_events():
                await self.write_event(wy_event.event() if hasattr(wy_event, "event") else wy_event)
        except Exception as err:
            _LOGGER.exception("Error reading from ElevenLabs: %s", err)

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