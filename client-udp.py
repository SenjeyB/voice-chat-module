import socket
import threading
import pyaudio
import platform
import signal
import math
import struct
import time
import os
import numpy as np
from collections import deque

from protocol import DataType, Protocol


class Client:
    def __init__(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.connected = False
        self.name = input('Enter the name of the client --> ')

        while 1:
            try:
                self.target_ip = input('Enter IP address of server --> ')
                self.target_port = int(input('Enter target port of server --> '))
                self.room = int(input('Enter the id of room  --> '))
                self.server = (self.target_ip, self.target_port)
                self.connect_to_server()
                break
            except Exception as err:
                print(err)
                print("Couldn't connect to server...")

        self.chunk_size = 512
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.rate = 48000
        self.threshold = 0
        self.short_normalize = (1.0 / 32768.0)
        self.swidth = 2
        self.timeout_length = 2
        self.audio_streams = {}
        self.buffer_lock = threading.Lock()
        self.max_buffer_size = 24
        self.prebuffer_chunks = 4
        self.fade_in_chunks = 3
        self.fade_out_chunks = 4
        self.silence_hold_chunks = 2

        # initialise microphone recording
        self.p = pyaudio.PyAudio()
        self.playing_stream = self.p.open(format=self.audio_format, channels=self.channels, rate=self.rate, output=True, frames_per_buffer=self.chunk_size, stream_callback=self.audio_callback)
        self.recording_stream = self.p.open(format=self.audio_format, channels=self.channels, rate=self.rate, input=True, frames_per_buffer=self.chunk_size)

        # Termination handler
        def handler(signum, frame):
            print("\033[2KTerminating...")
            message = Protocol(dataType=DataType.Terminate, room=self.room, data=self.name.encode(encoding='UTF-8'))
            self.s.sendto(message.out(), self.server)
            if platform.system() == "Windows":
                os.kill(os.getpid(), signal.SIGBREAK)
            else:
                os.kill(os.getpid(), signal.SIGKILL)

        if platform.system() == "Windows":
            signal.signal(signal.SIGBREAK, handler)
        else:
            signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)

        # start threads
        self.s.settimeout(0.5)
        receive_thread = threading.Thread(target=self.receive_server_data).start()
        self.send_data_to_server()

    def _get_stream_state(self, user_id):
        state = self.audio_streams.get(user_id)
        if state is None:
            state = {
                "buffer": deque(maxlen=self.max_buffer_size),
                "started": False,
                "fade_in_remaining": self.fade_in_chunks,
                "fade_out_remaining": 0,
                "last_chunk": np.zeros(self.chunk_size, dtype=np.float32),
                "silence_chunks": 0,
                "last_packet_time": time.monotonic(),
            }
            self.audio_streams[user_id] = state
        return state

    def audio_callback(self, in_data, frame_count, time_info, status):
        mixed_audio = np.zeros(frame_count, dtype=np.float32)
        users_talking = []
        current_time = time.monotonic()
        with self.buffer_lock:
            users_to_remove = []
            for user_id, state in list(self.audio_streams.items()):
                buffer = state["buffer"]
                if not state["started"] and len(buffer) >= self.prebuffer_chunks:
                    state["started"] = True
                    state["fade_in_remaining"] = self.fade_in_chunks
                    state["fade_out_remaining"] = 0
                    state["silence_chunks"] = 0
                chunk_array = None
                if state["started"] and len(buffer) > 0:
                    chunk_bytes = buffer.popleft()
                    audio_array = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    state["last_chunk"] = audio_array.copy()
                    state["silence_chunks"] = 0
                    chunk_array = audio_array
                else:
                    if state["started"]:
                        state["silence_chunks"] += 1
                        if state["silence_chunks"] <= self.silence_hold_chunks:
                            chunk_array = state["last_chunk"]
                        else:
                            if state["fade_out_remaining"] == 0 and state["last_chunk"] is not None:
                                state["fade_out_remaining"] = self.fade_out_chunks
                            if state["fade_out_remaining"] > 0 and state["last_chunk"] is not None:
                                factor = state["fade_out_remaining"] / self.fade_out_chunks
                                chunk_array = state["last_chunk"] * factor
                                state["fade_out_remaining"] -= 1
                                if state["fade_out_remaining"] == 0 and len(buffer) == 0:
                                    state["started"] = False
                                    state["fade_in_remaining"] = self.fade_in_chunks
                                    state["silence_chunks"] = 0
                                    state["last_chunk"] = np.zeros(self.chunk_size, dtype=np.float32)
                            else:
                                chunk_array = None
                    else:
                        chunk_array = None

                if chunk_array is None:
                    if not state["started"] and len(buffer) == 0 and current_time - state["last_packet_time"] > 5:
                        users_to_remove.append(user_id)
                    continue

                if state["fade_in_remaining"] > 0:
                    factor = (self.fade_in_chunks - state["fade_in_remaining"] + 1) / self.fade_in_chunks
                    chunk_array = chunk_array * factor
                    state["fade_in_remaining"] -= 1

                mixed_audio[:chunk_array.shape[0]] += chunk_array
                users_talking.append(user_id)

            for user_id in users_to_remove:
                if user_id in self.audio_streams and not self.audio_streams[user_id]["started"] and len(self.audio_streams[user_id]["buffer"]) == 0:
                    del self.audio_streams[user_id]

        if users_talking:
            print("Users talking: %s (room %s)         " % (', '.join(users_talking), self.room), end='\r')
            mixed_audio /= max(1, len(users_talking))

        mixed_audio = np.clip(mixed_audio, -1.0, 1.0)
        mixed_output = (mixed_audio * 32767).astype(np.int16)
        return (mixed_output.tobytes(), pyaudio.paContinue)

    def receive_server_data(self):
        while self.connected:
            try:
                data, addr = self.s.recvfrom(1026)
                message = Protocol(datapacket=data)
                if message.DataType == DataType.ClientData:
                    user_id = str(message.head)
                    with self.buffer_lock:
                        state = self._get_stream_state(user_id)
                        state["buffer"].append(message.data)
                        state["last_packet_time"] = time.monotonic()
                        if not state["started"] and len(state["buffer"]) >= self.prebuffer_chunks:
                            state["started"] = True
                            state["fade_in_remaining"] = self.fade_in_chunks
                            state["fade_out_remaining"] = 0
                            state["silence_chunks"] = 0

                elif message.DataType == DataType.Handshake or message.DataType == DataType.Terminate:
                    print(message.data.decode("utf-8"))
            except socket.timeout:
                print("\033[2K", end="\r")  # clearing line
                time.sleep(0.01)
            except Exception as err:
                pass

    def connect_to_server(self):
        if self.connected:
            return True

        message = Protocol(dataType=DataType.Handshake, room=self.room, data=self.name.encode(encoding='UTF-8'))
        self.s.sendto(message.out(), self.server)

        data, addr = self.s.recvfrom(1026)
        datapack = Protocol(datapacket=data)

        if addr == self.server and datapack.DataType == DataType.Handshake:
            print('Connected to server to room %s successfully!' % datapack.room)
            print(datapack.data.decode("utf-8"))
            self.connected = True
        return self.connected

    def rms(self, frame):
        count = len(frame) / self.swidth
        format = "%dh" % count
        shorts = struct.unpack(format, frame)

        sum_squares = 0.0
        for sample in shorts:
            n = sample * self.short_normalize
            sum_squares += n * n
        rms = math.pow(sum_squares / count, 0.5)

        return rms * 1000

    def record(self):
        current = time.time()
        end = time.time() + self.timeout_length

        while current <= end:
            data = self.recording_stream.read(self.chunk_size)
            if self.rms(data) >= self.threshold:
                end = time.time() + self.timeout_length
            try:
                message = Protocol(dataType=DataType.ClientData, room=self.room, data=data)
                self.s.sendto(message.out(), self.server)
            except:
                pass
            current = time.time()

    def listen(self):
        while True:
            try:
                inp = self.recording_stream.read(self.chunk_size)
                rms_val = self.rms(inp)
                if rms_val > self.threshold:
                    self.record()
                else:
                    time.sleep(0.01)
            except:
                pass

    def send_data_to_server(self):
        while self.connected:
            self.listen()


client = Client()