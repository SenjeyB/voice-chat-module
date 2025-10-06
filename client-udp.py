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
from collections import defaultdict, deque

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
        self.audio_buffers = defaultdict(deque)
        self.buffer_lock = threading.Lock()
        self.max_buffer_size = 10
        self.min_buffer_size = 3
        
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

    def audio_callback(self, in_data, frame_count, time_info, status):
        mixed_audio = np.zeros(frame_count, dtype=np.int16)
        users_talking = []
        
        chunks_to_process = []
        with self.buffer_lock:
            for user_id, buffer in list(self.audio_buffers.items()):
                if len(buffer) >= self.min_buffer_size:
                    chunk = buffer.popleft()
                    chunks_to_process.append((user_id, chunk))
                    
                if len(buffer) == 0:
                    del self.audio_buffers[user_id]
        
        for user_id, chunk in chunks_to_process:
            audio_array = np.frombuffer(chunk, dtype=np.int16)
            
            if len(audio_array) != frame_count:
                if len(audio_array) < frame_count:
                    padding = np.zeros(frame_count - len(audio_array), dtype=np.int16)
                    audio_array = np.concatenate([audio_array, padding])
                else:
                    audio_array = audio_array[:frame_count]
            
            fade_length = min(64, len(audio_array) // 4)
            fade_in = np.linspace(0, 1, fade_length)
            fade_out = np.linspace(1, 0, fade_length)
            audio_array[:fade_length] = (audio_array[:fade_length] * fade_in).astype(np.int16)
            audio_array[-fade_length:] = (audio_array[-fade_length:] * fade_out).astype(np.int16)
            
            mixed_audio = mixed_audio.astype(np.int32) + audio_array.astype(np.int32)
            users_talking.append(user_id)

        mixed_audio = np.clip(mixed_audio, -32768, 32767).astype(np.int16)
        if users_talking:
            print("Users talking: %s (room %s)         " % (', '.join(users_talking), self.room), end='\r')
        
        return (mixed_audio.tobytes(), pyaudio.paContinue)

    def receive_server_data(self):
        while self.connected:
            try:
                data, addr = self.s.recvfrom(1026)
                message = Protocol(datapacket=data)
                if message.DataType == DataType.ClientData:
                    user_id = str(message.head)
                    with self.buffer_lock:
                        if len(self.audio_buffers[user_id]) < self.max_buffer_size:
                            self.audio_buffers[user_id].append(message.data)
                        elif len(self.audio_buffers[user_id]) >= self.max_buffer_size:
                            self.audio_buffers[user_id].clear()

                elif message.DataType == DataType.Handshake or message.DataType == DataType.Terminate:
                    print(message.data.decode("utf-8"))
                    with self.buffer_lock:
                        for user_buffer in self.audio_buffers.values():
                            user_buffer.clear()
                        self.audio_buffers.clear()
            except socket.timeout:
                print("\033[2K", end="\r")
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