import os

import pyaudio
import numpy as np
import threading
import time

import tkinter as tk
import matplotlib.pyplot as plt
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib as mpl
mpl.rcParams['font.family'] = 'SimHei'

from scipy import signal
import scipy.io.wavfile as wavfile


class AudioFilterGUI:
	def __init__(self):
		self.RATE = 40000  # 音频采样率
		self.CHUNK = 1024  # 每次读取音频数据的大小
		self.DEGREE = 4
		self.amplitude_size = 1

		self.stream_in = None
		self.p = None
		self.sos = None
		self.frequency_domain = False
		self.filter_type = 4

		self.audio_data = None
		self.file_path = None
		self.frames = []

		self.stop_event = threading.Event()
		self.stop_event.set()

		self.animation_ylims = [[(-1e5, 1e5), (-1e5, 1e5), (-1e5, 1e5), (-1e5, 1e5), (-1e5, 1e5)],
								('低通滤波', '高通滤波', '带通滤波', '带阻滤波',"无滤波")]
		self.show_plot_ylims = [[(-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1)],
								('低通滤波', '高通滤波', '带通滤波', '带阻滤波',"无滤波")]

		self.mode = [['lowpass', 'highpass', 'bandpass', 'notch', None], [2000, 500, (500, 2000), (1000, 4000), None]]

	def get_audio(self):
		if self.stream_in:
			self.stream_in.stop_stream()
			self.stream_in.close()
		# 创建PyAudio对象
		self.p = pyaudio.PyAudio()
		print(self.p.get_default_input_device_info())
		# 打开音频输入流
		self.stream_in = self.p.open(format=pyaudio.paFloat32, channels=1, rate=self.RATE, input=True,
									 frames_per_buffer=self.CHUNK)
		self.stream_out = self.p.open(format=pyaudio.paInt16, channels=1, rate=self.RATE, input=True,
									  frames_per_buffer=self.CHUNK)

	def opening_threads(self, types, canvas):
		self.filter_type = types
		self.label_text.set(f"{self.mode[0][types]} 截止频率 {self.mode[1][types]}")
		self.create_filter()
		if self.stop_event.is_set() and self.file_path == None:
			self.stop_event.clear()
			# print(self.stream_in._frames_per_buffer, self.stream_in._rate)
			t = threading.Thread(target=self.show_plot,
								 args=(canvas, types))
			t.start()
		elif self.stop_event.is_set() and self.stream_in == None:
			self.stop_event.clear()
			x = threading.Thread(target=self.animation,
								 args=(canvas, types))
			x.start()
		else:
			self.stop_event.set()

	def animation(self, canvas, flag):
		fs, data = wavfile.read(self.file_path)
		rate = fs
		head, last, lens = 0, 0, len(data)
		head_time = 0
		if data.ndim == 2:
			data = data[:,0]
		while not self.stop_event.is_set():
			if self.file_path == None:
				return
			if head + self.CHUNK < lens:
				last = head + self.CHUNK
			elif last < lens:
				last = lens
			else:
				print("播放完成")
				self.label_text.set("播放完成")
				return

			if self.RATE != rate:
				resample_ratio = self.RATE / rate
				rate = self.RATE
				draw_data = signal.resample(data[head:last], int(len(data[head:last]) * resample_ratio))
			else:
				draw_data = data[head:last].astype(np.float32)

			head = last
			t = self.CHUNK / self.RATE
			time.sleep(2*t)

			if self.sos is not None:
				filtered_samples = signal.sosfilt(self.sos, draw_data) * self.amplitude_size
			else:
				filtered_samples = draw_data * self.amplitude_size

			if self.frequency_domain:
				fft_data = np.fft.fft(filtered_samples)
				freq = np.fft.fftfreq(len(filtered_samples), d=1.0 / self.RATE)

				fft_data2 = np.fft.fft(draw_data)
				freq2 = np.fft.fftfreq(len(draw_data), d=1.0 / self.RATE)

				normalized_data = np.abs(fft_data) / np.max(np.abs(fft_data))
				normalized_data2 = np.abs(fft_data2) / np.max(np.abs(fft_data2))

				plt.clf()
				plt.plot(freq[:len(freq) // 2], normalized_data[:len(freq) // 2], label='滤波波形')
				plt.plot(freq2[:len(freq2) // 2], normalized_data2[:len(freq2) // 2], label='原始波形',alpha=0.7)
				# plt.plot(freq[:len(freq) // 2], np.abs(fft_data)[:len(freq) // 2], label='Filtered Data')
				# plt.plot(freq2[:len(freq2) // 2], np.abs(fft_data2)[:len(freq2) // 2], label='original Data')
				plt.xlabel('频率 (Hz)')
				plt.ylabel('幅度')
				plt.title(self.animation_ylims[1][flag])
				plt.xlim(0,self.RATE*0.5*0.6)
				# plt.ylim(0,1)
				plt.margins(0.05,0.05)
				plt.legend()

			else:
				plt.clf()
				time_size = np.linspace(head_time, head_time + filtered_samples.shape[0] / self.RATE,
										filtered_samples.shape[0])
				time_size2 = np.linspace(head_time, head_time + draw_data.shape[0] / self.RATE,
										 draw_data.shape[0])

				head_time = head_time + filtered_samples.shape[0] / self.RATE

				plt.plot(time_size, filtered_samples, label='滤波波形')
				plt.plot(time_size2, draw_data, label='原始波形',alpha=0.7)

				plt.xlabel('时间 (s)')
				plt.ylabel('幅度')

				x = [-100000, -75000, -50000, -25000, 0, 25000, 50000, 75000, 100000]
				# self.animation_ylims = [(-1e2,1e2),(-1e5,1e5),(-5e2,5e2),(-1e5,1e5),(-1e5,1e5)]
				plt.ylim(self.animation_ylims[0][flag][0], self.animation_ylims[0][flag][1])
				plt.yticks(x, [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
				plt.title(self.animation_ylims[1][flag])
				plt.legend()
			canvas.draw()

			# 保存音频文件
			if self.save_flag:
				self.save_flag = False
				self.audio_data = np.frombuffer(b''.join(data), dtype=np.int16)
				print(self.audio_data)
				if self.sos is not None:
					self.audio_data = signal.sosfilt(self.sos, self.audio_data) * self.amplitude_size
				else:
					self.audio_data = np.copy(self.audio_data) * self.amplitude_size
				self.save_audio()

	def show_plot(self, canvas, flag):
		try:

			frames = []
			head_time = 0
			while not self.stop_event.is_set():
				# 从音频输入流中读取数据
				if self.stream_in == None:
					return
				data = self.stream_in.read(self.CHUNK, exception_on_overflow=False)

				frames.append(self.stream_out.read(self.CHUNK, exception_on_overflow=False))
				# 将二进制数据转换为numpy数组
				samples = np.frombuffer(data, dtype=np.float32)
				# 对音频数据进行滤波
				if self.sos is not None:
					filtered_samples = signal.sosfiltfilt(self.sos, samples) * self.amplitude_size
				else:
					filtered_samples = samples * self.amplitude_size

				if self.frequency_domain:
					fft_data = np.fft.fft(filtered_samples)
					freq = np.fft.fftfreq(len(filtered_samples), d=1.0 / self.RATE)

					fft_data1 = np.fft.fft(samples)
					freq1 = np.fft.fftfreq(len(samples), d=1.0 / self.RATE)

					normalized_data = np.abs(fft_data) / np.max(np.abs(fft_data))
					normalized_data2 = np.abs(fft_data1) / np.max(np.abs(fft_data1))

					plt.clf()
					plt.plot(freq[:len(freq) // 2], normalized_data[:len(freq) // 2], label='Filtered Data')
					plt.plot(freq1[:len(freq1) // 2], normalized_data2[:len(freq1) // 2], label='original Data',alpha=0.7)
					# plt.plot(freq[:len(freq) // 2], np.abs(fft_data)[:len(freq) // 2], label="Filtered Data")
					# plt.plot(freq[:len(freq1) // 2], np.abs(fft_data1)[:len(freq1) // 2], label="original Data")
					plt.xlabel('Frequency (Hz)')
					plt.ylabel('Amplitude')
					plt.title(self.show_plot_ylims[1][flag])
					plt.legend()
				else:
					plt.clf()
					time_size = np.linspace(head_time, head_time + filtered_samples.shape[0] / self.RATE,
											filtered_samples.shape[0])
					time_size2 = np.linspace(head_time, head_time + samples.shape[0] / self.RATE,
											 samples.shape[0])
					head_time = head_time + filtered_samples.shape[0] / self.RATE
					plt.plot(time_size, filtered_samples, label='Filtered Data')
					plt.plot(time_size2, samples, label='original Data',alpha=0.7)
					plt.xlabel('Time (s)')
					plt.ylabel('Amplitude (dB)')

					plt.ylim(self.show_plot_ylims[0][flag][0], self.show_plot_ylims[0][flag][1])
					plt.title(self.show_plot_ylims[1][flag])
					plt.legend()
				canvas.draw()

				# 保存音频文件
				if self.save_flag:
					self.save_flag = False
					self.audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
					print(self.audio_data)
					if self.sos is not None:
						self.audio_data = signal.sosfilt(self.sos, self.audio_data) * self.amplitude_size
					else:
						self.audio_data = np.copy(self.audio_data) * self.amplitude_size
					self.save_audio()

		except KeyboardInterrupt:
			self.stream_in.stop_stream()
			self.stream_in.close()
			self.p.terminate()

	def create_filter(self):
		cutoff_freq = self.mode[1][self.filter_type]
		if self.filter_type == 0:
			if self.DEGREE % 2 != 0:
				self.DEGREE += 1
			self.sos = signal.butter(self.DEGREE, 2 * cutoff_freq / self.RATE, 'low', output='sos')
		elif self.filter_type == 1:
			if self.DEGREE % 2 != 0:
				self.DEGREE += 1
			self.sos = signal.butter(self.DEGREE, 2 * cutoff_freq / self.RATE, 'high', output='sos')
		elif self.filter_type == 2:
			self.sos = signal.butter(self.DEGREE, [2 * cutoff_freq[0] / self.RATE, 2 * cutoff_freq[1] / self.RATE],
									 'bandpass', output='sos')
		elif self.filter_type == 3:
			self.sos = signal.butter(self.DEGREE, [2 * cutoff_freq[0] / self.RATE, 2 * cutoff_freq[1] / self.RATE],
									 'bandstop', output='sos')
		else:
			self.sos = None

	def save_audio(self):
		WAVE_OUTPUT_FILENAME = r".\audio.wav"

		self.audio_data = np.int16(self.audio_data)
		# 创建wave文件对象
		wavfile.write(WAVE_OUTPUT_FILENAME, self.RATE, self.audio_data)
		self.label_text.set("已保存滤波后的音频文件")
		print("已保存滤波后的音频文件")

	def change_rate(self, rate):
		self.RATE = int(rate)
		print("RATE = ", self.RATE)
		self.create_filter()
		if self.stop_event.is_set():
			self.stop_event.clear()

	def change_chunk(self, chunk):
		self.CHUNK = int(chunk)
		print("CHUNK = ", self.CHUNK)
		self.create_filter()
		if self.stop_event.is_set():
			self.stop_event.clear()

	def change_degree(self, degree):
		if self.stop_event.is_set():
			self.stop_event.clear()
		self.DEGREE = int(degree)
		self.create_filter()
		print("DEGREE = ", self.DEGREE)

	def change_amplitude(self, value):
		if self.stop_event.is_set():
			self.stop_event.clear()
		self.amplitude_size = float(value)
		print("amplitude_size  = ", self.amplitude_size)

	def select_file(self):
		if self.stop_event.is_set():
			self.stop_event.clear()
		self.p, self.stream_in = None, None
		self.file_path = filedialog.askopenfilename(filetypes=[("Audio files", "*.mp3;*.wav")])

	def recording_audio(self):
		if self.stop_event.is_set():
			self.stop_event.clear()
		self.file_path = None
		self.get_audio()

	def frequency_domain_on(self):
		self.frequency_domain = True

	def frequency_domain_off(self):
		self.frequency_domain = False

	def run(self):
		# 创建 Tkinter 窗口
		window = tk.Tk()
		window.title("选择音频处理方式")

		# 创建标签框架
		label_frame = tk.Frame(window)
		label_frame.pack(pady=10)

		# 处理方式标签
		label = tk.Label(label_frame, text="选择音频处理方式", font=("Arial", 16))
		label.pack()

		# 创建 Figure 对象
		fig = plt.figure(figsize=(8, 4))
		canvas = FigureCanvasTkAgg(fig, master=window)
		canvas.get_tk_widget().pack(padx=20)

		# 创建滑动条框架
		scale_frame = tk.Frame(window)
		scale_frame.pack(pady=5)

		# 创建滑动条1
		s1_label = tk.Label(scale_frame, text="音频采样率", font=("Arial", 12))
		s1_label.grid(row=0, column=0, padx=10, pady=5)
		s1 = tk.Scale(scale_frame, from_=20, to=40000, orient=tk.HORIZONTAL, length=300, showvalue=1,
					  resolution=1, command=lambda value: self.change_rate(value))
		s1.set(self.RATE)
		s1.grid(row=0, column=1, padx=10, pady=5)

		# 创建滑动条2
		s2_label = tk.Label(scale_frame, text="单次取得采样点个数", font=("Arial", 12))
		s2_label.grid(row=1, column=0, padx=10, pady=5)
		s2 = tk.Scale(scale_frame, from_=1, to=15000, orient=tk.HORIZONTAL, length=300, showvalue=1,
					  resolution=1, command=lambda value: self.change_chunk(value))
		s2.set(self.CHUNK)
		s2.grid(row=1, column=1, padx=10, pady=5)

		# 创建滑动条3
		s3_label = tk.Label(scale_frame, text="滤波器阶数", font=("Arial", 12))
		s3_label.grid(row=2, column=0, padx=10, pady=5)
		s3 = tk.Scale(scale_frame, from_=1, to=8, orient=tk.HORIZONTAL, length=300, showvalue=1,
					  resolution=1, command=lambda value: self.change_degree(value))
		s3.set(self.DEGREE)
		s3.grid(row=2, column=1, padx=10, pady=5)

		s4_label = tk.Label(scale_frame, text="幅度大小", font=("Arial", 12))
		s4_label.grid(row=3, column=0, padx=10, pady=5)
		s4 = tk.Scale(scale_frame, from_=0.1, to=2.0, orient=tk.HORIZONTAL, length=300, showvalue=1,
					  resolution=0.1, command=lambda value: self.change_amplitude(value))
		s4.set(self.amplitude_size)
		s4.grid(row=3, column=1, padx=10, pady=5)
		# 创建状态标签
		self.label_text = tk.StringVar()
		status_label = tk.Label(window, textvariable=self.label_text, font=("Arial", 12))
		status_label.pack()

		# 创建按钮框架
		button_frame = tk.Frame(window)
		button_frame.pack(pady=10)

		# 创建低通滤波按钮
		btn_lowpass = tk.Button(button_frame, text="低通滤波",
								command=lambda: self.opening_threads(0, canvas), width=15,
								height=2, font=("Arial", 12))
		btn_lowpass.grid(row=0, column=0, padx=10, pady=5)

		# 创建高通滤波按钮
		btn_highpass = tk.Button(button_frame, text="高通滤波",
								 command=lambda: self.opening_threads(1, canvas),
								 width=15, height=2, font=("Arial", 12))
		btn_highpass.grid(row=0, column=1, padx=10, pady=5)

		# 创建带通滤波按钮
		btn_bandpass = tk.Button(button_frame, text="带通滤波",
								 command=lambda: self.opening_threads(2, canvas),
								 width=15, height=2, font=("Arial", 12))
		btn_bandpass.grid(row=0, column=2, padx=10, pady=5)

		# 创建带阻滤波按钮
		btn_notch = tk.Button(button_frame, text="带阻滤波",
							  command=lambda: self.opening_threads(3, canvas), width=15,
							  height=2, font=("Arial", 12))
		btn_notch.grid(row=0, column=3, padx=10, pady=5)

		# 创建无处理按钮
		btn_none = tk.Button(button_frame, text="无处理",
							 command=lambda: self.opening_threads(4, canvas), width=15,
							 height=2, font=("Arial", 12))
		btn_none.grid(row=0, column=4, padx=10, pady=5)

		recording_button = tk.Button(button_frame, text="实时录音",
									 command=self.recording_audio, width=15,
									 height=2, font=("Arial", 12))
		recording_button.grid(row=1, column=0, padx=10, pady=5)

		select_button = tk.Button(button_frame, text="选择文件",
								  command=self.select_file, width=15,
								  height=2, font=("Arial", 12))
		select_button.grid(row=1, column=1, padx=10, pady=5)

		frequency_domain_on_button = tk.Button(button_frame, text="显示频域",
											   command=self.frequency_domain_on, width=15,
											   height=2, font=("Arial", 12))
		frequency_domain_on_button.grid(row=2, column=0, padx=10, pady=5)

		frequency_domain_off_button = tk.Button(button_frame, text="显示时域",
												command=self.frequency_domain_off, width=15,
												height=2, font=("Arial", 12))
		frequency_domain_off_button.grid(row=2, column=1, padx=10, pady=5)
		# 创建保存按钮
		self.save_flag = False
		btn_save = tk.Button(button_frame, text="保存音频文件",
							 command=lambda: setattr(self, "save_flag", True), width=15,
							 height=2, font=("Arial", 12))
		btn_save.grid(row=1, column=2, padx=10, pady=5)

		# 运行 Tkinter 主循环
		window.mainloop()
