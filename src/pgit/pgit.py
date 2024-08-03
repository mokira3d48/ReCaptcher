import math
import time


class ProgressIter:
	""" Implementatation of a progressbar

	:param length: The length of the process.
	:param bins:   The number of bins will be displayed.
	:param barf:   The string format of the progress bar.
	:param bchr:   The basic character used to indicate
				   that the progression is passed.
	:param lchr:   The character used to open bins.
	:param rchr:   The character used to close bins.
	:param pchr:   The character used for progression indication.
	:param empt:   The character used to indication
				   that the progression not passed yet.

	:type length: int,
	:type bins: int,
	:type barf: optional str,
	:type bchr: str
	:type lchr: str
	:type rchr: str
	:type pchr: str
	:type empt: str
	"""

	def __init__(self,
				 length,
				 bins = 20,
				 barf = None,
				 bchr = '-',
				 lchr = '[',
				 rchr = ']',
				 pchr = '>',
				 empt = ' ',
				 time_format=None,):

		self._length = length
		self._bins = bins
		self._strf = barf if barf \
			else ("{logger}"
				  " {pbar}"
				  " {purcent:6.2f}% -"
				  " {progress}/{n_iters}"
				  " [{time_rem}"
				  " {iter_rate:.2f}its/sec] "
				   )
		self._bchr = bchr
		self._lchr = lchr
		self._rchr = rchr
		self._pchr = pchr
		self._empt = empt
		self._time_format = time_format

		self._progress = 0
		self._starttime = 0
		self._data = {}
		self._make_reset()

	@property
	def length(self):
		""":int: the length of the JOB """
		return self._length

	@length.setter
	def length(self, value):
		self._length = value

	def _format_time(self, millis, str_format = None):
		"""Function to converte milliseconds to hh:mm:ss:millis

		:type millis: int
		:type str_format: str
		:rtype: str
		"""
		if not str_format:
			str_format = "{days}:{hours:02d}:{mins:02d}:{secs:02d}:{millis:03d}"

		sec = millis // 1000
		millis = millis % 1000

		mins = sec // 60
		sec = sec % 60

		hours = mins // 60
		mins = mins % 60

		days = hours // 24
		hours = hours % 24
		# return "Time Lapsed = {0}:{1}:{2}".format(int(hours),int(mins),sec))
		return str_format.format(
			days=days,
			hours=hours,
			mins=mins,
			secs=sec,
			millis=millis,
		)

	def get_progress_purcent(self):
		"""Function to compute and returns the progress purcent

		:rtype: float
		"""
		return self._progress * 100.0 / self._length

	def _print_update(self):
		pstring = self._strf.format(**self._data)
		print("\033[2K", end='\r')
		print(pstring, end=' ', flush=True)

	def step(self, value):
		"""Function to perform one step

		:param value: The value of one step performed.
		:type value: int
		"""
		self._progress += value
		if self._progress > self._length:
			self._progress = self._length

		# compute remaning
		if not self._starttime:
			self._starttime = time.time()

		rate = self._progress / (time.time() - self._starttime)
		str_rem = ""
		if rate != 0.0:
			remaining = (self._length - self._progress) / rate
			# convert to milisecond
			remaining = int(remaining*1000)
			str_rem = self._format_time(remaining, self._time_format)
		else:
			str_rem = self._time_format.format(
				days="--",
				hours="--",
				mins="--",
				secs="--",
				millis="----",
			)

		n_bins = math.floor(self._progress  * self._bins / self._length)
		purcent = self.get_progress_purcent()
		done = (n_bins == self._bins)
		pchr = self._pchr if not done else self._bchr
		self._data['purcent'] = purcent
		self._data['time_rem'] = str_rem
		self._data['pbar'] = (
			f"{self._lchr}{self._bchr * n_bins}{pchr}"
			f"{self._empt*(self._bins - n_bins)}{self._rchr}"
		)
		self._data['progress'] = self._progress
		self._data['n_iters'] = self._length
		self._data['iter_rate'] = rate
		self._print_update()

	def log(self, message):
		"""Function of log printing

		:param message: The message will be logged.
		:type message: str
		"""
		self._data['logger'] = f"{message}"
		self._print_update()

	def finalise(self, message = None):
		""" Function to finalise the progress counting with a message

		:param message: The resume message.
		:type message: str, optional
		"""
		# clear the progress bar, and print the message received by argument
		print("\033[2K", end='\r')
		print((message if message else self._data['logger']), flush=True)

	def _make_reset(self):
		"""Function to reset the progress counter"""
		self._progress = 0
		self._starttime = 0
		self._data['logger'] = ''
		self._data['purcent'] = 0.0
		self._data['time_rem'] = ""
		self._data['pbar'] = " "*self._bins
		self._data['progress'] = 0
		self._data['n_iters'] = self._length
		self._data['iter_rate'] = 0

	def reset(self):
		self._make_reset()


def main():
	""" Main function """
	pbar_format = "{logger:18s} {pbar} [\033[91m{purcent:6.2f}\033[0m - {time_rem}]"
	time_format = "{mins:02d}:{secs:02d}"
	progressbar = ProgressIter(
		2000, 80,
		barf=pbar_format,
		pchr='#',
		bchr='=',
		empt='-',
		time_format=time_format,
	)
	for i in range(2000):
		progressbar.step(1)
		time.sleep(0.1)
		progressbar.log("First step: " + str(i))

	time.sleep(10)
	progressbar.finalise("The current job is done.")
	progressbar.reset()

	for i in range(2000):
		progressbar.step(1)
		time.sleep(0.1)
		progressbar.log("Second step: " + str(i))

	progressbar.finalise("The current job is done.")


if __name__ == '__main__':
	import os
	main()
	os.sys.exit(0)
