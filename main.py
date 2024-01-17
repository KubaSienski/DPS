import matplotlib
import os
import numpy as np
import scipy
import queue
from matplotlib.figure import Figure
from PyQt5 import uic, QtWidgets, QtCore
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

matplotlib.use('Qt5Agg')
FORM_CLASS, _ = uic.loadUiType(os.path.join(os.path.dirname(__file__), 'interface2.ui'))


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

    @staticmethod
    def minimumSizeHint():
        return QtCore.QSize(800, 300)


def frequency_to_note_symbol(frequency):
    if frequency == 0:
        return "N/A"  # Return a placeholder value if the frequency is zero

    # Calculate the note number with the correct offset (69 for A4 = 440 Hz)
    note_number = round(12 * np.log2(frequency / 440) + 69)

    # Calculate the octave
    octave = note_number // 12

    # Calculate the note name
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    note_name = note_names[note_number % 12]

    # Return the note symbol
    return note_name + str(octave - 1)  # Subtract 1 to align octave with standard notation


def closest_match(freqs, reference_freqs):
    # Znajdź częstotliwość z 'freqs', która ma najbliższe dopasowanie do jednej z 'reference_freqs'
    closest_freq = min(
        reference_freqs,
        key=lambda ref_freq: min(abs(freq - ref_freq) for freq in freqs),
    )
    return closest_freq


def split_audio(audio_data, sample_rate, segment_length=0.1):
    # Długość segmentu w próbkach
    segment_length_samples = int(segment_length * sample_rate)
    for start in range(0, len(audio_data), segment_length_samples):
        yield audio_data[start:start + segment_length_samples]


class AppWidgetWithUI(QtWidgets.QWidget, FORM_CLASS):
    def __init__(self, parent=None):
        super(AppWidgetWithUI, self).__init__(parent)
        self.thread = None
        self.stream = None
        self.streaming = False
        self.setupUi(self)
        self._nazwa_pliku = ''

        self.sc = MplCanvas(self, width=5, height=4, dpi=100)
        self.img_layout.addWidget(self.sc)

        self.audio_queue = queue.Queue()
        self.max_tones = 5
        self.btn_load.clicked.connect(self.load_clicked)
        self.btn_recognize.clicked.connect(self.rec_clicked)

        self.dtmf_tones = {
            697: {1209: "1", 1336: "2", 1477: "3"},
            770: {1209: "4", 1336: "5", 1477: "6"},
            852: {1209: "7", 1336: "8", 1477: "9"},
            941: {1209: "*", 1336: "0", 1477: "#"},
        }

    def load_clicked(self):
        self._nazwa_pliku = 'test'
        self._nazwa_pliku = self.open_file_dialog()
        self.le_filePath.setText(self._nazwa_pliku)

    def display(self, detected_button):
        self.txt_harmonic.setPlainText(f"Pushed Keys: {detected_button}")

    def rec_clicked(self):
        self.sc.axes.cla()
        # Wczytywanie pliku WAV
        sample_rate, data = scipy.io.wavfile.read(self._nazwa_pliku)
        if len(data.shape) == 2:
            data = data[:, 0]

        detected_buttons = self.analyze_full_recording(data, sample_rate)

        detected_buttons_clear = [detected_buttons[0]]
        for i, element in enumerate(detected_buttons):
            if i > 0 and detected_buttons[i - 1] != element:
                detected_buttons_clear.append(element)

        self.display(detected_buttons_clear)

    def open_file_dialog(self):
        options = QtWidgets.QFileDialog.Options()
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Wybierz plik", "",
                                                             "Wszystkie Pliki (*);;Pliki tekstowe (*.txt)",
                                                             options=options)
        if file_name:
            return file_name

    def analyze_full_recording(self, audio_data, sample_rate):
        detected_keys = []
        for segment in split_audio(audio_data, sample_rate):
            detected_key = self.analyze_data(segment, sample_rate)
            detected_keys.append(detected_key)
        return detected_keys

    def analyze_data(self, data, sample_rate):
        # Okno Hanninga i FFT
        window = np.hanning(len(data))
        fft_result = scipy.fft.fft(data * window)
        magnitude_spectrum = np.abs(fft_result)

        # Skalowanie częstotliwości
        frequencies = np.linspace(0, sample_rate, len(magnitude_spectrum))
        half_length = len(frequencies) // 2
        frequencies = frequencies[:half_length]
        magnitude_spectrum = magnitude_spectrum[:half_length]

        # Wykrywanie tonów
        threshold = 0.5 * np.max(magnitude_spectrum)  # Progowa wartość magnitudy
        significant_indices = np.where(magnitude_spectrum > threshold)[0]
        significant_frequencies = frequencies[significant_indices]

        # Wyświetlanie wyników
        self.sc.axes.plot(frequencies, magnitude_spectrum, label='Magnitude Spectrum')
        self.sc.draw()

        peak_indices = np.argsort(significant_frequencies)[-1:]
        peak_frequencies = significant_frequencies[0], significant_frequencies[peak_indices]

        # Znajdź najbliższe dopasowania w tabeli DTMF
        row_freq = closest_match(peak_frequencies, self.dtmf_tones.keys())
        col_freq = closest_match(peak_frequencies, self.dtmf_tones[row_freq].keys())

        # Zwróć odpowiadający przycisk
        detected_button = self.dtmf_tones[row_freq][col_freq]

        return detected_button


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    widget = AppWidgetWithUI()
    widget.show()
    app.exec_()
