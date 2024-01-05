import matplotlib
import os
import numpy as np
import scipy
import queue
from matplotlib.figure import Figure
from PyQt5 import uic, QtWidgets, QtCore
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

matplotlib.use('Qt5Agg')
FORM_CLASS, _ = uic.loadUiType(os.path.join(os.path.dirname(__file__), 'interface.ui'))


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

    def load_clicked(self):
        self._nazwa_pliku = 'test'
        self._nazwa_pliku = self.open_file_dialog()
        self.le_filePath.setText(self._nazwa_pliku)

    def display(self, dominant_note, significant_notes):
        self.txt_note.setPlainText(f"Dominant Note: {dominant_note}")
        self.txt_harmonic.setPlainText(f"Other Notes: {significant_notes}")

    def rec_clicked(self):
        # Wczytywanie pliku WAV
        sample_rate, data = scipy.io.wavfile.read(self._nazwa_pliku)
        if len(data.shape) == 2:
            data = data[:, 0]

        dominant_note, significant_notes = self.analyze_data(data, sample_rate)
        self.display(dominant_note, significant_notes)

    def open_file_dialog(self):
        options = QtWidgets.QFileDialog.Options()
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Wybierz plik", "",
                                                             "Wszystkie Pliki (*);;Pliki tekstowe (*.txt)",
                                                             options=options)
        if file_name:
            return file_name

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
        threshold = 0.1 * np.max(magnitude_spectrum)  # Progowa wartość magnitudy
        significant_indices = np.where(magnitude_spectrum > threshold)[0]
        significant_frequencies = frequencies[significant_indices]
        significant_notes = [frequency_to_note_symbol(f) for f in significant_frequencies]
        significant_notes = set(significant_notes)

        # Dominujący ton
        dominant_index = np.argmax(magnitude_spectrum)
        dominant_frequency = frequencies[dominant_index]
        dominant_note = frequency_to_note_symbol(dominant_frequency)

        # Wyświetlanie wyników
        self.sc.axes.cla()
        self.sc.axes.plot(frequencies, magnitude_spectrum, label='Magnitude Spectrum')
        self.sc.draw()
        return dominant_note, significant_notes


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    widget = AppWidgetWithUI()
    widget.show()
    app.exec_()
