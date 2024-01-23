import matplotlib
import os
import numpy as np
import scipy
import queue

from matplotlib.figure import Figure
from PyQt5 import uic, QtWidgets, QtCore
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.widgets import SpanSelector

matplotlib.use('Qt5Agg')
FORM_CLASS, _ = uic.loadUiType(os.path.join(os.path.dirname(__file__), 'interface2.ui'))


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

    @staticmethod
    def minimumSizeHint():
        return QtCore.QSize(1000, 300)


def closest_match(freqs, reference_freqs):
    # Znajdź częstotliwość z 'freqs', która ma najbliższe dopasowanie do jednej z 'reference_freqs'
    closest_freq = min(
        reference_freqs,
        key=lambda ref_freq: min(abs(freq - ref_freq) for freq in freqs),
    )
    return closest_freq


class AppWidgetWithUI(QtWidgets.QWidget, FORM_CLASS):
    def __init__(self, parent=None):
        super(AppWidgetWithUI, self).__init__(parent)
        self.span = None
        self.thread = None
        self.stream = None
        self.streaming = False
        self.setupUi(self)
        self._nazwa_pliku = ''

        self.ax = MplCanvas(self, width=5, height=4, dpi=100)
        self.full_plot_layout.addWidget(self.ax)
        self.sc = MplCanvas(self, width=5, height=4, dpi=100)
        self.cut_plot_layout.addWidget(self.sc)

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
        self.ax.axes.cla()
        # Wczytywanie pliku WAV
        sample_rate, data = scipy.io.wavfile.read(self._nazwa_pliku)
        if len(data.shape) == 2:
            data = data[:, 0]

        # Okno Hanninga i FFT
        window = np.hanning(len(data))
        fft_result = scipy.fft.fft(data * window)
        magnitude_spectrum = np.abs(fft_result)

        # Skalowanie częstotliwości
        frequencies = np.linspace(0, sample_rate, len(magnitude_spectrum))
        half_length = len(frequencies) // 2
        frequencies = frequencies[:half_length]
        magnitude_spectrum = magnitude_spectrum[:half_length]

        def on_select(x_min, x_max):
            # Convert the selected time range to sample indices
            start_sample = int(x_min * sample_rate)
            end_sample = int(x_max * sample_rate)

            # Ensure end_sample does not exceed the length of the data
            end_sample = min(end_sample, len(data))

            # Select the data within the specified time range
            selected_data = data[start_sample:end_sample]
            print(selected_data)

            # Now you can further process the selected_data as needed
            self.analyze_data(selected_data, sample_rate)

        length = data.shape[0] / sample_rate
        time = np.linspace(0., length, data.shape[0])
        self.ax.axes.plot(time, data)
        self.ax.draw()

        self.span = SpanSelector(
            self.ax.axes,
            on_select,
            "horizontal",
            useblit=True,
            props=dict(alpha=0.5, facecolor="tab:blue"),
            interactive=True,
            drag_from_anywhere=True
        )

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
        threshold = 0.75 * np.max(magnitude_spectrum)  # Progowa wartość magnitudy
        significant_indices = np.where(magnitude_spectrum > threshold)[0]
        significant_frequencies = frequencies[significant_indices]

        # Wyświetlanie wyników
        self.sc.axes.cla()
        self.sc.axes.plot(frequencies, magnitude_spectrum, label='Magnitude Spectrum')
        self.sc.draw()

        peak_indices = np.argsort(significant_frequencies)[-1:]
        peak_frequencies = significant_frequencies[0], significant_frequencies[peak_indices]

        # Znajdź najbliższe dopasowania w tabeli DTMF
        row_freq = closest_match(peak_frequencies, self.dtmf_tones.keys())
        col_freq = closest_match(peak_frequencies, self.dtmf_tones[row_freq].keys())

        # Zwróć odpowiadający przycisk
        detected_button = self.dtmf_tones[row_freq][col_freq]

        self.display(detected_button)


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    widget = AppWidgetWithUI()
    widget.show()
    app.exec_()
