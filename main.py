import matplotlib
import os
import numpy as np
import scipy
from matplotlib import pyplot as plt
from PyQt5 import uic, QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

matplotlib.use('Qt5Agg')
FORM_CLASS, _ = uic.loadUiType(os.path.join(os.path.dirname(__file__), 'interface.ui'))


def lowpassFilter(data, fs, cutoff=1000, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = scipy.signal.filtfilt(b, a, data)
    return y


def highpassFilter(data, fs, cutoff=300, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.butter(order, normal_cutoff, btype='high', analog=False)
    y = scipy.signal.filtfilt(b, a, data)
    return y


class Canvas(FigureCanvas):
    def __init__(self, parent=None):
        super().__init__()

    @staticmethod
    def minimumSizeHint():
        return QtCore.QSize(800, 300)


class AppWidgetWithUI(QtWidgets.QWidget, FORM_CLASS):
    def __init__(self, parent=None):
        super(AppWidgetWithUI, self).__init__(parent)
        self.thread = None
        self.stream = None
        self.streaming = False
        self.setupUi(self)
        self._nazwa_pliku = ''

        self.sc = Canvas(plt.Figure())
        self.plt_layout.addWidget(self.sc)

        self.btn_load.clicked.connect(self.load_clicked)
        self.btn_recognize.clicked.connect(self.rec_clicked)

        self.box_window.addItems(['hann', 'hamming', 'blackman', 'bartlett'])
        self.box_filter.addItems(['None', 'Lowpass', 'Highpass'])

    def load_clicked(self):
        self._nazwa_pliku = 'test'
        self._nazwa_pliku = self.open_file_dialog()
        self.le_filePath.setText(self._nazwa_pliku)

    def open_file_dialog(self):
        options = QtWidgets.QFileDialog.Options()
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Wybierz plik", "",
                                                             "Wszystkie Pliki (*);;Pliki tekstowe (*.txt)",
                                                             options=options)
        if file_name:
            return file_name

    def display(self, dominant_note, significant_notes):
        self.txt_note.setPlainText(f"Dominant Note: {dominant_note}")
        self.txt_harmonic.setPlainText(f"Other Notes: {significant_notes}")

    def rec_clicked(self):
        # Wczytywanie pliku WAV
        sample_rate, data = scipy.io.wavfile.read(self._nazwa_pliku)
        if len(data.shape) == 2:
            data = data[:, 0]

        samples_per_segment = int(self.text_Nperseg.text())
        overlap_samples = int(self.text_Noverlap.text())
        fft_points = int(self.text_Nfft.text())
        window = str(self.box_window.currentText())

        filterType = str(self.box_filter.currentText())

        if filterType == 'Lowpass':
            data = lowpassFilter(data, sample_rate)
        elif filterType == 'Highpass':
            data = highpassFilter(data, sample_rate)

        f, t, Sxx = scipy.signal.spectrogram(data, fs=sample_rate, nperseg=samples_per_segment, noverlap=overlap_samples, nfft=fft_points,
                                             window=window)

        self.sc.figure.clear()
        ax = self.sc.figure.add_subplot(111)
        ax.pcolormesh(t, f, 10 * np.log10(Sxx))
        ax.set_ylabel('Frequency [Hz]')
        ax.set_xlabel('Time [sec]')
        self.sc.draw()


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    widget = AppWidgetWithUI()
    widget.show()
    app.exec_()
