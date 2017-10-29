from gui.average_streamline.widgets import AveStreamLineMainWindow
import sys
from PyQt5 import QtWidgets


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = AveStreamLineMainWindow()
    window.show()
    sys.exit(app.exec_())


