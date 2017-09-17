import unittest
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt
from gui.average_streamline.widgets import AveLineWidget, StageDataWidget
import sys

app = QApplication(sys.argv)


class AveLineWidgetTest(unittest.TestCase):
    def setUp(self):
        self.form = AveLineWidget()

    def test_increase_stage_number(self):
        self.form.stage_number.setValue(3)
        count = self.form.stackedWidget.count()
        self.assertEqual(3, count)
        self.form.stage_number.setValue(1)

    def test_decrease_stage_number(self):
        self.form.stage_number.setValue(3)
        self.assertEqual(3, self.form.stackedWidget.count())
        self.form.stage_number.setValue(2)
        self.assertEqual(2, self.form.stackedWidget.count())
        self.form.stage_number.setValue(1)

    def test_change_rho_auto_to_unchecked(self):
        self.form.stage_number.setValue(4)
        self.form.checkBox_rho_auto.setChecked(False)
        for i in range(4):
            stage_data_widget: StageDataWidget = self.form.stackedWidget.widget(i)
            self.assertFalse(stage_data_widget.label_rho.isHidden())
            self.assertFalse(stage_data_widget.rho.isHidden())
        self.form.stage_number.setValue(1)

    def test_change_rho_auto_to_checked(self):
        self.form.stage_number.setValue(4)
        self.form.checkBox_rho_auto.setChecked(False)
        self.form.checkBox_rho_auto.setChecked(True)
        for i in range(4):
            stage_data_widget: StageDataWidget = self.form.stackedWidget.widget(i)
            self.assertTrue(stage_data_widget.label_rho.isHidden())
            self.assertTrue(stage_data_widget.rho.isHidden())
        self.form.stage_number.setValue(1)

    def test_change_heat_drop_auto_to_unchecked(self):
        self.form.stage_number.setValue(4)
        self.form.checkBox_h0_auto.setChecked(False)
        for i in range(4):
            stage_data_widget: StageDataWidget = self.form.stackedWidget.widget(i)
            self.assertFalse(stage_data_widget.label_H0.isHidden())
            self.assertFalse(stage_data_widget.H0.isHidden())
        self.form.stage_number.setValue(1)

    def test_change_heat_drop_auto_to_checked(self):
        self.form.stage_number.setValue(4)
        self.form.checkBox_h0_auto.setChecked(False)
        self.form.checkBox_h0_auto.setChecked(True)
        for i in range(4):
            stage_data_widget: StageDataWidget = self.form.stackedWidget.widget(i)
            self.assertTrue(stage_data_widget.label_H0.isHidden())
            self.assertTrue(stage_data_widget.H0.isHidden())
        self.form.stage_number.setValue(1)

if __name__ == '__main__':
    unittest.main()