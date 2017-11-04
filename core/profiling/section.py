import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d


class BladeSection:
    def __init__(self, angle1=None, angle2=None, delta1=None, delta2=None, b_a=None,
                 gamma1_s=None, gamma1_k=None, gamma2_s=None, gamma2_k=None,
                 center_point_pos=0.5, x0_av=0, y0_av=0, pnt_count=20, r1=None, s2=0.001, convex='right'):
        """
        :param angle1: Угол потока на входе в решетку.
        :param angle2: Угол потока на выходе из решетки.
        :param delta1: Угол отставания потока на входе.
        :param delta2: Угол отставания потока на выходе.
        :param b_a: Ширина профиля в осевом направлении.
        :param gamma1_s:
        :param gamma1_k:
        :param gamma2_s:
        :param gamma2_k:
        :param center_point_pos: Позиция, заданная в относительном виде, центрального полюса средней линии профиля.
        :param x0_av: Координата входной точки средней линии.
        :param y0_av: Координата входной точки средней линии.
        :param pnt_count: Количество точек, координаты которых следует рассчитать по каждой линии.
        :param r1: Радиус скругления на входе.
        :param s2: Радиус скругления выходной кромки.
        :param convex: Направление выпуклости. Может иметь значения "right" или "left". Если параметр равен "right",
                то направление выпуклости будет правым, если смотреть по потоку.

        NOTE: Начало координат находится во входной конце средней лингии профиля. Ось x направлена по течению.
        """
        self._angle1 = angle1
        self._angle2 = angle2
        self._delta1 = delta1
        self._delta2 = delta2
        self._b_a = b_a
        self._x0_av = x0_av
        self._y0_av = y0_av
        self._gamma1_s = gamma1_s
        self._gamma1_k = gamma1_k
        self._gamma2_s = gamma2_s
        self._gamma2_k = gamma2_k
        self._center_point_pos = center_point_pos
        self._pnt_count = pnt_count
        self._r1 = r1
        self._s2 = s2
        self._convex = convex
        try:
            self._compute_coordinates()
        except AssertionError:
            pass

    @property
    def convex(self):
        assert self._convex is not None, 'convex must not be None'
        return self._convex

    @convex.setter
    def convex(self, value):
        self._convex = value
        self._compute_coordinates()

    @property
    def y0_av(self):
        assert self._y0_av is not None, 'y0 must not be None'
        return self._y0_av

    @y0_av.setter
    def y0_av(self, value):
        self._y0_av = value
        self._compute_coordinates()

    @property
    def x0_av(self):
        assert self._x0_av is not None, 'x0 must not be None'
        return self._x0_av

    @x0_av.setter
    def x0_av(self, value):
        self._x0_av = value
        self._compute_coordinates()

    @property
    def center_point_pos(self):
        assert self._center_point_pos is not None, 'center_point_pos must not be None'
        return self._center_point_pos

    @center_point_pos.setter
    def center_point_pos(self, value):
        self._center_point_pos = value
        self._compute_coordinates()

    @property
    def s2(self):
        assert self._s2 is not None, 's2 must not be None'
        return self._s2

    @s2.setter
    def s2(self, value):
        self._s2 = value
        self._compute_coordinates()

    @property
    def r1(self):
        assert self._r1 is not None, 'r1 must not be None'
        return self._r1

    @r1.setter
    def r1(self, value):
        self._r1 = value
        self._compute_coordinates()

    @property
    def pnt_count(self):
        assert self._pnt_count is not None, 'pnt_count must not be None'
        return self._pnt_count

    @pnt_count.setter
    def pnt_count(self, value):
        self._pnt_count = value
        self._compute_coordinates()

    @property
    def gamma1_k(self):
        assert self._gamma1_k is not None, 'gamma1_k must not be None'
        return self._gamma1_k

    @gamma1_k.setter
    def gamma1_k(self, value):
        self._gamma1_k = value
        self._compute_coordinates()

    @property
    def gamma1_s(self):
        assert self._gamma1_s is not None, 'gamma1_s must not be None'
        return self._gamma1_s

    @gamma1_s.setter
    def gamma1_s(self, value):
        self._gamma1_s = value
        self._compute_coordinates()

    @property
    def gamma2_k(self):
        assert self._gamma1_k is not None, 'gamma2_k must not be None'
        return self._gamma1_k

    @gamma2_k.setter
    def gamma2_k(self, value):
        self._gamma1_k = value
        self._compute_coordinates()

    @property
    def gamma2_s(self):
        assert self._gamma1_s is not None, 'gamma2_s must not be None'
        return self._gamma1_s

    @gamma2_s.setter
    def gamma2_s(self, value):
        self._gamma1_s = value
        self._compute_coordinates()

    @property
    def b_a(self):
        assert self._b_a is not None, 'b_a must not be None'
        return self._b_a

    @b_a.setter
    def b_a(self, value):
        self._b_a = value
        self._compute_coordinates()

    @property
    def delta1(self):
        """Угол отставания потока на входе."""
        assert self._delta1 is not None, 'delta1 must not be None'
        return self._delta1

    @delta1.setter
    def delta1(self, value):
        self._delta1 = value
        self._compute_coordinates()

    @property
    def delta2(self):
        """Угол отставания потока на выходе"""
        assert self._delta2 is not None, 'delta2 must not be None'
        return self._delta2

    @delta2.setter
    def delta2(self, value):
        self._delta2 = value
        self._compute_coordinates()

    @property
    def angle1(self):
        """Угол потока на входе."""
        assert self._angle1 is not None, 'angle1 must not be None'
        return self._angle1

    @angle1.setter
    def angle1(self, value):
        self._angle1 = value
        self._compute_coordinates()

    @property
    def angle2(self):
        """Угол потока на выходе."""
        assert self._angle2 is not None, 'angle2 must not be None'
        return self._angle2

    @angle2.setter
    def angle2(self, value):
        self._angle2 = value
        self._compute_coordinates()

    @property
    def angle2_l(self):
        """Угол лопатки на выходе."""
        return self.angle2 - self.delta2

    @property
    def angle1_l(self):
        """Угол лопатки на входе."""
        return self.angle1 - self.delta1

    def _get_central_line_poles(self):
        """Возвращает координаты полюсов средней линии профиля"""
        x1 = self.center_point_pos * self.b_a + self.x0_av
        y1 = self.y0_av + (x1 - self.x0_av) / np.tan(self.angle1_l)
        x2 = self.b_a + self.x0_av
        y2 = y1 - (x2 - x1) / np.tan(self.angle2_l)
        return x1, y1, x2, y2

    def _get_bezier_cyrve_coord(self, x0, y0, x1, y1, x2, y2):
        """Возвращает по полюсам координаты кривой Безье"""
        t = np.array(np.linspace(0, 1, self.pnt_count))
        x = x0 * (1 - t) ** 2 + 2 * t * (1 - t) * x1 + t ** 2 * x2
        y = y0 * (1 - t) ** 2 + 2 * t * (1 - t) * y1 + t ** 2 * y2
        return x, y

    def _get_circle_centers(self, x_av, y_av):
        """Возвращает центры окружностей скругления, 01 - вход, 02 - выход."""
        x1, y1, x2, y2 = self._get_central_line_poles()
        x01 = self.r1 + self.x0_av
        y01 = interp1d(x_av, y_av)(x01)
        x02 = x2 - self.s2
        y02 = interp1d(x_av, y_av)(x02)
        return x01, y01, x02, y02

    def _get_profile_line_poles(self, x01, y01, x02, y02, angle1, angle2, curve_type='k'):
        """
        :param x01: Координата центра окружности скругления на входе.
        :param y01: Координата центра окружности скругления на входе.
        :param x02: Координата центра окружности скругления на выходе.
        :param y02: Координата центра окружности скругления на выходе.
        :param angle1: Угол кривой профиля на входе.
        :param angle2: Угол кривой профиля на выходе.
        :param curve_type: 's' - спинка, 'k' - корыто.
        :return: tuple

        Возвращает кортеж значений координат полюсов кривой профиля
        """
        # x0, x1, x2, y0, y1, y2 - координаты полюсов кривой Безье, являющейся либо спинкой, либо корытом профиля,
        # точки с индексом 0 и 2 одновременно являются точками касания окружностей скругления

        # значение производной y по x на входе
        dir_in = np.tan(1.5 * np.pi - angle1)
        # значении производной y по x на выходе
        dir_out = np.tan(1.5 * np.pi + angle2)

        # координаты точки касания на входе и выходе
        if curve_type == 'k':
            x0 = x01 + dir_in * self.r1 / np.sqrt(1 + dir_in**2)
            y0 = y01 - np.sqrt(self.r1**2 - (x0 - x01)**2)
            x2 = x02 + dir_out * self.s2 / np.sqrt(1 + dir_out**2)
            y2 = y02 - np.sqrt(self.s2 ** 2 - (x2 - x02) ** 2)
        elif curve_type == 's':
            x0 = x01 - dir_in * self.r1 / np.sqrt(1 + dir_in ** 2)
            y0 = y01 + np.sqrt(self.r1 ** 2 - (x0 - x01) ** 2)
            x2 = x02 - dir_out * self.s2 / np.sqrt(1 + dir_out ** 2)
            y2 = y02 + np.sqrt(self.s2 ** 2 - (x2 - x02) ** 2)
        else:
            raise ValueError("Key curve_type can not be equal to %s" % curve_type)

        # координаты центрального полюса
        x1 = (y2 - y0 + x2 / np.tan(angle2) + x0 / np.tan(angle1)) / (1 / np.tan(angle1) + 1 / np.tan(angle2))
        y1 = y0 + (x1 - x0) / np.tan(angle1)
        return x0, y0, x1, y1, x2, y2

    @classmethod
    def _get_y_c(cls, x_k: np.ndarray, y_k: np.ndarray, x_s: np.ndarray,
                 y_s: np.ndarray, x_in_edge: np.ndarray, y_in_edge: np.ndarray,
                 x_out_edge: np.ndarray, y_out_edge: np.ndarray):
        """Возвращает координату центра сечения по оси y"""

        # разделение массивов координат входной кромки на две части по принадлежности к корыту и спинке
        x_s_in_edge, x_k_in_edge = np.split(x_in_edge, [list(x_in_edge).index(x_in_edge.min())])
        y_s_in_edge, y_k_in_edge = np.split(y_in_edge, [list(x_in_edge).index(x_in_edge.min())])

        # разделение массивов координат выходной кромки на две части по принадлежности к корыту и спинке
        x_k_out_edge, x_s_out_edge = np.split(x_out_edge, [list(x_out_edge).index(x_out_edge.max())])
        y_k_out_edge, y_s_out_edge = np.split(y_out_edge, [list(x_out_edge).index(x_out_edge.max())])

        # объединение массивов координат корыта и спинки
        x_k = np.array(list(x_k_in_edge) + list(x_k) + list(x_k_out_edge))
        y_k = np.array(list(y_k_in_edge) + list(y_k) + list(y_k_out_edge))
        x_s = np.array(list(x_s_in_edge) + list(x_s) + list(x_s_out_edge))
        y_s = np.array(list(y_s_in_edge) + list(y_s) + list(y_s_out_edge))

        # интерполяция
        y_k_int = interp1d(x_k, y_k)
        y_s_int = interp1d(x_s, y_s)

        # площадь
        square = quad(y_s_int, x_s.min(), x_s.max())[0] - quad(y_k_int, x_k.min(), x_k.max())[0]

        # статический момент относительно оси x
        s_x = 0.5 * (quad(lambda x: y_s_int(x)**2, x_s.min(), x_s.max())[0] -
                     quad(lambda x: y_k_int(x)**2, x_k.min(), x_k.max())[0])

        # координата y центра сечения
        y_c = s_x / square

        return y_c, square

    @classmethod
    def _get_x_c(cls, x_k: np.ndarray, y_k: np.ndarray, x_s: np.ndarray,
                 y_s: np.ndarray, x_in_edge: np.ndarray, y_in_edge: np.ndarray,
                 x_out_edge: np.ndarray, y_out_edge: np.ndarray):
        """Возвращает координату центра сечения по оси x."""
        # получение кооридинаты x центра сечения
        # разделениие массива координат выходной кромки по принадлежности к разным участкам интегрирования
        x_out_edge_01, x_out_edge_30 = np.split(x_out_edge, [list(y_out_edge).index(y_out_edge.min())])
        y_out_edge_01, y_out_edge_30 = np.split(y_out_edge, [list(y_out_edge).index(y_out_edge.min())])

        # разделение массивов координат корыта
        x_k_12, x_k_01 = np.split(x_k, [list(y_k).index(y_k.max())])
        y_k_12, y_k_01 = np.split(y_k, [list(y_k).index(y_k.max())])

        # разделение массивов координат входной кромки
        x_in_edge_23, x_in_edge_12 = np.split(x_in_edge, [list(y_in_edge).index(y_in_edge.min())])
        y_in_edge_23, y_in_edge_12 = np.split(y_in_edge, [list(y_in_edge).index(y_in_edge.min())])

        # разделение массивов координат спинки
        x_s_23, x_s_30 = np.split(x_s, [list(y_s).index(y_s.max())])
        y_s_23, y_s_30 = np.split(y_s, [list(y_s).index(y_s.max())])

        # объединение массивов

        x_01 = np.array(list(x_k_01) + list(x_out_edge_01))
        y_01 = np.array(list(y_k_01) + list(y_out_edge_01))

        x_12 = np.array(list(x_in_edge_12) + list(x_k_12))
        y_12 = np.array(list(y_in_edge_12) + list(y_k_12))

        x_23 = np.array(list(x_in_edge_23) + list(x_s_23))
        y_23 = np.array(list(y_in_edge_23) + list(y_s_23))

        x_30 = np.array(list(x_s_30) + list(x_out_edge_30))
        y_30 = np.array(list(y_s_30) + list(y_out_edge_30))

        # интерполяция
        x_01_int = interp1d(y_01, x_01)
        x_12_int = interp1d(y_12, x_12)
        x_23_int = interp1d(y_23, x_23)
        x_30_int = interp1d(y_30, x_30)

        # вычисление площади
        square = -(quad(lambda y: x_01_int(y), y_01.min(), y_01.max())[0] -
                   quad(lambda y: x_12_int(y), y_12.min(), y_12.max())[0] +
                   quad(lambda y: x_23_int(y), y_23.min(), y_23.max())[0] -
                   quad(lambda y: x_30_int(y), y_30.min(), y_30.max())[0])

        # статический момент относительно оси y
        s_y = -0.5 * (quad(lambda y: x_01_int(y)**2, y_01.min(), y_01.max())[0] -
                      quad(lambda y: x_12_int(y)**2, y_12.min(), y_12.max())[0] +
                      quad(lambda y: x_23_int(y)**2, y_23.min(), y_23.max())[0] -
                      quad(lambda y: x_30_int(y)**2, y_30.min(), y_30.max())[0])

        x_c = s_y / square

        return x_c, square

    def _compute_coordinates(self):
        self.x1_av, self.y1_av, self.x2_av, self.y2_av = self._get_central_line_poles()

        # координаты средней линии  профиля
        self.x_av, self.y_av = self._get_bezier_cyrve_coord(self.x0_av, self.y0_av, self.x1_av, self.y1_av,
                                                            self.x2_av, self.y2_av)
        self.x01, self.y01, self.x02, self.y02 = self._get_circle_centers(self.x_av, self.y_av)

        # полюса корыта
        self.x0_k, self.y0_k, self.x1_k, self.y1_k, \
        self.x2_k, self.y2_k = self._get_profile_line_poles(self.x01,
                                                            self.y01,
                                                            self.x02,
                                                            self.y02,
                                                            self.angle1_l + self.gamma1_k,
                                                            self.angle2_l + self.gamma2_k,
                                                            curve_type='k')

        # полюса спинки
        self.x0_s, self.y0_s, self.x1_s, self.y1_s, \
        self.x2_s, self.y2_s = self._get_profile_line_poles(self.x01,
                                                            self.y01,
                                                            self.x02,
                                                            self.y02,
                                                            self.angle1_l - self.gamma1_s,
                                                            self.angle2_l - self.gamma2_s,
                                                            curve_type='s')

        # координаты корыта
        self.x_k, self.y_k = self._get_bezier_cyrve_coord(self.x0_k, self.y0_k, self.x1_k, self.y1_k, self.x2_k,
                                                          self.y2_k)
        # координаты спинки
        self.x_s, self.y_s = self._get_bezier_cyrve_coord(self.x0_s, self.y0_s, self.x1_s, self.y1_s, self.x2_s,
                                                          self.y2_s)
        # координаты передней кромки
        phi1 = np.arctan((self.y0_s - self.y01) / (self.x0_s - self.x01)) + np.pi
        if np.arctan((self.y0_k - self.y01) / (self.x0_k - self.x01)) < 0:
            phi2 = np.arctan((self.y0_k - self.y01) / (self.x0_k - self.x01)) + 2 * np.pi
        else:
            phi2 = np.arctan((self.y0_k - self.y01) / (self.x0_k - self.x01)) + np.pi
        phi = np.linspace(phi1, phi2, int(self.pnt_count / 2))
        self.x_in_edge = self.x01 + self.r1 * np.cos(phi)
        self.y_in_edge = self.y01 + self.r1 * np.sin(phi)

        # координаты выходной кромки
        phi1 = np.arctan((self.y2_k - self.y02) / (self.x2_k - self.x02)) + np.pi
        phi2 = np.arctan((self.y2_s - self.y02) / (self.x2_s - self.x02)) + 2 * np.pi
        phi = np.linspace(phi1, phi2, int(self.pnt_count / 2))
        self.x_out_edge = self.x02 + self.s2 * np.cos(phi)
        self.y_out_edge = self.y02 + self.s2 * np.sin(phi)

        self.y_c, self.square_y = self._get_y_c(self.x_k, self.y_k, self.x_s, self.y_s, self.x_in_edge,
                                 self.y_in_edge, self.x_out_edge, self.y_out_edge)
        self.x_c, self.square_x = self._get_x_c(self.x_k, self.y_k, self.x_s, self.y_s, self.x_in_edge,
                                 self.y_in_edge, self.x_out_edge, self.y_out_edge)

        if self.convex == 'left':
            self.x1_av, self.y1_av, self.x2_av, self.y2_av = self.x1_av, -self.y1_av, self.x2_av, -self.y2_av
            self.x_av, self.y_av = self.x_av, - self.y_av
            self.x01, self.y01, self.x02, self.y02 = self.x01, -self.y01, self.x02, -self.y02
            self.x0_k, self.y0_k, self.x1_k, self.y1_k, self.x2_k, self.y2_k = self.x0_k, -self.y0_k, self.x1_k, \
                                                                               -self.y1_k, self.x2_k, -self.y2_k
            self.x0_s, self.y0_s, self.x1_s, self.y1_s, self.x2_s, self.y2_s = self.x0_s, -self.y0_s, self.x1_s, \
                                                                               -self.y1_s, self.x2_s, -self.y2_s
            self.x_k, self.y_k = self.x_k, -self.y_k
            self.x_s, self.y_s = self.x_s, -self.y_s
            self.y_in_edge = -self.y_in_edge
            self.y_out_edge = -self.y_out_edge
            self.y_c = -self.y_c

    def plot(self):
        plt.plot(self.y_av, self.x_av, lw=0.5, ls='--', color='black')
        plt.plot(self.y_s, self.x_s, lw=1, color='red')
        plt.plot(self.y_k, self.x_k, lw=1, color='red')
        plt.plot(self.y_in_edge, self.x_in_edge, lw=1, color='red')
        plt.plot(self.y_out_edge, self.x_out_edge, lw=1, color='red')
        plt.plot([self.y0_av, self.y1_av, self.y2_av], [self.x0_av, self.x1_av, self.x2_av], lw=0.5, ls=':',
                 color='blue', marker='o', ms=2)
        plt.plot([self.y0_k, self.y1_k, self.y2_k], [self.x0_k, self.x1_k, self.x2_k], lw=0.5, ls=':',
                 color='blue', marker='o', ms=2)
        plt.plot([self.y0_s, self.y1_s, self.y2_s], [self.x0_s, self.x1_s, self.x2_s], lw=0.5, ls=':',
                 color='blue', marker='o', ms=2)
        plt.plot([self.y_c], [self.x_c], linestyle='', marker='o', ms=8, mfc='black', color='red')
        plt.grid()
        plt.show()

if __name__ == '__main__':
    bs = BladeSection(angle1=np.radians([25])[0],
                      angle2=np.radians([30])[0],
                      delta1=np.radians([1])[0],
                      delta2=np.radians([2])[0],
                      b_a=0.03,
                      gamma1_k=np.radians([4.5])[0],
                      gamma1_s=np.radians([4.5])[0],
                      gamma2_k=np.radians([2.5])[0],
                      gamma2_s=np.radians([2.5])[0],
                      r1=0.002,
                      convex='right',
                      pnt_count=30,
                      s2=0.0003)
    print('y_c = %.5f' % bs.y_c)
    bs.plot()